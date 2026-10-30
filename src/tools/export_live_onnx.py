"""Export the live demo's translators to ONNX for the in-browser engine, copy the pose ONNX files, write manifest.json.

Contract: scratchpad browser_engine_contract.md (graph interfaces, file layout, manifest schema, gate G1).
Every value in the manifest is read from its source (demo/service.py MODELS, the checkpoints' saved cfg, the installed
rtmlib), never retyped. Members are built exactly as demo/service.py::Translator builds them.

  encoder.onnx  src f32 [B,T,F], src_pad bool [B,T] (+ fusion: rgb f32 [B,R,1152], rgb_pad bool [B,R])
                -> memory f32 [B,M,D], mem_pad bool [B,M]        == ensemble_decode.Member.memory
  decoder.onnx  ys i64 [B,L], memory f32 [B,M,D], mem_pad bool [B,M] -> logp f32 [B,V]   == Member.step_logp

Every exported graph has the exporter's debug metadata removed (clear_metadata: node stack traces carry this file's
absolute path; fx nodes, name scopes, the ExportedProgram signature, doc strings). Nodes, names, weights, shapes and
value_info are untouched, and write_manifest refuses any artefact that still contains a local path.

G1 (run at the end): each graph vs PyTorch, run the way the live service runs it (eval, no_grad, default fast path),
on the first 20 clips of results/_local_val_manifest.tsv. The decoder is fed PyTorch's memory and the beam prefixes
recorded in the reference n-best dumps, so it is checked at real batch sizes and prefix lengths.

  python src/tools/export_live_onnx.py                   # export + manifest + G1 (writes)
  python src/tools/export_live_onnx.py --g1_only         # G1 against existing graphs (read-only)
  python src/tools/export_live_onnx.py --compare_to DIR  # graphs vs an earlier export (read-only):
        protos equal once metadata is cleared, ORT outputs bit-identical on the G1 clips and prefixes
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import onnx
from torch.export import Dim

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'src' / 'tools'), str(ROOT)]
from config import Config                                            # noqa: E402
from slt.data import Vocab, BOS, EOS, PAD, UNK, load_pose, normalize_clip, read_manifest  # noqa: E402
from slt.preproc_guard import assert_preproc_matches_checkpoints, _PREPROC_FIELDS  # noqa: E402
from ensemble_decode import Member                                   # noqa: E402
from demo.service import MODELS, FRAME_MS, check_asset               # noqa: E402

OUT = ROOT / 'results' / 'live_onnx'
VAL = ROOT / 'results' / '_local_val_manifest.tsv'
RGB_VAL = ROOT / 'results' / 'rgb_feats_val'
DUMPS = {'single': ROOT / 'results/live_demo_2026-09-24/dump_w4A_s202_val/nbest.jsonl',
         'ensemble': ROOT / 'results/final_ensemble_validation_decodes/nbest.jsonl'}
OPSET = 18                      # the torch.export-based exporter's minimum
# Member directory names, keyed by checkpoint file; the ORDER comes from MODELS (S6: jobs/slurm/mbr_wer_test2.sbatch).
NAMES = {'w4A_s202_best.pt': 'w4A_s202', 'comm_dwpose_ctc_vlplong80.pt': 'vlp80',
         'comm_dwpose_rgb_fusion.pt': 'rgb_fusion', 'comm_dwpose_vlp_ctc.pt': 'vlp40',
         'comm_dwpose_ctc_s202.pt': 'ctc_s202', 'comm_dwpose_fus_s101.pt': 'fus_s101',
         'comm_dwpose_fus_s202.pt': 'fus_s202'}
POSE_CACHE = Path.home() / '.cache/rtmlib/hub/checkpoints'
POSE_FILES = {'detector': ('yolox_m_8xb8-300e_humanart-c2c7a14a.onnx', 'pose/yolox_m.onnx'),
              'pose': ('rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.onnx', 'pose/rtmw_256x192.onnx')}


def sha256(path) -> str:
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def home_rel(p) -> str:
    """A path for the public manifest: '~/...' under the home directory (no user name), else as given."""
    p = Path(p)
    return f'~/{p.relative_to(Path.home())}' if p.is_relative_to(Path.home()) else str(p)


LOCAL_MARKERS = [m.encode() for m in (str(Path.home()), str(ROOT), '/Users/', '/home/', '/private/')]


def assert_no_local_paths(path):
    """The artefacts are published as-is; none may carry a path from this machine."""
    data = Path(path).read_bytes()
    hits = [m.decode() for m in LOCAL_MARKERS if m in data]
    assert not hits, f'{path} contains local path markers {hits}'


def clear_metadata(m: onnx.ModelProto) -> onnx.ModelProto:
    """Drop the exporter's debug metadata in place; the computation (nodes, names, weights, value_info) is untouched."""
    g = m.graph
    assert not m.functions and not any(a.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
                                       for n in g.node for a in n.attribute), 'subgraphs present: extend clear_metadata'
    for x in (m, g, *g.node, *g.input, *g.output, *g.value_info, *g.initializer):
        del x.metadata_props[:]
        x.doc_string = ''
    return m


def strip_metadata(path):
    onnx.save(clear_metadata(onnx.load(str(path))), str(path))


def live_cfg() -> Config:
    """The decode Config demo/service.py::Translator builds (device fixed to cpu)."""
    return Config(num_keypoints=133, keypoint_subset='all', coord_norm='frame', rgb_feature_dim=1152, device='cpu')


def load_members(model: str):
    """(names, checkpoints, members) for one operating point, built as the live Translator builds them."""
    spec = MODELS[model]
    cfg = live_cfg()
    cks = [torch.load(check_asset(str(ROOT / p), sha), map_location='cpu', weights_only=False)
           for p, sha in spec['ckpts']]
    assert_preproc_matches_checkpoints(cfg, cks)
    ref = list(cks[0]['vocab'])
    assert all(list(c['vocab']) == ref for c in cks), 'members must share one vocabulary (service hard-errors otherwise)'
    members = [Member(c, cfg, Vocab(ref), torch.device('cpu')) for c in cks]
    names = [NAMES[Path(p).name] for p, _ in spec['ckpts']]
    return names, cks, members


class Encoder(torch.nn.Module):
    def __init__(self, member):
        super().__init__()
        self.model, self.member = member.model, member      # registered so weights become initializers

    def forward(self, src, src_pad, rgb=None, rgb_pad=None):
        mem, pad = self.member.memory(src, src_pad, rgb, rgb_pad)
        return mem, pad.clone() if pad is src_pad else pad  # clone: an ONNX output may not alias an input


class Decoder(torch.nn.Module):
    def __init__(self, member):
        super().__init__()
        self.model, self.member = member.model, member

    def forward(self, ys, memory, mem_pad):
        return self.member.step_logp(ys, memory, mem_pad)


def val_rows(n):
    return read_manifest(str(VAL))[:n]


def clip_inputs(clip, cfg):
    """Source features exactly as AuslanDailyDataset builds them; RGB exactly as RGBDataset._load_rgb."""
    kps, bbox = load_pose(clip)
    feats, _ = normalize_clip(kps, bbox, cfg)
    cid = Path(clip).name.removesuffix('.npz')
    with np.load(RGB_VAL / f'rgb_{cid}.npz') as z:
        rgb = np.asarray(z['FEATURES'], np.float32)
    assert rgb.ndim == 2 and rgb.shape[1] == 1152 and rgb.shape[0] <= cfg.max_src_frames
    return feats, rgb


@contextlib.contextmanager
def plain_ops():
    """Trace the plain PyTorch ops, not the dispatch shortcuts the eager model takes.

    (1) The fused CPU fast path (torch._transformer_encoder_layer_fwd / native MHA) has no ONNX form.
    (2) nn.TransformerDecoder checks at run time whether tgt_mask equals a causal mask and, when it does, hands
        SDPA is_causal=True instead of the mask. That check is data-dependent, which torch.export cannot trace
        (GuardOnDataDependentSymNode), and the legacy TorchScript exporter baked the traced prefix length into a
        Reshape. Returning the caller's hint (None -> False) makes the decoder apply the SAME boolean causal mask
        (PoseSLT.decode_step's triu) explicitly. The computation is unchanged; G1 measures the graphs against the
        eager model with both shortcuts active.
    """
    import torch.nn.modules.transformer as tr
    detect = tr._detect_is_causal_mask
    tr._detect_is_causal_mask = lambda mask, is_causal=None, size=None: bool(is_causal)
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        yield
    finally:
        tr._detect_is_causal_mask = detect
        torch.backends.mha.set_fastpath_enabled(True)


def export_member(member, name, feats, rgb):
    """torch.export-based ONNX export with symbolic B/T/R/L/M (example batch 2: torch.export specialises 0/1 sizes)."""
    d = OUT / 'translators' / name
    d.mkdir(parents=True, exist_ok=True)
    B, L = 2, 5
    src = torch.from_numpy(feats)[None].expand(B, -1, -1).contiguous()
    src_pad = torch.zeros(B, src.shape[1], dtype=torch.bool)
    Bd, Td, Rd = Dim('B', min=1, max=1024), Dim('T', min=1, max=4096), Dim('R', min=1, max=4096)
    Ld, Md = Dim('L', min=1, max=4096), Dim('M', min=1, max=8192)
    t0 = time.perf_counter()
    with plain_ops():
        if member.fusion:
            r = torch.from_numpy(rgb)[None].expand(B, -1, -1).contiguous()
            args = (src, src_pad, r, torch.zeros(B, r.shape[1], dtype=torch.bool))
            names = ['src', 'src_pad', 'rgb', 'rgb_pad']
            dyn = {'src': {0: Bd, 1: Td}, 'src_pad': {0: Bd, 1: Td}, 'rgb': {0: Bd, 1: Rd}, 'rgb_pad': {0: Bd, 1: Rd}}
        else:
            args, names = (src, src_pad), ['src', 'src_pad']
            dyn = {'src': {0: Bd, 1: Td}, 'src_pad': {0: Bd, 1: Td}}
        torch.onnx.export(Encoder(member).eval(), args, str(d / 'encoder.onnx'), input_names=names,
                          output_names=['memory', 'mem_pad'], dynamic_shapes=dyn, opset_version=OPSET,
                          dynamo=True, external_data=False, verbose=False)
        strip_metadata(d / 'encoder.onnx')
        with torch.no_grad():
            mem, pad = member.memory(*args) if member.fusion else member.memory(src, src_pad, None, None)
        ys = torch.randint(4, 50, (B, L)); ys[:, 0] = BOS
        torch.onnx.export(Decoder(member).eval(), (ys, mem, pad), str(d / 'decoder.onnx'),
                          input_names=['ys', 'memory', 'mem_pad'], output_names=['logp'],
                          dynamic_shapes={'ys': {0: Bd, 1: Ld}, 'memory': {0: Bd, 1: Md}, 'mem_pad': {0: Bd, 1: Md}},
                          opset_version=OPSET, dynamo=True, external_data=False, verbose=False)
        strip_metadata(d / 'decoder.onnx')
    return time.perf_counter() - t0


def op_types(path):
    m = onnx.load(str(path))
    onnx.checker.check_model(m)
    return sorted({n.op_type for n in m.graph.node}), [(o.domain or 'ai.onnx', o.version) for o in m.opset_import]


def rtmlib_params():
    """Pose parameters read from the INSTALLED rtmlib (the patched copy the live service runs)."""
    import rtmlib
    from rtmlib import Wholebody, YOLOX, RTMPose
    from rtmlib.version import __version__ as rtmlib_version
    yd = inspect.signature(YOLOX.__init__).parameters
    rd = inspect.signature(RTMPose.__init__).parameters
    post = inspect.signature(RTMPose.postprocess).parameters
    bal = Wholebody.MODE['balanced']
    return dict(
        rtmlib_version=rtmlib_version,
        rtmlib_path=str(Path(rtmlib.__file__).parent),
        detector=dict(input=list(bal['det_input_size']), input_order='[h, w]', pad_value=114,  # yolox.py:91 (ones*114)
                      # This graph ends in NonMaxSuppression and outputs dets [1, N, 5] (x1, y1, x2, y2, score), so
                      # rtmlib takes its NMS-in-graph branch (yolox.py:171-179): boxes / ratio, keep score > 0.3.
                      # YOLOX(score_thr, nms_thr) are applied only to graphs WITHOUT NMS; recorded for reference.
                      output='dets [1, N, 5] after in-graph NonMaxSuppression', score_gt=0.3, nms='in graph',
                      unused_yolox_defaults=dict(score_thr=yd['score_thr'].default, nms_thr=yd['nms_thr'].default),
                      source_url=bal['det']),
        pose=dict(input=list(bal['pose_input_size']), input_order='[w, h]', mean=list(rd['mean'].default),
                  std=list(rd['std'].default), simcc_split_ratio=post['simcc_split_ratio'].default,
                  bbox_padding=1.25,             # rtmlib/tools/pose_estimation/rtmpose.py preprocess
                  source_url=bal['pose']))


def write_manifest(timings, ops):
    params = rtmlib_params()
    pose = {}
    for key, (cache_name, rel) in POSE_FILES.items():
        src, dst = POSE_CACHE / cache_name, OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or sha256(dst) != sha256(src):
            shutil.copyfile(src, dst)
        h = sha256(dst)
        assert h == sha256(src), f'{rel} is not a byte copy of {src}'
        assert_no_local_paths(dst)
        pose[key] = dict(file=rel, sha256=h, bytes=dst.stat().st_size, source=home_rel(src), **params[key])
    pose['rtmlib'] = dict(version=params['rtmlib_version'], path=home_rel(params['rtmlib_path']))
    det = onnx.load(str(OUT / pose['detector']['file']))
    assert any(n.op_type == 'NonMaxSuppression' for n in det.graph.node) and \
        det.graph.output[0].type.tensor_type.shape.dim[-1].dim_value == 5, 'detector no longer has NMS in graph'

    old = json.loads((OUT / 'manifest.json').read_text()) if (OUT / 'manifest.json').exists() else {}
    old_secs = {m['name']: m.get('export_seconds') for v in old.get('models', {}).values() for m in v['members']}
    models, pre = {}, None
    for model in ('single', 'ensemble'):
        spec = MODELS[model]
        names, cks, members = load_members(model)
        mem_entries = []
        for name, (ck_path, ck_sha), ck, member in zip(names, spec['ckpts'], cks, members):
            d = OUT / 'translators' / name
            vocab_file = d / 'vocab.json'
            vocab_file.write_text(json.dumps(list(ck['vocab'])))
            files = {k: f'translators/{name}/{k}.onnx' for k in ('encoder', 'decoder')} | \
                    {'vocab': f'translators/{name}/vocab.json'}
            for v in files.values():
                assert_no_local_paths(OUT / v)
            mem_entries.append(dict(
                name=name, fusion=member.fusion, **files,
                sha256={k: sha256(OUT / v) for k, v in files.items()},
                bytes={k: (OUT / v).stat().st_size for k, v in files.items()},
                checkpoint=dict(path=ck_path, sha256=ck_sha),
                inputs=['src', 'src_pad'] + (['rgb', 'rgb_pad'] if member.fusion else []),
                ops=ops.get(name), export_seconds=timings.get(name, old_secs.get(name))))
            cfg_pre = {f: ck['cfg'].get(f) for f in _PREPROC_FIELDS + ('drop_zero_frames',) if f in ck['cfg']}
            pre = pre or cfg_pre
            assert all(pre.get(f) == v for f, v in cfg_pre.items()), f'{name}: preprocessing differs from member 0'
        models[model] = dict(label=spec['label'], beam=spec['beam'], lp=spec['lp'], rgb=spec['rgb'],
                             tau={str(q): v for q, v in sorted(spec['tau'].items())}, evidence=spec['evidence'],
                             weights='uniform', members=mem_entries)
    cfg = live_cfg()
    assert all(getattr(cfg, f) == v for f, v in pre.items()), 'the live decode Config differs from the checkpoints'
    manifest = dict(
        version=1, hf_repo='preetprasad/auslan-slt-live', opset=OPSET,
        pose=pose,
        dinov2=dinov2_entry(old.get('dinov2')),
        models=models,
        preprocess=dict(pre, keypoint_subset=cfg.keypoint_subset, frame_std_eps=1e-6,
                        feature_dim=cfg.feature_dim(), rgb_feature_dim=cfg.rgb_feature_dim,
                        frame_ms=FRAME_MS, note='normalize_clip in src/slt/data.py; coord_norm frame: '
                        'per-frame np.mean / np.std (ddof 0, float32) pooled over all x and y of that frame, std + 1e-6; BBOX all zero; features '
                        '(x, y, conf) per keypoint; all-zero frames dropped; > max_src_frames subsampled with '
                        'np.linspace(0, T-1, max).round()'),
        decode=dict(max_len=cfg.decode_max_len or cfg.max_tgt_tokens, pad=PAD, bos=BOS, eos=EOS, unk=UNK,
                    select='argmax raw / max(1, words) ** lp over the finished n-best (first wins ties)',
                    confidence='-raw / (words + 1)', text='Vocab.decode then slt/metrics.py detokenize',
                    reference='src/tools/onnx_reference_decode.py (mirror of '
                              'tools/ensemble_mbr_wer.py::ensemble_nbest)'))
    text = json.dumps(manifest, indent=2) + '\n'
    assert not any(m.decode() in text for m in LOCAL_MARKERS), 'manifest.json would publish a local path'
    (OUT / 'manifest.json').write_text(text)
    return manifest


def dinov2_entry(d):
    """manifest.dinov2 in the contract's schema. The RGB stage chose and verified the graph and wrote this key
    (demo/tests/fixtures/make_rgb_fixtures.py: Hub LFS hash, parity vs transformers pooler_output); here the file is
    re-hashed from the local Hub cache and the output is named the way rgb.js reads it."""
    assert d and {'hf_model', 'revision', 'path', 'sha256', 'bytes'} <= d.keys(), \
        'manifest.dinov2 missing: run demo/tests/fixtures/make_rgb_fixtures.py first'
    hub = Path(os.environ.get('HF_HUB_CACHE') or Path(os.environ.get('HF_HOME', Path.home() / '.cache/huggingface')) / 'hub')
    local = hub / f"models--{d['hf_model'].replace('/', '--')}" / 'snapshots' / d['revision'] / d['path']
    assert sha256(local) == d['sha256'] and local.stat().st_size == d['bytes'], f'{local} differs from manifest.dinov2'
    assert [o.name for o in onnx.load(str(local), load_external_data=False).graph.output] == ['last_hidden_state']
    d = {k: v for k, v in d.items() if k != 'pool'}
    return d | dict(output='last_hidden_state[:,0]')       # token 0 == transformers pooler_output (see parity)


def enc_feed(feats, rgb, fusion):
    feed = {'src': feats[None], 'src_pad': np.zeros((1, len(feats)), bool)}
    return feed | ({'rgb': rgb[None], 'rgb_pad': np.zeros((1, len(rgb)), bool)} if fusion else {})


def prefix_batches(row, stoi):
    """Decoder inputs from one recorded n-best: per prefix length, the hypotheses still running at that length."""
    seqs = [[BOS] + [stoi[w] for w in text.split()] + [EOS] for _, text in row['nbest']]
    for L in range(1, max(map(len, seqs))):
        yield np.array([s[:L] for s in seqs if len(s) > L], np.int64)


def compare_to(old: Path, n_clips=20):
    """Read-only: every translator graph in OUT vs the same file under `old` (e.g. a copy taken before a re-export).
    Protos must serialise identically once clear_metadata is applied to both, and ORT outputs (encoder memory and
    mem_pad; decoder logp at the recorded n-best prefixes, fed the old graph's memory) must be bit-identical."""
    import onnxruntime as ort
    manifest = json.loads((OUT / 'manifest.json').read_text())
    cfg, rows, report = live_cfg(), val_rows(n_clips), {}
    bits = lambda a, b: a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    for model, spec in manifest['models'].items():
        dump = [json.loads(l) for l in open(DUMPS[model])][:n_clips]
        for m in spec['members']:
            if m['name'] in report:
                continue
            paths = {k: (OUT / m[k], old / m[k]) for k in ('encoder', 'decoder')}
            proto = {k: clear_metadata(onnx.load(str(a))).SerializeToString(deterministic=True) ==
                     clear_metadata(onnx.load(str(b))).SerializeToString(deterministic=True) for k, (a, b) in paths.items()}
            ses = {k: [ort.InferenceSession(str(x), providers=['CPUExecutionProvider']) for x in ab]
                   for k, ab in paths.items()}
            stoi = {w: i for i, w in enumerate(json.loads((OUT / m['vocab']).read_text()))}
            same, n_runs = True, 0
            for (clip, ref), row in zip(rows, dump):
                assert row['ref'] == ref, 'dump rows must follow the manifest'
                feats, rgb = clip_inputs(clip, cfg)
                new_out, old_out = (s.run(None, enc_feed(feats, rgb, m['fusion'])) for s in ses['encoder'])
                same &= all(map(bits, new_out, old_out))
                mem, pad = old_out
                for ys in prefix_batches(row, stoi):
                    f = {'ys': ys, 'memory': np.repeat(mem, len(ys), 0), 'mem_pad': np.repeat(pad, len(ys), 0)}
                    same &= all(map(bits, *(s.run(None, f) for s in ses['decoder'])))
                    n_runs += 1
            report[m['name']] = dict(proto_equal_without_metadata=proto, outputs_bit_identical=same,
                                     clips=len(rows), decoder_batches=n_runs)
            print(f"[compare] {m['name']:10s} proto equal (metadata cleared) {proto}  outputs bit-identical {same} "
                  f'({len(rows)} clips, {n_runs} decoder batches)', flush=True)
    return report


def g1(n_clips=20):
    """Max |ORT - PyTorch| per graph. PyTorch runs as the service runs it (eval, no_grad, default fast path)."""
    import onnxruntime as ort
    cfg = live_cfg()
    rows = val_rows(n_clips)
    report = {}
    for model in ('single', 'ensemble'):
        names, cks, members = load_members(model)
        vocab = Vocab(list(cks[0]['vocab']))
        dump = [json.loads(l) for l in open(DUMPS[model])][:n_clips]
        for name, member in zip(names, members):
            if name in report:
                continue
            d = OUT / 'translators' / name
            enc = ort.InferenceSession(str(d / 'encoder.onnx'), providers=['CPUExecutionProvider'])
            dec = ort.InferenceSession(str(d / 'decoder.onnx'), providers=['CPUExecutionProvider'])
            e_max = l_max = mag = 0.0
            pad_ok, n_dec = True, 0
            for (clip, ref), row in zip(rows, dump):
                assert row['ref'] == ref, 'dump rows must follow the manifest'
                feats, rgb = clip_inputs(clip, cfg)
                src = torch.from_numpy(feats)[None]
                src_pad = torch.zeros(1, src.shape[1], dtype=torch.bool)
                r = torch.from_numpy(rgb)[None]
                r_pad = torch.zeros(1, r.shape[1], dtype=torch.bool)
                with torch.no_grad():
                    mem, pad = member.memory(src, src_pad, r, r_pad)
                o_mem, o_pad = enc.run(None, enc_feed(feats, rgb, member.fusion))
                e_max = max(e_max, float(np.abs(o_mem - mem.numpy()).max()))
                mag = max(mag, float(mem.abs().max()))
                pad_ok &= bool(np.array_equal(o_pad, pad.numpy()))
                # decoder: the recorded n-best prefixes, batched per prefix length
                for ys in prefix_batches(row, vocab.stoi):
                    Bt = len(ys)
                    m_b = mem.expand(Bt, -1, -1).contiguous()
                    p_b = pad.expand(Bt, -1).contiguous()
                    with torch.no_grad():
                        lp = member.step_logp(torch.from_numpy(ys), m_b, p_b).numpy()
                    (o_lp,) = dec.run(None, {'ys': ys, 'memory': m_b.numpy(), 'mem_pad': p_b.numpy()})
                    l_max = max(l_max, float(np.abs(o_lp - lp).max()))
                    n_dec += 1
            report[name] = dict(encoder_max_abs=e_max, memory_max_magnitude=mag, mem_pad_equal=pad_ok, decoder_logp_max_abs=l_max,
                                decoder_batches=n_dec, clips=len(rows))
            print(f'[G1] {name:10s} encoder max|d|={e_max:.3e} (max|memory| {mag:.2f})  mem_pad equal={pad_ok}  '
                  f'decoder logp max|d|={l_max:.3e} over {n_dec} batches', flush=True)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--g1_only', action='store_true', help='G1 on the existing graphs; writes nothing')
    ap.add_argument('--compare_to', type=Path, help='compare the existing graphs with DIR/translators; writes nothing')
    ap.add_argument('--g1_clips', type=int, default=20)
    a = ap.parse_args()
    if a.g1_only or a.compare_to:
        report = g1(a.g1_clips) if a.g1_only else compare_to(a.compare_to, a.g1_clips)
        print(json.dumps(report, indent=2))
        return
    cfg = live_cfg()
    timings, ops = {}, {}
    feats, rgb = clip_inputs(val_rows(1)[0][0], cfg)
    for model in ('single', 'ensemble'):
        names, _, members = load_members(model)
        for name, member in zip(names, members):
            timings[name] = round(export_member(member, name, feats, rgb), 2)
            print(f'[export] {name:10s} fusion={member.fusion} {timings[name]:.2f}s', flush=True)
    for name in NAMES.values():
        for part in ('encoder', 'decoder'):
            types, opsets = op_types(OUT / 'translators' / name / f'{part}.onnx')
            ops.setdefault(name, {})[part] = types
        ops[name]['opset_import'] = opsets
    manifest = write_manifest(timings, ops)
    all_ops = sorted({t for v in ops.values() for p in ('encoder', 'decoder') for t in v[p]})
    print('[ops] union over all translator graphs:', ', '.join(all_ops))
    report = g1(a.g1_clips)
    (OUT / 'g1_report.json').write_text(json.dumps(dict(opset=OPSET, clips=a.g1_clips, graphs=report,
                                                        export_seconds=timings, ops_union=all_ops),
                                                   indent=2) + '\n')
    print('[manifest]', OUT / 'manifest.json', 'models:', list(manifest['models']))


if __name__ == '__main__':
    main()
