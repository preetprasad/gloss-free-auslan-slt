"""Extract GloFE's public WLASL-pretrained pose features from Auslan HALPE-136.

The ACL 2023 GloFE release provides a CTR-GCN + multi-scale temporal-convolution
pose backbone and its isolated-sign checkpoint.  This tool deliberately imports
that public implementation instead of copying/reinterpreting it.  It converts
Auslan-Daily's AlphaPose HALPE-136 tensors into the 76-joint ordering used by
GloFE, applies the released inference normalisation, and stores the four-region
pooled ``(ceil(T/4), 1024)`` feature sequence in an NPZ file.

The only reconstruction is the layout adapter.  COCO-WholeBody and HALPE share
the conventional 68-landmark face order, so a COCO face index ``23+i`` maps to
HALPE ``26+i``.  Hands are placed in the *released GloFE loader's actual tensor
order* (body, COCO 112:133 hand, COCO 91:112 hand, face), even though its local
variable names appear reversed.  The assumption is recorded in every artifact.

Example:
    python tools/extract_glofe_features.py \
      --input_manifest ../comm_run/overfit200.tsv \
      --output_manifest ../glofe_overfit200/manifest.tsv \
      --output_dir ../glofe_overfit200/features \
      --glofe_repo ../tmp/reference_repos/GloFE
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch


GLOFE_FACE_COCO_GLOBAL = (
    [71, 77, 85, 89]
    + [40, 42, 44, 45, 47, 49]
    + list(range(59, 65))
    + list(range(65, 71))
    + [50]
)
GLOFE_FACE_HALPE = [index + 3 for index in GLOFE_FACE_COCO_GLOBAL]


def read_manifest(path: Path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line and "\t" in line:
                rows.append(tuple(line.split("\t", 1)))
    return rows


def load_halpe(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        with np.load(path) as payload:
            keypoints = np.asarray(payload["KPS"], dtype=np.float32)
    else:
        value = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(value, dict):
            value = value.get("KPS", value.get("kps"))
        keypoints = (value.detach().cpu().numpy() if torch.is_tensor(value)
                     else np.asarray(value, dtype=np.float32))
    if keypoints.ndim != 3 or keypoints.shape[1] != 136 or keypoints.shape[2] < 3:
        raise ValueError(f"expected HALPE KPS (T,136,3+), got {keypoints.shape} in {path}")
    return keypoints[..., :3]


def halpe_to_glofe76(keypoints: np.ndarray, cap: int = 512) -> tuple[np.ndarray, int]:
    """Return released-GloFE-normalised ``(cap,76,3)`` and valid length."""
    frame_valid = np.abs(keypoints).reshape(keypoints.shape[0], -1).sum(1) > 1e-6
    keypoints = keypoints[frame_valid]
    if len(keypoints) == 0:
        keypoints = np.zeros((1, 136, 3), dtype=np.float32)
    keypoints = keypoints[:cap]
    valid_length = len(keypoints)

    # Match train_openasl_pose_DDP_inter_VN.py exactly.  In that released
    # loader the first hand after body comes from COCO 112:133 and the second
    # from 91:112; hence HALPE right then left here.
    pose = np.concatenate(
        [
            keypoints[:, :11],
            keypoints[:, 115:136],
            keypoints[:, 94:115],
            keypoints[:, GLOFE_FACE_HALPE],
        ],
        axis=1,
    ).astype(np.float32, copy=True)
    if pose.shape[1:] != (76, 3):
        raise AssertionError(pose.shape)

    # Released preprocessing first uses this fixed 256 divisor, then pads and
    # applies clip-wise min/max scaling.  Preserve even the redundant affine
    # step so the checkpoint sees its training-time convention.
    pose[..., :2] = 2.0 * ((pose[..., :2] / 256.0) - 0.5)
    padded = np.zeros((cap, 76, 3), dtype=np.float32)
    padded[:valid_length] = pose
    xy = padded[..., :2].reshape(-1, 2)
    xy = (xy - xy.min(axis=0)) / (xy.max(axis=0) - xy.min(axis=0) + 1e-5)
    padded[..., :2] = (xy * 2.0 - 1.0).reshape(cap, 76, 2)
    return padded, valid_length


def build_backbone(repo: Path, checkpoint: Path, device: torch.device):
    sys.path.insert(0, str(repo.resolve()))
    from models.ctrgcn_base_p76 import Model  # type: ignore

    model = Model(
        num_class=2000,
        num_point=76,
        num_person=1,
        graph="models.graph.mmpose_p76.Graph",
        graph_args={"labeling_mode": "spatial"},
        drop_out=0,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(incompatible)
    return model.eval().to(device)


def pooled_features(backbone, pose: np.ndarray, valid_length: int,
                    device: torch.device) -> np.ndarray:
    tensor = torch.from_numpy(pose).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.inference_mode():
        encoded = backbone(tensor)  # (1,256,128,76) for a 512-frame input
        regions = torch.cat(
            [
                encoded[..., :11].mean(-1, keepdim=True),
                encoded[..., 11:32].mean(-1, keepdim=True),
                encoded[..., 32:53].mean(-1, keepdim=True),
                encoded[..., 53:].mean(-1, keepdim=True),
            ],
            dim=-1,
        )
        features = regions.permute(0, 2, 3, 1).reshape(1, regions.shape[2], -1)
        output_length = min(features.shape[1], math.ceil(valid_length / 4))
        features = features[0, :output_length].float().cpu().numpy()
    if features.shape[1] != 1024 or not np.isfinite(features).all():
        raise ValueError(f"invalid GloFE features {features.shape}")
    return features


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_manifest", required=True)
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--glofe_repo", default="tmp/reference_repos/GloFE")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--clip_limit", type=int, default=0)
    args = parser.parse_args()

    repo = Path(args.glofe_repo).resolve()
    checkpoint = (Path(args.checkpoint).resolve() if args.checkpoint else
                  repo / "models/ckpt/ctr_76_e1/runs-88-100144.pt")
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_manifest = Path(args.output_manifest).resolve()
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    backbone = build_backbone(repo, checkpoint, device)

    rows = read_manifest(Path(args.input_manifest))
    if args.clip_limit > 0:
        rows = rows[:args.clip_limit]
    converted = []
    for index, (source, text) in enumerate(rows, 1):
        source_path = Path(source).resolve()
        digest = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:12]
        destination = output_dir / f"{source_path.stem}-{digest}.npz"
        if not destination.exists():
            keypoints = load_halpe(source_path)
            pose, valid_length = halpe_to_glofe76(keypoints)
            features = pooled_features(backbone, pose, valid_length, device)
            np.savez_compressed(
                destination,
                FEATURES=features,
                SOURCE_FRAMES=np.int32(valid_length),
                SOURCE_PATH=np.asarray(str(source_path)),
                PIPELINE=np.asarray("GloFE ACL2023 public CTR-GCN WLASL checkpoint"),
                LAYOUT_ASSUMPTION=np.asarray(
                    "HALPE face26+i=COCO face23+i; released GloFE hand tensor order"
                ),
            )
        converted.append((str(destination), text))
        if index == 1 or index % 25 == 0 or index == len(rows):
            print(f"[glofe] {index}/{len(rows)} -> {destination.name}", flush=True)

    temporary = output_manifest.with_suffix(output_manifest.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for path, text in converted:
            handle.write(f"{path}\t{text}\n")
    os.replace(temporary, output_manifest)
    print(f"[done] {len(converted)} clips | manifest={output_manifest} | device={device}")


if __name__ == "__main__":
    main()
