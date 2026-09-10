#!/bin/bash
# PRE-PURGE BACKUP — run this the moment Kaya is reachable again.
#
# UWA HPC notice (Chris Bording, 2nd reminder): the /scratch purge policy is enforced
# **Friday 11 September at 1200**. Anything left on /scratch for more than 21 days is deleted.
# /group/pmc099/pprasad is the 1 TB DURABLE filesystem and is the correct place for this.
#
# WHAT IS ACTUALLY AT RISK (irreplaceable, expensive to regenerate):
#   * MM-WLAuslan DWPose clips  — 45,010 npz, extracted 2026-08-12 (~27 days old). A multi-DAY
#     GPU extraction. Losing these kills every stitching experiment and the ISLR line.
#   * Auslan-Daily DWPose clips — 14,041 npz, extracted 2026-08-08 (~31 days old). This is the
#     extraction that broke the project's plateau (2.87 -> 9.63). Losing it kills EVERYTHING.
#   * manifests — tiny, but every job path depends on them.
# The raw source datasets are mirrored in Google Drive, but these DERIVED extractions are not:
# they were computed on Kaya and exist nowhere else.
#
# NOT DOING: `touch`-ing files to reset mtime. That evades an institutional policy on shared
# infrastructure rather than managing the data, which is what the notice actually asks for.
set -u
G=/group/pmc099/pprasad
S=/scratch/pmc099/pprasad
DEST="$G/scratch_backup"

echo "=== free space on the durable filesystem ==="
df -h "$G" | tail -1
echo
echo "=== sizes + ages on scratch (purge threshold: 21 days) ==="
now=$(date +%s)
for d in "$S/dwpose/manifests" "$S/dwpose/clips" "$S/mmwl/dwpose_clips" "$S/mmwl/labels" \
         "$S/stitch_v2" "$S/stitch_v3" "$S/dwpose/runs"; do
  [ -d "$d" ] || { printf "  %-42s (absent)\n" "$d"; continue; }
  o=$(find "$d" -type f -printf '%T@\n' 2>/dev/null | sort -n | head -1)
  n=$(find "$d" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1)
  [ -z "${o:-}" ] && { printf "  %-42s (empty)\n" "$d"; continue; }
  printf "  %-42s %8s  oldest %3dd  newest %3dd%s\n" "$d" \
    "$(du -sh "$d" 2>/dev/null | cut -f1)" $(( (now - ${o%.*}) / 86400 )) $(( (now - ${n%.*}) / 86400 )) \
    "$([ $(( (now - ${o%.*}) / 86400 )) -ge 21 ] && echo '   <-- AT RISK')"
done

echo
read -r -p "Proceed with rsync to $DEST ? [y/N] " ok
[ "$ok" = "y" ] || { echo "aborted — nothing copied"; exit 0; }

mkdir -p "$DEST"
# Order matters: the two extractions are the irreplaceable items, so they go first in case the
# filesystem fills or the session drops. Manifests are tiny and come along immediately.
for src in "$S/dwpose/manifests" "$S/mmwl/labels" "$S/mmwl/dwpose_clips" "$S/dwpose/clips"; do
  [ -d "$src" ] || continue
  echo "=== rsync $src ==="
  rsync -a --info=progress2 --partial "$src" "$DEST/$(basename "$(dirname "$src")")/" || {
      echo "!! rsync FAILED for $src — stopping so the failure is not silent"; exit 1; }
done

echo
echo "=== verify: file counts must match ==="
for pair in "$S/mmwl/dwpose_clips:$DEST/mmwl/dwpose_clips" "$S/dwpose/clips:$DEST/dwpose/clips"; do
  a="${pair%%:*}"; b="${pair##*:}"
  [ -d "$a" ] || continue
  na=$(find "$a" -name '*.npz' | wc -l); nb=$(find "$b" -name '*.npz' 2>/dev/null | wc -l)
  printf "  %-34s source %6d  backup %6d  %s\n" "$(basename "$a")" "$na" "$nb" \
    "$([ "$na" = "$nb" ] && echo OK || echo 'MISMATCH — re-run')"
done
echo
df -h "$G" | tail -1
echo "BACKUP_DONE"
