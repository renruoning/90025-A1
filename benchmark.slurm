#!/bin/bash
#SBATCH --job-name=bpe_benchmark
#SBATCH --partition=sapphire
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --exclusive

set -euo pipefail

INPUT="${1:-${INPUT:-data.txt}}"
THREADS="${THREADS:-1 2 4 8 16 32}"

RESULT="results.csv"
PARTITION="${SLURM_JOB_PARTITION:-unknown}"
VERSION="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
MD5="$(md5sum "$INPUT" | awk '{print $1}')"

printf 'filename,file_md5,version,partition,threads,walltime,ram,output_md5\n' > "$RESULT"
for t in $THREADS; do
  if command -v /usr/bin/time >/dev/null 2>&1; then
    TIMING="$(OMP_NUM_THREADS="$t" /usr/bin/time -f '%e %M' ./bin/bpe "$INPUT" \
      2>&1 >/dev/null | tail -n 1)"
    WALLTIME="$(echo "$TIMING" | awk '{print $1}')"
    RAM="$(echo "$TIMING" | awk '{print $2}')"
  else
    START="$(date +%s.%N)"
    OMP_NUM_THREADS="$t" ./bin/bpe "$INPUT" >/dev/null 2>&1
    END="$(date +%s.%N)"
    WALLTIME="$(awk -v s="$START" -v e="$END" 'BEGIN{printf "%.3f", e-s}')"
    RAM="0"
  fi
  RAM="$(awk -v kb="$RAM" 'BEGIN{printf "%.1f", kb/1024}')"
  OUTPUT_MD5="$(md5sum output.txt | awk '{print $1}')"
  printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$INPUT" "$MD5" "$VERSION" "$PARTITION" "$t" "$WALLTIME" "$RAM" "$OUTPUT_MD5" \
    >> "$RESULT"
done

echo "wrote $RESULT"
cat "$RESULT"
