#!/bin/bash
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓

# Benchmarks the four graded configurations and writes results.csv.
#
# ALL RANKS OF A CONFIGURATION RUN ON ONE NODE, on different cores:
#
#   nodes  ranks  OpenMP threads per rank
#   -----  -----  -----------------------
#     1      4    4,4,4,4      uniform
#     1      3    4,4,8        imbalanced
#     1      3    2,6,8        imbalanced
#     1      3    2,4,10       extreme
#
# A step asks for --cpus-per-task = the largest count in its tuple, for every
# rank, so the allocation must hold ranks x largest: {2,4,10} needs 3 x 10 = 30
# CPUs while running 16 threads. Sizing it for 16 makes srun refuse that step.
#
# Launch convention: the tuple reaches the program in BPE_THREADS and each rank
# takes its count from it (rule R1.4); OMP_NUM_THREADS is exported as a fallback
# for the uniform tuple only. If your interface differs, change the run below and
# document it in README.md -- the teaching team runs the command you document.
#
# Usage:  sbatch benchmark.slurm <corpus>     (from the repository root)
#         REPS=5 sbatch benchmark.slurm corpora/8G.txt

set -euo pipefail

▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
▓▓▓▓▓▓▓▓▓▓
▓▓▓▓

INPUT="${1:-${INPUT:-data.txt}}"
REPS="${REPS:-5}"   # five runs per configuration, as the specification requires

# The four graded configurations: "<nodes> <ranks> <threads per rank>".
CONFIGS=(
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
▓▓▓▓▓▓▓▓▓▓▓▓▓
▓▓▓▓▓▓▓▓▓▓▓▓▓
▓▓▓▓▓▓▓▓▓▓▓▓▓▓
)

# The allocation has to hold the widest step. Checked here rather than left to
# srun, because srun refuses one step at a time and the summary at the end would
# still print.
ALLOC_CPUS="${SLURM_CPUS_PER_TASK:-0}"
for config in "${CONFIGS[@]}"; do
  read -r nodes ranks tuple <<< "$config"
  max_threads="$(awk -F, '{m=0; for (i=1;i<=NF;i++) if ($i+0>m) m=$i+0; print m}' <<< "$tuple")"
  if [ $((ranks * max_threads)) -gt "$ALLOC_CPUS" ]; then
    echo "allocation has $ALLOC_CPUS CPUs but the {$tuple} step needs $((ranks * max_threads))" >&2
    echo "raise --cpus-per-task in this script's SBATCH header" >&2
    exit 1
  fi
done

# Every step below runs on this node. Logged, so a run that somehow landed
# elsewhere is visible in the log rather than inferred from the timings.
echo "node: $(hostname)   partition: ${SLURM_JOB_PARTITION:-unknown}"
echo "allocated CPUs: ${SLURM_CPUS_PER_TASK:-unknown}, memory: ${SLURM_MEM_PER_NODE:-unknown}M"

RESULT="results.csv"
PARTITION="${SLURM_JOB_PARTITION:-unknown}"
VERSION="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
FILE_MD5="$(md5sum "$INPUT" | awk '{print $1}')"

printf 'filename,file_md5,version,partition,nodes,ranks,omp_threads,walltime,ram,output_md5\n' > "$RESULT"

# Wall time (HH:MM:SS) and peak RAM (MB) of the most recent srun step, as SLURM
# reports them. sacct can lag a moment behind a finished step, so retry briefly.
step_time_and_ram() {
  local row elapsed maxrss attempt
  row=""
  for attempt in 1 2 3 4 5; do
    row="$(sacct -j "$SLURM_JOB_ID" -P -n --format=JobID,Elapsed,MaxRSS 2>/dev/null \
           | awk -F'|' -v job="$SLURM_JOB_ID" '$1 ~ "^" job "\\." && $1 !~ /\.(batch|extern)$/ {last = $0} END {print last}')"
    elapsed="$(echo "$row" | cut -d'|' -f2)"
    maxrss="$(echo "$row" | cut -d'|' -f3)"
    [ -n "$elapsed" ] && [ -n "$maxrss" ] && break
    sleep 1
  done
  [ -n "$elapsed" ] || elapsed="00:00:00"
  case "$maxrss" in
    *K) maxrss="$(awk -v kb="${maxrss%K}" 'BEGIN{printf "%.1f", kb/1024}')" ;;
    *M) maxrss="${maxrss%M}" ;;
    *G) maxrss="$(awk -v g="${maxrss%G}" 'BEGIN{printf "%.1f", g*1024}')" ;;
    *) maxrss="0" ;;
  esac
  printf '%s %s\n' "$elapsed" "$maxrss"
}

for config in "${CONFIGS[@]}"; do
  read -r nodes ranks tuple <<< "$config"
  max_threads="$(awk -F, '{m=0; for (i=1;i<=NF;i++) if ($i+0>m) m=$i+0; print m}' <<< "$tuple")"
  # Semicolons, not commas: the tuple is one CSV field (rule: results.csv).
  tuple_csv="${tuple//,/;}"
  # A uniform configuration may fall back to OMP_NUM_THREADS; an imbalanced one
  # must not, because the ranks run different counts (rule R1.4).
  uniform="$(awk -F, '{for (i=2;i<=NF;i++) if ($i != $1) {print "no"; exit}} END{print "yes"}' <<< "$tuple")"
  for rep in $(seq 1 "$REPS"); do
    echo "== nodes=$nodes ranks=$ranks omp_threads=$tuple rep=$rep =="
    export BPE_THREADS="$tuple"
    if [ "$uniform" = "yes" ]; then
      export OMP_NUM_THREADS="${tuple%%,*}"
    else
      unset OMP_NUM_THREADS
    fi
    start="$(date +%s)"
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
         --time=00:20:00 ./bin/bpe "$INPUT" >/dev/null
    fallback="$(awk -v s="$start" -v e="$(date +%s)" 'BEGIN{
      d = e - s; printf "%02d:%02d:%02d", d/3600, (d%3600)/60, d%60}')"
    read -r walltime ram <<< "$(step_time_and_ram)"
    if [ "$walltime" = "00:00:00" ]; then
      walltime="$fallback"
    fi
    output_md5="$(md5sum output.txt | awk '{print $1}')"
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "$INPUT" "$FILE_MD5" "$VERSION" "$PARTITION" "$nodes" "$ranks" "$tuple_csv" \
      "$walltime" "$ram" "$output_md5" >> "$RESULT"
  done
done

echo "wrote $RESULT"
cat "$RESULT"

# ------------------------------------------------------- baseline and speed ---
# T_baseline is the wall time of the sequential reference -- bin/bpe_seq, the
# executable `make seq` builds -- on this corpus, on this machine, one node and
# one thread. The specification fixes no number for it, because the corpus used
# for marking is a different 8 GiB file and a node under load is slower for the
# sequential run and for the distributed ones alike; measuring both here, in one
# window, is what keeps those differences out of the marks.
#
# It is built without MPI, so no launcher is needed and no rank layout applies.
# BASELINE=... skips this and uses the value you give it.
if [ -z "${BASELINE:-}" ]; then
  echo
  echo "== baseline: the sequential reference, make seq =="
  make bin/bpe_seq >/dev/null 2>&1 || { echo "make bin/bpe_seq failed" >&2; exit 1; }
  base_start="$(date +%s)"
  OMP_NUM_THREADS=1 ./bin/bpe_seq "$INPUT" >/dev/null || { echo "bin/bpe_seq failed" >&2; exit 1; }
  BASELINE=$(( $(date +%s) - base_start ))
  echo "  T_baseline = ${BASELINE} s (one node, one thread, $(hostname))"
  echo "  baseline output md5 $(md5sum output.txt | awk '{print $1}')"
fi

# The per-configuration time is the MEDIAN of the five runs, as the
# specification defines it, taken from the walltime column of results.csv.
summary="$(awk -F, -v base="$BASELINE" '
  NR > 1 {
    split($8, parts, ":")                  # walltime is HH:MM:SS
    secs = parts[1] * 3600 + parts[2] * 60 + parts[3] + 0
    key = $7                               # omp_threads, e.g. 4;4;4;4
    n[key]++; v[key, n[key]] = secs
  }
  END {
    order[1] = "4;4;4;4"; order[2] = "4;4;8"
    order[3] = "2;6;8";   order[4] = "2;4;10"
    total = 0
    for (i = 1; i <= 4; i++) {
      k = order[i]
      if (n[k] == 0) { printf "  %-9s no runs\n", k; continue }
      m = n[k]
      for (a = 1; a <= m; a++) for (b = a + 1; b <= m; b++)
        if (v[k, b] < v[k, a]) { swap = v[k, a]; v[k, a] = v[k, b]; v[k, b] = swap }
      med = (m % 2) ? v[k, (m + 1) / 2] : int((v[k, m / 2] + v[k, m / 2 + 1]) / 2)
      printf "  {%s}  median %4d s   runs:", k, med
      for (a = 1; a <= m; a++) printf " %d", v[k, a]
      printf "\n"
      total += med
    }
    printf "\n  T_baseline (bin/bpe_seq)      = %d s\n", base
    printf "  sum of the four configurations = %d s\n", total
    if (total > 0) {
      S = 4 * base / total
      band = (S <= 1) ? "none" : (S <= 1.8) ? "one quarter" \
           : (S <= 2.6) ? "one half" : (S <= 3.2) ? "three quarters" : "full"
      printf "  S = 4*%d/%d = %.2f  ->  %s of the speed marks\n", base, total, S, band
    }
  }' "$RESULT")"
echo
echo "== speed-up =="
echo "$summary"
