# COMP90025 2026 Project 1 — BPE skeleton

This is the provided skeleton for the byte-pair encoding (BPE) project. It
contains the sequential reference implementation of every task, the
`Results` data structure that holds each task's results (rule R6.1), and a
Makefile. Your only task is to replace the two parallel API entries in
`src/parallel.cpp` with OpenMP implementations (rule R7.2), namely:
1. `parallel_task1` including `split_words`
2. `parallel_task2`

## Layout

```
src/bpe.h          public API + the Results data structure (rule R6.1)
src/corpus.cpp     input reading + word splitting (provided)
src/task1.cpp      Task 1 sequential reference: word counts + character splits (provided)
src/task2.cpp      Task 2 sequential reference: BPE merge loop (provided, self-contained)
src/output.cpp        Task 2 output.txt writing + Task 1 stdout dump (provided)
src/pipeline.cpp   pipeline + CLI (provided)
src/parallel.cpp   update the two parallel API entries (you can create new files in src/ if you want, but do not modify the provided sequential reference)
abseil/            vendored Abseil logging subset (provided; see "Logging")
Makefile
```

## Build and run

```
make              # builds bin/bpe
make run INPUT=file.txt          # run bin/bpe on a corpus
make bench INPUT=file.txt        # wall-time the run with 1 2 4 8 16 32 OpenMP threads
make clean
```

`bin/bpe input.txt` prints walltimes and writes `output.txt`
in the current directory. `output.txt` has one `token count` line per token,
ordered by decreasing count, ties by the lexicographically smallest token
first. For a fixed input the output is byte-exact, and the evaluation compares
your output against the sequential reference and the spec's expected output.


## Style expectations (Google C++, format-only)

Style is **not marked**. These targets are advisory and never
fail a build:

```
make format   # apply Google C++ formatting to all C++ files (auto-fix)
make check    # advisory: report whether all files already match (exits 0)
```

The repository ships a `.clang-format` derived from the Google C++ Style Guide
(clang-format 14.0.6), tuned to the skeleton's 4-space indentation. Running
`make format` before committing is recommended so everyone's code reads the
same way — it is purely mechanical and cannot change behaviour.

## Reading and splitting

`read_file` loads the file into memory in a single read. `split_words` then
splits it in place: it overwrites each whitespace byte with a NUL byte and
returns the word starts; each `Word` is a pointer to a NUL-terminated byte
string into the input buffer (it no longer owns its bytes). The input buffer
must outlive the words that point into it (the pipeline owns both; `Results`
copies each distinct word's bytes, so the results stay valid once the buffer is
gone). The input is assumed to be ASCII text, so no word contains a NUL byte.
Do not call `split_words` twice on the same buffer.

## Logging (Abseil)

The CLI uses the Abseil logging library (`absl::log`). It writes progress
(`LOG(INFO)`) and diagnostics (`LOG(ERROR)`) to **stderr**; stdout carries only
the Task 1 output, so the byte-exact output contract is unaffected. The
`abseil/` directory is a vendored subset of Abseil, trimmed to exactly the
files the skeleton's logging needs. See more details of Abseil logging at https://abseil.io/docs/cpp/guides/logging.

## What you must implement (rule R7.2)

Until you implement the parallel versions, `parallel_task1` and `parallel_task2`
call the sequential reference, so the program is correct but shows no speed-up. A purely sequential implementation scores 0 for correctness and 0 for
speed on Tasks 1 and 2 (rule R7.2), even you introduce your own sequential implementation with speedup. The provided sequential reference is there only for you to understand the problem and to verify your parallel implementation.

You may extend the API in `src/bpe.h` or add new source files in `src/` (the
Makefile compiles `src/*.cpp` automatically). Do not modify the sequential
reference or the output format, because the evaluation compares your parallel result
against them. You are allowed to add new logging statements for debugging, but do not remove the existing ones. During assessments, we will overwrite thoes sequential reference files with the provided skeleton versions, so any changes you make to them will be lost. ***Do not try to fake performance logs; doing so will result you in zero marks of this assignment.***

## Notes
- The parallel version `does not have to` be based on the sequential reference provided by this skeleton. It means you can implement your own parallel version from scratch, as long as it meets the requirements of the project specification. Or, if you choose to, you can also use the sequential reference as a starting point and then parallelize it, altough it might not be the most `fast` parallel implementation.
- It is common that your initial parallel version is slower than our sequential reference (:D), but you should be able to improve it with profiling and tuning.
- You are allowed to show your ideas, thoughts, designs, and results on [Ed](https://edstem.org/au/courses/38370/discussion) to get feedback or help others. However, ***you are never allowed to share any of your code anywhere; doing so will result you (and any other who submitted any part of your code) zero marks of this assignment.***.


-------
Add your readme content below.
-------

## Authorship
- Student Name: Ruoning Ren
- Login ID: ruoningr
- Student ID: 1690503

## Instructions

### Testing correctness

```bash
make test                      # unit + differential tests (gtest)
make smoke INPUT=test.txt      # byte-exact md5 check against a locked baseline
                                # (locked baselines exist for test.txt, 10M.txt, 100M.txt)
```

`make smoke` also accepts a thread count via the environment, e.g.
`OMP_NUM_THREADS=8 make smoke INPUT=10M.txt`, to confirm the parallel
output matches the sequential reference at any thread count.

### Testing performance

Two ways:

**1. Quick interactive check**, using a Spartan interactive session:

```bash
sinteractive -p sapphire --qos=punim0520 --time=00:40:00 --cpus-per-task=8
# once inside the allocated node:
cd 90025/skeleton
make clean && make
for t in 1 2 4 8; do
  OMP_NUM_THREADS=$t ./bin/bpe /data/gpfs/projects/punim0520/2026/project1/100M.txt
done
```

This is a shared (non-exclusive) allocation, so timings can be noisy if
other jobs share the node; it is meant for quick sanity checks during
development, not for the numbers reported in the written report.

**2. Full formal benchmark**, matching Task 3's required output:

```bash
sbatch benchmark.slurm
squeue --me                    # check status
cat results.csv                # once the job completes
```

`benchmark.slurm` requests an `--exclusive` `sapphire` node and sweeps
`OMP_NUM_THREADS` = 1, 2, 4, 8, 16, 32 across `10M.txt`, `100M.txt` and
`1G.txt` by default (pass one or more file paths as arguments to
`sbatch benchmark.slurm <file> ...` to override), writing `results.csv`
with the columns `filename,file_md5,version,partition,threads,walltime,
ram,output_md5`. This exclusive, full sweep is what the written
report's measured results are based on.

## Acknowledgements
To be filled by you.