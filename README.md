# COMP90025 2026 Project 2 — BPE skeleton (MPI + OpenMP)

A modified Project 1 skeleton: the sequential implementation of both tasks, the
`Results` structure (rule R1.2), a minimal MPI entry (`src/mpi_entry.cpp`) and a
Makefile. You replace the two parallel API entries in `src/parallel.cpp` with
hybrid MPI and OpenMP implementations (rule R2.3):

1. `parallel_task1`, including `split_words`
2. `parallel_task2`

The skeleton does not distribute the corpus: rank 0 reads it and the other ranks
start empty (`src/pipeline.cpp`). That distribution is your work. A submission
that never calls MPI scores 0 for the whole project (rule R2.2).

## What is new from Project 1, and why it can be faster

Project 1 merged **one** pair per step: find the most frequent eligible pair,
merge it, find the next. Every step reads counts the previous step changed, so
Task 2 is a sequential loop that no number of threads can spread out.

Project 2 changes that one rule (specification, Task 2): **each round selects the
256 most frequent eligible pairs, merges all 256, and only then selects the next
256**, as a snapshot taken before any of them is merged. That one change is what
makes a cluster worth using:

* one global selection per 256 merges instead of one per merge, and it is done
  once by agreement rather than by every rank independently;
* the 256 pairs are disjoint, so every rank can merge its own occurrences of all
  of them at the same time, and the frequency changes travel once per round
  instead of once per merge;
* the counts still have to agree, so deltas are exchanged every round — that
  communication, and the synchronisation around it, is the part the
  specification asks you to make cheap.

The output is **not** Project 1's: 256 pairs merged together can differ from 256
merged one at a time, and the specification's own worked example answers
differently under the two rules. The reference is this rule run on one process --
`make seq`, built from `src/task2.cpp`, which already implements it.

## Layout

```
src/bpe.h          public API + Results (rule R1.2)
src/corpus.cpp     input reading + word splitting (provided)
src/task1.cpp      Task 1 sequential reference (provided)
src/task2.cpp      Task 2 sequential reference: the 256-pair round (provided)
src/output.cpp     output.txt writing + Task 1 dump (provided)
src/pipeline.cpp   pipeline + CLI (provided)
src/mpi_entry.cpp  MPI start-up: init, finalize, rank, size (provided)
src/parallel.cpp   the two parallel API entries — your work
abseil/            vendored Abseil logging subset
Makefile
```

`src/task2.cpp` implements **Project 2's** rule: the 256-pair round. Your parallel
`parallel_task2` has to reproduce its output byte for byte, and it is the place to
read the rule if the specification's description is not enough.

## Build and run

```
make                    # bin/bpe, the MPI build
make run INPUT=file.txt
make seq INPUT=file.txt # bin/bpe_seq, the sequential reference
make bench INPUT=file.txt
make clean
```

On Spartan, `module load GCC OpenMPI` first. `make` uses `mpicxx` when it is on
`PATH`; without it the build falls back to `g++` and `src/mpi_entry.cpp` becomes
a single-rank stub, so the skeleton still builds and its tests still run.

```
make mpi INPUT=file.txt RANKS=3
srun --nodes=1 --ntasks=4 --cpus-per-task=4 --time=00:05:00 ./bin/bpe file.txt
```

The graded configurations put **every rank of a configuration on one node**, on
different cores: they vary how the machine's 16 OpenMP threads are split into
ranks and thread teams, and no graded run touches the network. A step asks for
`--cpus-per-task` = the largest count in its tuple, for every rank, so it needs
`ranks x largest` CPUs — `{2,4,10}` needs 30 while running 16 threads.

### The sequential reference

`bin/bpe_seq` is the same sources built as a plain single-process program: **no
MPI at all**, not `mpirun -np 1` and not the MPI binary with one rank. It links
no `libmpi` and needs no launcher, and it implements the specification's rule, so
its `output.txt` is the reference your parallel version must match byte for byte. Because it builds from the same sources, your
code has to compile without MPI too: guard MPI calls with `#ifdef BPE_HAVE_MPI`,
as `src/mpi_entry.cpp` does.

### Output and self-check

`bin/bpe input.txt` prints walltimes and writes `output.txt` in the current
directory: one `token count` line per token, by decreasing count, ties by the
lexicographically smallest token first. For a fixed input the output is
byte-exact. The ranked run, the one-rank MPI run and the sequential binary must
all agree — rule R2.1 in miniature:

```
make seq INPUT=file.txt && cp output.txt seq.txt
OMP_NUM_THREADS=1 mpirun -np 1 ./bin/bpe file.txt
cmp seq.txt output.txt
```

## Benchmarking (Task 3)

```
sbatch benchmark.slurm corpora/8G.txt          # writes results.csv
REPS=5 sbatch benchmark.slurm corpora/8G.txt   # five runs per configuration
```

The four masked blocks are yours: the SLURM resource requests, the toolchain
load and build, the four configuration tuples, and the `srun` launch line. The
rest — the allocation check, the node log, `results.csv`, and the speed-up
arithmetic — is provided. Size the allocation for the widest step, not for 16
threads. Each rank reads the `BPE_THREADS` tuple and sets its own thread count
from its MPI rank index (rule R1.4); if you change that convention, update
`benchmark.slurm` and this README, because the teaching team runs the command you
document.

`results.csv` has one row per run (`filename,file_md5,version,partition,nodes,
ranks,omp_threads,walltime,ram,output_md5`). The script prints each
configuration's median and the aggregate speed-up

    S = 4 * T_baseline / (T1 + T2 + T3 + T4)

against the baseline, and the speed-mark band it falls in. The baseline is **the
sequential reference itself**: `T_baseline` is the wall time of `bin/bpe_seq`, the
executable `make seq` builds, on the same corpus on the same machine, one node and
one thread. The script builds it, times it, prints the number it used and checks
that its `output.txt` matches the four configurations.

The specification fixes no number for it on purpose. The corpus used for marking
is a different 8 GiB file, and a node under load slows the sequential run and your
distributed runs alike, so measuring both in one window is what keeps either from
moving your marks. `BASELINE=...` skips the baseline run and uses a value you
supply, which is only useful when re-using a measurement you already have.

Develop on the 8 GiB corpus you are given, but do not tune to it: a fixed word
count, a hard-coded shard size or a data-dependent threshold will meet a corpus
it has never seen.

## Style (advisory, not marked)

```
make format   # apply Google C++ formatting (clang-format 14.0.6)
make check    # report whether files already match; never fails a build
```

## Reading, splitting and logging

`split_words` splits in place: each whitespace byte becomes NUL and the word
starts are returned. A `Word` points into the input buffer and does not own it,
so the buffer must outlive it (`Results` copies the distinct words' bytes, so
results outlive it too). Input is ASCII, so no word contains NUL. Do not call
`split_words` twice on one buffer.

Logging is Abseil (`absl::log`) to **stderr**; stdout carries only the Task 1
output, so the byte-exact contract is unaffected.
https://abseil.io/docs/cpp/guides/logging

## What you must implement (rules R2.2 and R2.3)

Until you implement the parallel versions, `parallel_task1` and `parallel_task2`
call the sequential reference, so the program is correct but shows no speed-up.
A purely sequential implementation scores 0 for correctness and 0 for speed
(rule R2.3), even if you introduce your own sequential implementation with a
speed-up. A submission that never calls MPI scores 0 out of 20 for the whole
project (rule R2.2). The provided sequential reference is there only for you to
understand the problem and to verify your distributed implementation.

You may extend the API in `src/bpe.h` or add new source files in `src/` (the
Makefile compiles `src/*.cpp` automatically). Do not modify the sequential
reference or the output format, because the evaluation compares your parallel
result against them. You may add logging for debugging, but do not remove the
existing statements. During assessment we overwrite the sequential reference
files with the provided skeleton versions, so changes to them are lost. ***Do
not try to fake performance logs; doing so will result in zero marks for this
assignment.***

## Notes

- The parallel version does not have to be based on the provided sequential
  reference. You may implement your own from scratch, as long as it meets the
  specification; the reference is also a starting point you can parallelise,
  although it may not be the fastest one.
- It is common for a first parallel version to be slower than the sequential
  reference (:D). Profiling and tuning are part of the exercise.
- You may discuss ideas, designs and results on
  [Ed](https://edstem.org/au/courses/38370/discussion). ***You are never allowed
  to share your code anywhere; doing so results in zero marks for you and for
  anyone who submitted any part of it.***

-------
Add your readme content below.
-------

## Authorship
- Student Name: John Doe
- Login ID: johndoe
- Student ID: 12345678

## Instructions
To be filled by you.

## Acknowledgements
To be filled by you.