#include "bpe.h"
#include "absl/log/log.h"
#include <chrono>
namespace bpe {
namespace {
std::int64_t elapsed_ms(const std::chrono::steady_clock::time_point& start,
                        const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
        .count();
}
}
// Project 2: word counting and character splitting over every rank, combined
// into the global result (rule R1.2).
void parallel_task1(std::vector<Byte>& input, Results& results) {
    const std::chrono::steady_clock::time_point t0 =
        std::chrono::steady_clock::now();
    const std::vector<Word> words = split_words(input);
    const std::chrono::steady_clock::time_point t1 =
        std::chrono::steady_clock::now();
    LOG(INFO) << "split words: " << elapsed_ms(t0, t1) << " ms";
    task1(words, results);
}
// Project 2, Task 2: every round the ranks agree on the same 256 most frequent
// eligible pairs, counted over ALL ranks, and each rank then merges its own
// occurrences of those 256 before the next round selects again (rule R1.2).
// The 256 are a snapshot, chosen before any of them is merged -- not the single
// best pair merged one at a time, which is the sequential rule src/task2.cpp
// provides.
void parallel_task2(const std::vector<CharSplit>& splits, Results& results) {
    task2(splits, results);
}

}
