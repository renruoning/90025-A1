#include "bpe.h"
#include "absl/log/log.h"
#include <chrono>
#include <omp.h>
#include <string_view>
#include <unordered_map>
namespace bpe {
namespace {
std::int64_t elapsed_ms(const std::chrono::steady_clock::time_point& start,
                        const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
        .count();
}
}
void parallel_task1(std::vector<Byte>& input, Results& results) {
    const std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    const std::vector<Word> words = split_words(input);
    const std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    LOG(INFO) << "split words: " << elapsed_ms(t0, t1) << " ms";
    //scan words
    const int num_threads = omp_get_max_threads();
    std::vector<std::unordered_map<std::string_view, std::size_t>> local_counts(num_threads);

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        auto& mine = local_counts[tid];
        mine.reserve(words.size() / num_threads);

        #pragma omp for schedule(static)
        for (std::size_t i = 0; i < words.size(); ++i) {
            std::string_view sv(reinterpret_cast<const char*>(words[i].bytes));
            ++mine[sv];
        }
    }

    //merge counts
    std::unordered_map<std::string_view, std::size_t> merged_counts;
    for (const auto& local_count : local_counts) {
        for (const auto& [word, count] : local_count) {
            merged_counts[word] += count;
        }
    }
    //change format
    results.word_counts.clear();
    results.char_splits.clear();
    results.word_counts.reserve(merged_counts.size());
    results.char_splits.reserve(merged_counts.size());
    for (const auto& [word, count] : merged_counts) {
        const Byte* bytes = reinterpret_cast<const Byte*>(word.data());
        std::vector<Byte> word_bytes(bytes, bytes + word.size());
        results.word_counts.push_back(WordCount{word_bytes, count});
        results.char_splits.push_back(CharSplit{word_bytes, count});
    }
}
void parallel_task2(const std::vector<CharSplit>& splits, Results& results) {
    task2(splits, results);
}

}
