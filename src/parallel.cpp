#include "bpe.h"
#include "absl/log/log.h"
#include <chrono>
#include <omp.h>
#include <string_view>
#include <unordered_map>
#include <algorithm>
namespace bpe {
namespace {
std::int64_t elapsed_ms(const std::chrono::steady_clock::time_point& start,
                        const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
        .count();
}
}
namespace {
    inline bool is_separator(Byte b){
        return b==0x20 || b==0x09 || b==0x0A || b==0x0D || b==0x00;
    }
}

std::vector<Word> parallel_split_words(std::vector<Byte>& input) {
    input.push_back(Byte('\0'));
    const std::size_t n = input.size();
    Byte* buf = input.data();
    //每个字节独立判断是否为分隔符，是就改成'\0'，否则不变
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        if (is_separator(buf[i])) {
            buf[i] = Byte('\0');
        }
    }
    //每个线程负责一段连续区间，找出这段里的词首指针，存进本地vector，然后按线程号拼接，保持原有的顺序
    const int num_threads = omp_get_max_threads();
    std::vector<std::vector<Word>> local_words(num_threads);
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const std::size_t chunk = (n + num_threads - 1) / num_threads;
        const std::size_t lo = std::min(n,static_cast<std::size_t>(tid) * chunk);
        const std::size_t hi = std::min(n, lo+chunk);
        auto& mine = local_words[tid];
        for(std::size_t i=lo;i<hi;++i){
            const bool is_start = buf[i]!=0 && (i==0 || buf[i-1]==0);
            if(is_start){
                mine.push_back(Word{buf+i});
            }
        }
    }
    // 拼接所有线程的词
    std::vector<Word> words;
    std::size_t total=0;
    for(const auto& lw: local_words){
        total+=lw.size();
    }
    words.reserve(total);
    for (const auto& lw : local_words) {
        words.insert(words.end(), lw.begin(), lw.end());
    }
    return words;
}

void parallel_task1(std::vector<Byte>& input, Results& results) {
    const std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    const std::vector<Word> words = parallel_split_words(input);
    const std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    LOG(INFO) << "split words: " << elapsed_ms(t0, t1) << " ms";
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

    std::unordered_map<std::string_view, std::size_t> merged_counts;
    for (const auto& local_count : local_counts) {
        for (const auto& [word, count] : local_count) {
            merged_counts[word] += count;
        }
    }
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
