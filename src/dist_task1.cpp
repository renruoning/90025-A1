#include "dist.h"
#include "bpe.h"
#include "absl/log/log.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <omp.h>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#ifdef BPE_HAVE_MPI
#include <mpi.h>
#endif
namespace bpe {
namespace {

std::uint64_t hash_bytes(const Byte* s, std::size_t n) {
    std::uint64_t h = 1469598103934665603ULL;
    for (std::size_t i = 0; i < n; ++i) h = (h ^ s[i])*1099511628211ULL;
    h ^= h >> 29;   // fnv is weak in the low bits, mix a bit
    return h;
}
struct SvHash {
    std::size_t operator()(std::string_view v) const {
        return hash_bytes(reinterpret_cast<const Byte*>(v.data()), v.size());
    }
};
using CountMap = std::unordered_map<std::string_view, std::size_t, SvHash>;

// record on the wire: worker (u32), length (u32), count (u64), then the bytes
const std::size_t kHead = 16;
void put_record(std::vector<Byte>& out, std::uint32_t worker, std::string_view w,
                std::uint64_t count) {
    const std::uint32_t len = static_cast<std::uint32_t>(w.size());
    const std::size_t at = out.size();
    out.resize(at + kHead+len);
    std::memcpy(&out[at], &worker, 4);
    std::memcpy(&out[at+4], &len, 4);
    std::memcpy(&out[at + 8], &count, 8);
    std::memcpy(&out[at + kHead], w.data(), len);
}

std::int64_t ms_since(const std::chrono::steady_clock::time_point& t) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now() - t).count();
}
}
namespace {
std::vector<std::size_t> g_bounds;
}
const std::vector<std::size_t>& task1_bounds() { return g_bounds; }

int worker_of(const Byte* s, std::size_t n) {
    return static_cast<int>(hash_bytes(s, n) % num_workers());
}

void distributed_task1(std::vector<Byte>& input, Results& results) {
    const auto t0 = std::chrono::steady_clock::now();
    const int ranks=mpi_size();
    const int me = mpi_rank();
    const int T=threads_of(me);

    const std::vector<Word> words = split_words(input);
    LOG(INFO) << "split words: " << ms_since(t0) << " ms";

    // 1. every thread counts its slice of the words, then packs the entries
    //    into one buffer per destination rank
    std::vector<std::vector<std::vector<Byte>>> out(
        T, std::vector<std::vector<Byte>>(ranks));
#pragma omp parallel num_threads(T)
    {
        const int t = omp_get_thread_num();
        const std::size_t lo = words.size()*t / T;
        const std::size_t hi = words.size()*(t+1) / T;
        CountMap local;
        for (std::size_t i = lo; i < hi; ++i) {
            const char* p = reinterpret_cast<const char*>(words[i].bytes);
            ++local[std::string_view(p, std::strlen(p))];
        }
        for (const auto& e : local) {
            const int w = worker_of(reinterpret_cast<const Byte*>(e.first.data()),
                                    e.first.size());
            put_record(out[t][worker_rank(w)],w, e.first, e.second);
        }
    }
    const auto t1 = std::chrono::steady_clock::now();

    // 2. one Alltoallv moves every entry to the rank that owns it
    std::vector<int> scount(ranks,0), sdisp(ranks, 0), rcount(ranks, 0), rdisp(ranks, 0);
    std::vector<Byte> sbuf;
    for (int r = 0; r < ranks; ++r) {
        sdisp[r] = static_cast<int>(sbuf.size());
        for (int t = 0; t<T; ++t)
            sbuf.insert(sbuf.end(), out[t][r].begin(), out[t][r].end());
        const std::size_t sz = sbuf.size()-sdisp[r];
        if (sbuf.size() >= (std::size_t(1) << 31))
            throw std::runtime_error("task1 exchange buffer too large for MPI int counts");
        scount[r] = static_cast<int>(sz);
    }
    out.clear();
    std::vector<Byte> rbuf;
    if (ranks == 1) {
        rbuf.swap(sbuf);  // nothing to send anywhere
    }
#ifdef BPE_HAVE_MPI
    else {
    MPI_Alltoall(scount.data(), 1, MPI_INT, rcount.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::size_t total = 0;
    for (int r = 0; r < ranks; ++r) {
        rdisp[r]=static_cast<int>(total);
        total+= rcount[r];
    }
    if (total >= (std::size_t(1) << 31))
        throw std::runtime_error("task1 exchange buffer too large for MPI int counts");
    rbuf.resize(total);
    MPI_Alltoallv(sbuf.data(), scount.data(), sdisp.data(), MPI_BYTE, rbuf.data(),
                  rcount.data(), rdisp.data(), MPI_BYTE, MPI_COMM_WORLD);
    }
#endif
    std::vector<Byte>().swap(sbuf);
    const auto t2 = std::chrono::steady_clock::now();

    // 3. each thread owns one worker: scan the received records, keep its own
    const int w0 =first_worker(me);
    std::vector<std::vector<WordCount>> mine(T);
#pragma omp parallel num_threads(T)
    {
        const int t = omp_get_thread_num();
        CountMap merged;
        std::size_t at=0;
        while (at < rbuf.size()) {
            std::uint32_t worker,len;
            std::uint64_t count;
            std::memcpy(&worker, &rbuf[at], 4);
            std::memcpy(&len, &rbuf[at + 4], 4);
            std::memcpy(&count, &rbuf[at + 8], 8);
            if (static_cast<int>(worker)==w0 + t) {
                merged[std::string_view(
                    reinterpret_cast<const char*>(&rbuf[at + kHead]), len)] += count;
            }
            at += kHead+len;
        }
        std::vector<std::pair<std::string_view, std::size_t>> sorted(merged.begin(),
                                                                     merged.end());
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        mine[t].reserve(sorted.size());
        for (const auto& e : sorted) {
            const Byte* b = reinterpret_cast<const Byte*>(e.first.data());
            mine[t].push_back(WordCount{std::vector<Byte>(b, b + e.first.size()), e.second});
        }
    }

    results.word_counts.clear();
    results.char_splits.clear();
    g_bounds.assign(1, 0);
    for (int t = 0; t < T; ++t){
        for (WordCount& wc : mine[t]) {
            results.char_splits.push_back(CharSplit{wc.word, wc.count});
            results.word_counts.push_back(std::move(wc));
        }
        g_bounds.push_back(results.word_counts.size());
    }
    std::size_t freq=0;
    for (const WordCount& wc : results.word_counts) freq += wc.count;
    LOG(INFO) << "rank " << me << " owns " << results.word_counts.size()
              << " distinct words, freq sum " << freq << " (local "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
              << " ms, exchange "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
              << " ms, total " << ms_since(t0) << " ms)";
}

}
