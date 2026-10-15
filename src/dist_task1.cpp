#include "dist.h"
#include "bpe.h"
#include "absl/log/log.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <omp.h>
#include <stdexcept>
#ifdef BPE_HAVE_MPI
#include <mpi.h>
#endif
namespace bpe {
namespace {

// hash of a word's bytes, 8 at a time. The length goes in too.
std::uint64_t hash_word(const Byte* p, std::size_t n) {
    const std::uint64_t k = 0x9fb21c651e98df25ULL;
    std::uint64_t h = 0x9e3779b97f4a7c15ULL ^ (n * 0xff51afd7ed558ccdULL);
    while (n >= 8) {
        std::uint64_t c;
        std::memcpy(&c, p, 8);
        h = (h ^ c) * k;
        h ^= h >> 32;
        p += 8;
        n -= 8;
    }
    if (n) {
        std::uint64_t c = 0;
        std::memcpy(&c, p, n);
        h = (h ^ c) * k;
        h ^= h >> 32;
    }
    h *= 0xff51afd7ed558ccdULL;
    return h ^ (h >> 29);
}

bool sep_table[256];
bool sep_ready = [] {
    for (int b : {0x20, 0x09, 0x0A, 0x0D, 0x00}) sep_table[b] = true;
    return true;
}();

// open addressing, linear probing. Keeps the hash so growing never rehashes.
struct Slot {
    const Byte* p = nullptr;
    std::uint64_t hash = 0;
    std::uint64_t count = 0;
    std::uint32_t len = 0;
};
class WordTable {
  public:
    WordTable() : slots_(1 << 14) {}
    void add(const Byte* p, std::uint32_t len, std::uint64_t h, std::uint64_t c) {
        const std::size_t mask = slots_.size() - 1;
        std::size_t i = h & mask;
        for (;;) {
            Slot& s = slots_[i];
            if (s.p == nullptr) {
                s.p = p; s.hash = h; s.count = c; s.len = len;
                if (++used_ * 2 > slots_.size()) grow();
                return;
            }
            if (s.hash == h && s.len == len && std::memcmp(s.p, p, len) == 0) {
                s.count += c;
                return;
            }
            i = (i + 1) & mask;
        }
    }
    const std::vector<Slot>& slots() const { return slots_; }
    std::size_t size() const { return used_; }

  private:
    void grow() {
        std::vector<Slot> old(slots_.size() * 2);
        old.swap(slots_);
        const std::size_t mask = slots_.size() - 1;
        for (const Slot& s : old) {
            if (s.p == nullptr) continue;
            std::size_t i = s.hash & mask;
            while (slots_[i].p != nullptr) i = (i + 1) & mask;
            slots_[i] = s;
        }
    }
    std::vector<Slot> slots_;
    std::size_t used_ = 0;
};

// record on the wire: hash (u64), count (u64), length (u32), then the bytes
const std::size_t kHead = 20;
void put_record(std::vector<Byte>& out, const Slot& s) {
    const std::size_t at = out.size();
    out.resize(at + kHead + s.len);
    std::memcpy(&out[at], &s.hash, 8);
    std::memcpy(&out[at + 8], &s.count, 8);
    std::memcpy(&out[at + 16], &s.len, 4);
    std::memcpy(&out[at + kHead], s.p, s.len);
}

std::int64_t ms_since(const std::chrono::steady_clock::time_point& t) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now() - t).count();
}
std::vector<std::size_t> g_bounds;
}
const std::vector<std::size_t>& task1_bounds() { return g_bounds; }

int worker_of(const Byte* s, std::size_t n) {
    return static_cast<int>(hash_word(s, n) % num_workers());
}

void distributed_task1(std::vector<Byte>& input, Results& results) {
    (void)sep_ready;
    const auto t0 = std::chrono::steady_clock::now();
    const int ranks = mpi_size();
    const int me = mpi_rank();
    const int T = threads_of(me);
    const std::uint64_t W = num_workers();

    // 1. every thread takes a slice of the bytes (cut between words), counts
    //    its words and packs the entries into one buffer per destination rank
    const std::vector<std::size_t> cuts =
        split_points(input.data(), input.size(), std::vector<int>(T, 1));
    std::vector<std::vector<std::vector<Byte>>> out(
        T, std::vector<std::vector<Byte>>(ranks));
#pragma omp parallel num_threads(T)
    {
        const int t = omp_get_thread_num();
        const Byte* p = input.data() + cuts[t];
        const Byte* const end = input.data() + cuts[t + 1];
        WordTable local;
        while (p < end) {
            while (p < end && sep_table[*p]) ++p;
            const Byte* start = p;
            while (p < end && !sep_table[*p]) ++p;
            if (p == start) break;
            const std::uint32_t len = static_cast<std::uint32_t>(p - start);
            local.add(start, len, hash_word(start, len), 1);
        }
        for (const Slot& s : local.slots())
            if (s.p) put_record(out[t][worker_rank(int(s.hash % W))], s);
    }
    const auto t1 = std::chrono::steady_clock::now();

    // 2. one Alltoallv moves every entry to the rank that owns it
    std::vector<int> scount(ranks, 0), sdisp(ranks, 0), rcount(ranks, 0), rdisp(ranks, 0);
    std::vector<Byte> sbuf;
    for (int r = 0; r < ranks; ++r) {
        sdisp[r] = static_cast<int>(sbuf.size());
        for (int t = 0; t < T; ++t)
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
    std::vector<WordTable> merged(T);
#pragma omp parallel num_threads(T)
    {
        const int t = omp_get_thread_num();
        std::size_t at=0;
        while (at < rbuf.size()) {
            std::uint64_t hash, count;
            std::uint32_t len;
            std::memcpy(&hash, &rbuf[at], 8);
            std::memcpy(&count, &rbuf[at + 8], 8);
            std::memcpy(&len, &rbuf[at + 16], 4);
            if (static_cast<int>(hash % W)==w0 + t)
                merged[t].add(&rbuf[at + kHead], len, hash, count);
            at += kHead+len;
        }
    }

    results.word_counts.clear();
    results.char_splits.clear();
    g_bounds.assign(1, 0);
    std::size_t freq=0;
    for (int t = 0; t < T; ++t){
        for (const Slot& s : merged[t].slots()) {
            if (!s.p) continue;
            std::vector<Byte> w(s.p, s.p + s.len);
            results.char_splits.push_back(CharSplit{w, s.count});
            results.word_counts.push_back(WordCount{std::move(w), s.count});
            freq += s.count;
        }
        g_bounds.push_back(results.word_counts.size());
    }
    LOG(INFO) << "rank " << me << " owns " << results.word_counts.size()
              << " distinct words, freq sum " << freq << " (local "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
              << " ms, exchange "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
              << " ms, total " << ms_since(t0) << " ms)";
}

}
