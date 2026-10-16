#include "dist.h"
#include "bpe.h"
#include "absl/log/log.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <queue>
#include <string>
#include <omp.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#ifdef BPE_HAVE_MPI
#include <mpi.h>
#endif
namespace bpe {
namespace {

using u8 = std::uint8_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using i64 = std::int64_t;

constexpr u32 kNone = std::numeric_limits<u32>::max();
constexpr std::size_t kBatch = 256;  // pairs merged per round

u64 pack(u32 l, u32 r) { return (u64(l)<<32) | r; }
u32 left_of(u64 key) { return u32(key >> 32); }
u32 right_of(u64 key) { return u32(key); }

// token texts, same on every rank. ids 1..255 are the bytes themselves
struct Vocab {
    std::vector<std::string> text;
    Vocab() {
        text.resize(256);
        for (u32 v = 1; v<256; ++v) text[v].assign(1, char(v));
    }
    std::string merged(u64 key) const {
        return text[left_of(key)] + text[right_of(key)];
    }
};

// first 8 bytes of the merged text, so most ties never need a strcmp
u64 fingerprint(const Vocab& v, u64 key) {
    const std::string& a = v.text[left_of(key)];
    const std::string& b = v.text[right_of(key)];
    u64 f = 0;
    std::size_t i = 0;
    for (; i < a.size() && i < 8; ++i) f |= u64(u8(a[i])) << (8*(7 - i));
    for (std::size_t j = 0; j < b.size() && i < 8; ++j, ++i)
        f |= u64(u8(b[j])) << (8 * (7 - i));
    return f;
}

struct Cand {
    u64 count;
    u64 fp;
    u64 key;
};

// true when a ranks below b: fewer occurrences, then larger text, then larger key
struct LowerPrio {
    const Vocab* vocab;
    bool operator()(const Cand& a, const Cand& b) const {
        if (a.count != b.count) return a.count < b.count;
        if (a.fp != b.fp) return a.fp > b.fp;
        const int c = std::strcmp(vocab->merged(a.key).c_str(),
                                  vocab->merged(b.key).c_str());
        if (c != 0) return c > 0;
        return a.key > b.key;
    }
};

struct Delta {
    u64 key;
    i64 dcount;
    i64 dwords;
};

struct Pair {
    u64 key = 0;
    u64 count = 0;
    u32 words = 0;
    u64 sent_count = 0;
    u32 sent_words = 0;
    u32 touched = 0;
    u32 last_word = kNone;
    u32 last_group = kNone;
    std::vector<u32> pos;
};

// The words one worker owns, as linked lists of tokens. It keeps its own view
// of every pair and reports how that view changed (Delta) after each round.
class Worker {
  public:
    void build(const CharSplit* begin, const CharSplit* end) {
        const std::size_t nwords = end - begin;
        std::size_t slots = 0;
        for (const CharSplit* s = begin; s != end; ++s) slots += s->chars.size() + 1;
        token_.reserve(slots);
        prev_.reserve(slots);
        next_.reserve(slots);
        word_of_.reserve(slots);
        edge_group_.reserve(slots);
        alive_.reserve(slots);
        token_count_.assign(256, 0);
        ensure_cache(256);
        std::vector<u32> first(256 * 256, kNone);
        for (u32 w = 0; w < nwords; ++w) {
            const CharSplit& s = begin[w];
            freq_.push_back(s.count);
            const u32 start = u32(token_.size());
            for (std::size_t i = 0; i < s.chars.size(); ++i) {
                const u32 p = u32(token_.size());
                token_.push_back(s.chars[i]);
                prev_.push_back(i == 0 ? kNone : p - 1);
                next_.push_back(p + 1);
                word_of_.push_back(w);
                edge_group_.push_back(kNone);
                alive_.push_back(1);
                token_count_[s.chars[i]] += s.count;
            }
            // sentinel slot closes the word
            const u32 sent = u32(token_.size());
            token_.push_back(0);
            prev_.push_back(sent - 1);
            next_.push_back(kNone);
            word_of_.push_back(w);
            edge_group_.push_back(kNone);
            alive_.push_back(0);
            for (u32 p = start; live(p) && live(next_[p]); p = next_[p]) {
                u32& id = first[token_[p] * 256 + token_[next_[p]]];
                if (id == kNone) id = new_pair(pack(token_[p], token_[next_[p]]));
                add_edge(p, id, w, s.count);
            }
        }
    }

    // merge every current (left,right) into `merged`, left to right in a word
    void merge(u64 key, u32 merged) {
        if (token_count_.size() <= merged) token_count_.resize(merged + 1, 0);
        ensure_cache(merged + 1);
        auto it = index_.find(key);
        if (it == index_.end()) return;
        std::vector<u32> positions = std::move(pairs_[it->second].pos);
        pairs_[it->second].pos.clear();
        const u32 lt = left_of(key), rt=right_of(key);
        u32 cur_word = kNone;
        u64 f = 0;
        for (u32 p : positions) {
            if (!live(p) || token_[p] != lt) continue;
            const u32 rp = next_[p];
            if (!live(rp)||token_[rp] != rt) continue;
            const u32 w = word_of_[p];
            if (w != cur_word) {
                cur_word = w;
                f = freq_[w];
            }
            const u32 lp = prev_[p];
            const u32 ap = next_[rp];
            if (live(lp)) remove_edge(lp, f);
            remove_edge(p, f);
            if (live(ap)) remove_edge(rp, f);
            token_count_[lt]-= f;
            token_count_[rt] -= f;
            token_count_[merged] += f;
            token_[p] = merged;
            alive_[rp] = 0;
            next_[p] = ap;
            if (ap != kNone) prev_[ap] = p;
            prev_[rp] = kNone;
            next_[rp] = kNone;
            edge_group_[rp] = kNone;
            if (live(lp)) add_edge(lp, left_pair(token_[lp], merged), w, f);
            if (live(ap)) add_edge(p, right_pair(merged, token_[ap]), w, f);
        }
    }

    // what changed since the last call
    void take_deltas(std::vector<Delta>& out) {
        for (u32 id : touched_) {
            Pair& p = pairs_[id];
            const i64 dc = i64(p.count)-i64(p.sent_count);
            const i64 dw = i64(p.words) - i64(p.sent_words);
            p.sent_count = p.count;
            p.sent_words = p.words;
            if (dc != 0 || dw != 0) out.push_back(Delta{p.key, dc, dw});
        }
        touched_.clear();
        ++round_;
    }

    const std::vector<u64>& token_count() const { return token_count_; }

  private:
    bool live(u32 p) const { return p != kNone && alive_[p] != 0; }

    void touch(u32 id) {
        if (pairs_[id].touched != round_) {
            pairs_[id].touched = round_;
            touched_.push_back(id);
        }
    }
    void ensure_cache(std::size_t n) {
        if (left_stamp_.size() >= n) return;
        const std::size_t sz = std::max<std::size_t>(left_stamp_.size()*2, n+1024);
        left_stamp_.resize(sz, kNone);
        left_id_.resize(sz, kNone);
        right_stamp_.resize(sz, kNone);
        right_id_.resize(sz, kNone);
    }
    u32 new_pair(u64 key) {
        const u32 id = u32(pairs_.size());
        pairs_.emplace_back();
        pairs_.back().key=key;
        index_[key] = id;
        touch(id);
        return id;
    }
    // pair (l, merged); every pair made while merging `merged` is made once
    u32 left_pair(u32 l, u32 merged) {
        if (l==merged) return same_pair(merged);
        if (left_stamp_[l] != merged) {
            left_stamp_[l] = merged;
            left_id_[l] = new_pair(pack(l, merged));
        }
        return left_id_[l];
    }
    u32 right_pair(u32 merged, u32 r) {
        if (r == merged) return same_pair(merged);
        if (right_stamp_[r] != merged) {
            right_stamp_[r] = merged;
            right_id_[r] = new_pair(pack(merged, r));
        }
        return right_id_[r];
    }
    u32 same_pair(u32 merged) {
        if (same_stamp_ != merged) {
            same_stamp_ = merged;
            same_id_ = new_pair(pack(merged, merged));
        }
        return same_id_;
    }

    // one group = the edges of one pair inside one word
    u32 group_of(u32 id, u32 word) {
        Pair& p = pairs_[id];
        if (p.last_word == word) return p.last_group;
        u32 g;
        if (free_head_ != kNone) {
            g = free_head_;
            free_head_ = group_count_[g];
            group_count_[g] = 0;
            group_pair_[g] = id;
        } else {
            g = u32(group_pair_.size());
            group_pair_.push_back(id);
            group_count_.push_back(0);
        }
        p.last_word = word;
        p.last_group = g;
        return g;
    }
    void add_edge(u32 at, u32 id, u32 word, u64 f) {
        const u32 g = group_of(id, word);
        Pair& p = pairs_[id];
        if (group_count_[g]++==0) ++p.words;
        p.count += f;
        p.pos.push_back(at);
        edge_group_[at] = g;
        touch(id);
    }
    void remove_edge(u32 at, u64 f) {
        const u32 g = edge_group_[at];
        const u32 id = group_pair_[g];
        Pair& p = pairs_[id];
        p.count -= f;
        if (--group_count_[g]==0) {
            --p.words;
            if (p.last_group == g) {
                p.last_group = kNone;
                p.last_word = kNone;
            }
            group_count_[g] = free_head_;
            free_head_ = g;
        }
        edge_group_[at] = kNone;
        touch(id);
    }

    std::vector<u64> freq_;
    std::vector<u32> token_, prev_, next_, word_of_, edge_group_;
    std::vector<u8> alive_;
    std::vector<u64> token_count_;
    std::vector<Pair> pairs_;
    std::unordered_map<u64, u32> index_;
    std::vector<u32> group_pair_, group_count_;
    u32 free_head_ = kNone;
    std::vector<u32> left_stamp_, left_id_, right_stamp_, right_id_;
    u32 same_stamp_ = kNone, same_id_ = kNone;
    std::vector<u32> touched_;
    u32 round_ = 1;
};

// Global count / word count of the pairs a worker is responsible for.
class Owner {
  public:
    explicit Owner(const Vocab* v) : vocab_(v), heap_(LowerPrio{v}) {}

    void add(const Delta& d) {
        auto r = table_.try_emplace(d.key);
        if (r.second) born_.push_back(d.key);
        r.first->second.count += u64(d.dcount);  // wraps for negatives, still right
        r.first->second.words += d.dwords;
    }
    // new pairs join the queue once the whole round has been applied
    void end_round() {
        for (u64 key : born_) {
            const G& g = table_[key];
            if (g.words>=2 && g.count != 0)
                heap_.push(Cand{g.count, fingerprint(*vocab_, key), key});
        }
        born_.clear();
    }
    // up to n best eligible pairs, best first. Counts only ever shrink, so a
    // stale entry is fixed by pushing it back with its real count.
    void pop_cands(std::size_t n, std::vector<Cand>& out) {
        while (out.size()<n && !heap_.empty()) {
            Cand c = heap_.top();
            heap_.pop();
            const G& g = table_[c.key];
            if (g.words < 2 ||g.count == 0) continue;
            if (g.count != c.count) {
                c.count = g.count;
                heap_.push(c);
                continue;
            }
            out.push_back(c);
        }
    }
    void give_back(const Cand& c) { heap_.push(c); }

  private:
    struct G {
        u64 count = 0;
        i64 words = 0;
    };
    const Vocab* vocab_;
    std::unordered_map<u64, G> table_;
    std::vector<u64> born_;
    std::priority_queue<Cand, std::vector<Cand>, LowerPrio> heap_;
};


u64 mix(u64 x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
int owner_of(u64 key) { return int(mix(key) % u64(num_workers())); }

struct WireCand {
    u64 count;
    u64 key;
};

#ifdef BPE_HAVE_MPI
// every rank ends up with the concatenation of everyone's items
template <class T>
std::vector<T> allgather(const std::vector<T>& mine) {
    const int ranks = mpi_size();
    int n = int(mine.size() * sizeof(T));
    std::vector<int> cnt(ranks), disp(ranks);
    MPI_Allgather(&n, 1, MPI_INT, cnt.data(), 1, MPI_INT, MPI_COMM_WORLD);
    int total = 0;
    for (int r = 0; r < ranks; ++r) {
        disp[r] = total;
        total += cnt[r];
    }
    std::vector<T> all(total / sizeof(T));
    MPI_Allgatherv(mine.data(), n, MPI_BYTE, all.data(), cnt.data(), disp.data(),
                   MPI_BYTE, MPI_COMM_WORLD);
    return all;
}
#endif

// Send every delta to the worker that owns its pair. Each thread sorts its own
// deltas into one bucket per destination worker, the buckets are laid out by
// worker (so a rank's share is one block), and a short header of counts lets
// each receiving thread read just its own part.
void route_deltas(std::vector<std::vector<Delta>>& out, std::vector<Owner>& owners) {
    const int ranks = mpi_size();
    const int me = mpi_rank();
    const int T = int(owners.size());
    const int W = num_workers();
    static std::vector<std::vector<std::vector<Delta>>> bk;  // [source thread][dest worker]
    static std::vector<Delta> sbuf, rbuf;
    const std::size_t big = std::size_t(1) << 20;
    bk.resize(T);
    for (auto& v : bk) v.resize(W);

#pragma omp parallel num_threads(T)
    {
        const int t = omp_get_thread_num();
        for (auto& v : bk[t]) {
            if (v.capacity() > big) std::vector<Delta>().swap(v);
            v.clear();
        }
        for (const Delta& d: out[t]) bk[t][owner_of(d.key)].push_back(d);
        if (out[t].capacity()> big) std::vector<Delta>().swap(out[t]);
        out[t].clear();
    }

    // counts per destination worker, and where each bucket lands in sbuf
    std::vector<int> hw(W, 0);
    std::vector<std::size_t> base(W+1, 0);
    for (int w = 0; w<W; ++w) {
        for (int t =0;t < T;++t) hw[w] +=int(bk[t][w].size());
        base[w+1]=base[w] + std::size_t(hw[w]);
    }
    if (sbuf.capacity() >big&&base[W]<big/8) std::vector<Delta>().swap(sbuf);
    sbuf.resize(base[W]);
#pragma omp parallel num_threads(T)
    {
        const int t = omp_get_thread_num();
        for (int w = 0; w<W; ++w) {
            std::size_t pos=base[w];
            for (int u =0; u < t;++u)pos += bk[u][w].size();
            std::copy(bk[t][w].begin(),bk[t][w].end(), sbuf.begin() + pos);
        }
    }

    std::vector<int> hdr;          // [source rank * T + my thread] -> delta count
    std::vector<std::size_t> rdisp(ranks, 0);
    if (ranks == 1) {
        hdr = hw;
        rbuf.swap(sbuf);
    }
#ifdef BPE_HAVE_MPI
    else {
        std::vector<int> hs(ranks), hsd(ranks), hr(ranks, T), hrd(ranks);
        std::vector<int> scount(ranks), sdisp(ranks), rcount(ranks), rdsp(ranks);
        for (int r = 0; r < ranks; ++r) {
            hs[r] = threads_of(r);
            hsd[r] = first_worker(r);
            hrd[r] = r * T;
            const int lo = first_worker(r), hi = lo + threads_of(r);
            sdisp[r] = int(base[lo] * sizeof(Delta));
            scount[r] = int((base[hi] - base[lo]) * sizeof(Delta));
        }
        hdr.assign(std::size_t(ranks) * T, 0);
        MPI_Alltoallv(hw.data(), hs.data(), hsd.data(), MPI_INT,
                      hdr.data(), hr.data(), hrd.data(), MPI_INT, MPI_COMM_WORLD);
        int total = 0;
        for (int r = 0; r < ranks; ++r) {
            long cnt = 0;
            for (int t = 0; t < T; ++t) cnt += hdr[std::size_t(r) * T + t];
            rcount[r] = int(cnt * long(sizeof(Delta)));
            rdsp[r] = total;
            rdisp[r] = std::size_t(total) / sizeof(Delta);
            total += rcount[r];
        }
        if (rbuf.capacity() > big && std::size_t(total) < big) std::vector<Delta>().swap(rbuf);
        rbuf.resize(std::size_t(total) / sizeof(Delta));
        MPI_Alltoallv(sbuf.data(), scount.data(), sdisp.data(), MPI_BYTE,
                      rbuf.data(), rcount.data(), rdsp.data(), MPI_BYTE,
                      MPI_COMM_WORLD);
    }
#endif
#pragma omp parallel num_threads(T)
    {
        const int t = omp_get_thread_num();
        for (int r = 0; r < ranks; ++r) {
            std::size_t pos = rdisp[r];
            for (int u = 0; u < t; ++u) pos += std::size_t(hdr[std::size_t(r) * T + u]);
            const std::size_t len = std::size_t(hdr[std::size_t(r) * T + t]);
            for (std::size_t i = 0; i < len; ++i) owners[t].add(rbuf[pos + i]);
        }
        owners[t].end_round();
    }
    (void)me;
}

}
void distributed_task2(const std::vector<CharSplit>& splits, Results& results) {
    const auto t0 = std::chrono::steady_clock::now();
    auto us = [](std::chrono::steady_clock::time_point a) {
        return std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::steady_clock::now() - a).count();
    };
    auto ms = [&](std::chrono::steady_clock::time_point a) { return us(a) / 1000; };
    const int me = mpi_rank();
    const int T = threads_of(me);
    const std::vector<std::size_t>& bounds = task1_bounds();

    Vocab vocab;
    std::vector<Worker> workers(T);
    std::vector<Owner> owners(T, Owner(&vocab));
    std::vector<std::vector<Delta>> deltas(T);

    // build local words, then tell the owners what the pairs look like
#pragma omp parallel num_threads(T)
    {
        const int t = omp_get_thread_num();
        workers[t].build(splits.data() + bounds[t], splits.data() + bounds[t + 1]);
        workers[t].take_deltas(deltas[t]);
    }
    route_deltas(deltas, owners);
    const auto t_build = ms(t0);

    std::int64_t t_sel = 0, t_back = 0, t_merge = 0, t_route = 0;
    int rounds = 0;
    std::vector<std::vector<Cand>> popped(T);
    for (;;) {
        // each owner offers its best 256, the best 256 of all of them win
        auto ts = std::chrono::steady_clock::now();
        std::vector<WireCand> mine;
#pragma omp parallel num_threads(T)
        {
            const int t = omp_get_thread_num();
            popped[t].clear();
            owners[t].pop_cands(kBatch, popped[t]);
        }
        for (int t = 0; t < T; ++t)
            for (const Cand& c : popped[t]) mine.push_back(WireCand{c.count, c.key});
        std::vector<WireCand> all;
        if (mpi_size() == 1) {
            all.swap(mine);
        }
#ifdef BPE_HAVE_MPI
        else {
            all = allgather(mine);
        }
#endif
        std::vector<Cand> cands;
        cands.reserve(all.size());
        for (const WireCand& w : all)
            cands.push_back(Cand{w.count, fingerprint(vocab, w.key), w.key});
        const LowerPrio lower{&vocab};
        std::sort(cands.begin(), cands.end(),
                  [&](const Cand& a, const Cand& b) { return lower(b, a); });
        if (cands.size() > kBatch) cands.resize(kBatch);
        t_sel += us(ts);
        if (cands.empty()) break;

        // the ones that lost go back into their owner's queue
        auto tb = std::chrono::steady_clock::now();
        std::unordered_set<u64> won;
        for (const Cand& c : cands) won.insert(c.key);
#pragma omp parallel num_threads(T)
        {
            const int t = omp_get_thread_num();
            for (const Cand& c : popped[t])
                if (!won.count(c.key)) owners[t].give_back(c);
        }

        t_back += us(tb);
        // new tokens in batch order, same ids on every rank
        auto tm = std::chrono::steady_clock::now();
        const u32 first_id = u32(vocab.text.size());
        for (const Cand& c : cands) vocab.text.push_back(vocab.merged(c.key));
#pragma omp parallel num_threads(T)
        {
            const int t = omp_get_thread_num();
            for (std::size_t i = 0; i < cands.size(); ++i)
                workers[t].merge(cands[i].key, first_id + u32(i));
            workers[t].take_deltas(deltas[t]);
        }
        t_merge += us(tm);
        auto tr = std::chrono::steady_clock::now();
        route_deltas(deltas, owners);
        t_route += us(tr);
        ++rounds;
    }

    const auto t_loop = ms(t0);
    // add up the token counts of all workers on all ranks
    std::vector<u64> cnt(vocab.text.size(), 0);
    for (int t = 0; t < T; ++t) {
        const std::vector<u64>& c = workers[t].token_count();
        for (std::size_t i = 0; i < c.size() && i < cnt.size(); ++i) cnt[i] += c[i];
    }
#ifdef BPE_HAVE_MPI
    if (mpi_size() > 1) {
        std::vector<u64> sum(cnt.size(), 0);
        MPI_Reduce(cnt.data(), sum.data(), int(cnt.size()), MPI_UNSIGNED_LONG_LONG,
                   MPI_SUM, 0, MPI_COMM_WORLD);
        cnt.swap(sum);
    }
#endif
    results.tokens.clear();
    if (me == 0) {
        std::vector<u32> live;
        for (u32 t = 1; t < cnt.size(); ++t)
            if (cnt[t] != 0) live.push_back(t);
        std::sort(live.begin(), live.end(), [&](u32 a, u32 b) {
            if (cnt[a] != cnt[b]) return cnt[a] > cnt[b];
            return std::strcmp(vocab.text[a].c_str(), vocab.text[b].c_str()) < 0;
        });
        for (u32 t : live) {
            const std::string& s = vocab.text[t];
            results.tokens.push_back(TokenCount{std::vector<Byte>(s.begin(), s.end()),
                                                std::size_t(cnt[t])});
        }
        LOG(INFO) << "task2: " << rounds << " rounds; build " << t_build
                  << " ms, select " << t_sel / 1000 << " ms, give back " << t_back / 1000
                  << " ms, merge " << t_merge / 1000 << " ms, route " << t_route / 1000
                  << " ms, loop " << t_loop << " ms, total " << ms(t0) << " ms";
    }
    // free the big structures in parallel, one thread each, instead of one
    // thread walking millions of list nodes when the function returns
    const auto tf = std::chrono::steady_clock::now();
#pragma omp parallel num_threads(T)
    {
        const int t = omp_get_thread_num();
        Worker gone_w;
        std::swap(workers[t], gone_w);
        Owner gone_o(&vocab);
        std::swap(owners[t], gone_o);
    }
    if (me == 0) LOG(INFO) << "task2: teardown " << ms(tf) << " ms";
}

}
