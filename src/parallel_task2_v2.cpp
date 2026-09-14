#include "bpe.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace bpe {
namespace {

using u8=std::uint8_t;
using u32=std::uint32_t;
using u64=std::uint64_t;

constexpr u32 no_position=std::numeric_limits<u32>::max();
constexpr u32 byte_value_count =256;
constexpr std::size_t kParallelThreshold=20000;

u64 pack_pair(u32 left,u32 right) {
    return (static_cast<u64>(left) << 32) | static_cast<u64>(right);
}
u32 pair_left(u64 key) { 
    return static_cast<u32>(key >> 32); 
}
u32 pair_right(u64 key) { 
    return static_cast<u32>(key); 
}

u64 text_fingerprint(const std::string& left, const std::string& right) {
    u64 fingerprint=0;
    std::size_t i=0;
    const std::size_t n = std::min<std::size_t>(left.size(), 8);
    for (; i<n;++i) {
        fingerprint |= static_cast<u64>(static_cast<u8>(left[i])) << (8*(7-i));
    }
    const std::size_t m = std::min<std::size_t>(right.size(), 8-i);
    for (std::size_t j=0; j<m; ++j,++i) {
        fingerprint |= static_cast<u64>(static_cast<u8>(right[j])) << (8*(7-i));
    }
    return fingerprint;
}

struct PairRecord {
    u64 count=0;
    u32 word_count=0;
    u64 fingerprint=0;
    std::vector<u32> positions;
    std::unordered_map<u32,u32> word_edges;
};

struct State2 {
    std::vector<u64> word_frequencies;
    std::vector<u32> token;
    std::vector<u32> previous;
    std::vector<u32> next;
    std::vector<u32> word_of;
    std::vector<u8> alive;
    std::vector<u64> token_count;
    std::vector<std::string> vocabulary;
    std::unordered_map<u64, PairRecord> pairs;
    std::vector<u64> born_keys;
};

bool is_live(const State2& s, u32 position) {
    return position !=no_position && s.alive[position] != 0;
}

PairRecord& get_pair(State2& s, u64 key) {
    auto [it, inserted]=s.pairs.try_emplace(key);
    if (inserted) {
        it->second.fingerprint = text_fingerprint(s.vocabulary[pair_left(key)],s.vocabulary[pair_right(key)]);
        s.born_keys.push_back(key);
    }
    return it->second;
}

void remove_edge_v2(State2& s,u32 left,u32 right,u32 word,u64 frequency) {
    PairRecord& rec = get_pair(s, pack_pair(left,right));
    if (rec.count < frequency) {
        std::abort();
    }
    rec.count-= frequency;
    auto it=rec.word_edges.find(word);
    if (it==rec.word_edges.end() ||it->second == 0) {
        std::abort();
    }
    if (--(it->second)==0) {
        if (rec.word_count== 0) {
            std::abort();
        }
        --rec.word_count;
        rec.word_edges.erase(it);
    }
}

void add_edge_v2(State2& s,u32 left,u32 right,u32 word,u32 position,u64 frequency) {
    PairRecord& rec=get_pair(s, pack_pair(left, right));
    u32& live= rec.word_edges[word];
    if (live==0) {
        ++rec.word_count;
    }
    ++live;
    rec.count += frequency;
    rec.positions.push_back(position);
}

bool pair_is_at(const State2& s,u32 position,u32 left,u32 right) {
    if (!is_live(s, position)||s.token[position] != left) {
        return false;
    }
    const u32 next_position=s.next[position];
    return is_live(s,next_position) && s.token[next_position] == right;
}

struct QueueEntry {
    u64 count=0;
    u64 fingerprint=0;
    u64 key=0;
};

struct QueueCompare {
    const State2* state = nullptr;
    bool operator()(const QueueEntry& left, const QueueEntry& right) const;
};

struct QueueHeap {
    std::vector<QueueEntry> data;
    QueueCompare comp;

    bool empty() const { 
        return data.empty(); 
    }
    const QueueEntry& top() const { 
        return data.front(); 
    }

    void push(QueueEntry entry) {
        data.push_back(entry);
        sift_up(data.size() - 1);
    }

    void pop() {
        data[0] = std::move(data.back());
        data.pop_back();
        if (!data.empty()) {
            sift_down(0);
        }
    }

   private:
    static std::size_t parent(std::size_t i) { return (i-1)/4; }
    static std::size_t first_child(std::size_t i) { return 4*i+1; }

    void sift_up(std::size_t i) {
        while (i > 0) {
            const std::size_t p = parent(i);
            if (!comp(data[p], data[i])) break;
            std::swap(data[p], data[i]);
            i = p;
        }
    }
    void sift_down(std::size_t i) {
        for (;;) {
            const std::size_t fc = first_child(i);
            if (fc >= data.size()) break;
            std::size_t best = fc;
            const std::size_t end = std::min(fc + 4, data.size());
            for (std::size_t c = fc + 1; c < end; ++c) {
                if (comp(data[best], data[c])) best = c;
            }
            if (!comp(data[i], data[best])) break;
            std::swap(data[i], data[best]);
            i = best;
        }
    }
};

bool QueueCompare::operator()(const QueueEntry& left,const QueueEntry& right) const {
    if (left.count!= right.count) {
        return left.count < right.count;
    }
    if (left.fingerprint!= right.fingerprint) {
        return left.fingerprint > right.fingerprint;
    }
    const std::string left_text=state->vocabulary[pair_left(left.key)]+state->vocabulary[pair_right(left.key)];
    const std::string right_text=state->vocabulary[pair_left(right.key)]+state->vocabulary[pair_right(right.key)];
    return std::strcmp(left_text.c_str(), right_text.c_str())> 0;
}

void push_born_states(State2& state, QueueHeap& queue) {
    for (u64 key : state.born_keys) {
        const PairRecord& rec=state.pairs[key];
        if (rec.word_count>=2 && rec.count != 0) {
            queue.push(QueueEntry{rec.count,rec.fingerprint, key});
        }
    }
    state.born_keys.clear();
}

constexpr u64 kNoKey = ~0ULL;

u64 pop_best_key(State2& state, QueueHeap& queue) {
    while (!queue.empty()) {
        const QueueEntry entry = queue.top();
        queue.pop();
        const auto it = state.pairs.find(entry.key);
        if (it == state.pairs.end()) {
            continue;
        }
        const PairRecord& rec = it->second;
        if (rec.word_count<2 || rec.count == 0) {
            continue;
        }
        if (entry.count != rec.count) {
            queue.push(QueueEntry{rec.count, rec.fingerprint, entry.key});
            continue;
        }
        return entry.key;
    }
    return kNoKey;
}

void build_state_v2(const std::vector<CharSplit>& splits, State2& state) {
    if (splits.size() >= no_position) {
        throw std::length_error("too many distinct words");
    }

    state.vocabulary.resize(byte_value_count);
    state.token_count.assign(byte_value_count, 0);
    for (u32 value = 1; value < byte_value_count;++value) {
        state.vocabulary[value].assign(1, static_cast<char>(value));
    }

    state.word_frequencies.reserve(splits.size());
    std::size_t slot_count = 0;
    for (const CharSplit& split : splits) {
        slot_count+=split.chars.size() + 1;
    }
    if (slot_count >= no_position) {
        throw std::length_error("input has too many byte positions");
    }
    state.token.reserve(slot_count);
    state.previous.reserve(slot_count);
    state.next.reserve(slot_count);
    state.word_of.reserve(slot_count);
    state.alive.reserve(slot_count);

    for (u32 word=0; word < splits.size();++word) {
        const CharSplit& split = splits[word];
        state.word_frequencies.push_back(split.count);
        const u32 first = static_cast<u32>(state.token.size());

        for (std::size_t index=0; index < split.chars.size(); ++index) {
            const u32 position=static_cast<u32>(state.token.size());
            const u32 value=split.chars[index];
            if (value==0) {
                throw std::invalid_argument("word contains a NUL byte");
            }
            state.token.push_back(value);
            state.previous.push_back(index == 0 ? no_position : position - 1);
            state.next.push_back(position + 1);
            state.word_of.push_back(word);
            state.alive.push_back(1);
            state.token_count[value] += split.count;
        }

        const u32 sentinel=static_cast<u32>(state.token.size());
        state.token.push_back(0);
        state.previous.push_back(split.chars.empty() ? no_position : sentinel - 1);
        state.next.push_back(no_position);
        state.word_of.push_back(word);
        state.alive.push_back(0);

        if (split.chars.empty()) {
            continue;
        }
        for (u32 position=first; is_live(state, position);position = state.next[position]) {
            const u32 next_position = state.next[position];
            if (!is_live(state, next_position)) {
                break;
            }
            add_edge_v2(state, state.token[position],state.token[next_position],word, position, split.count);
        }
    }
    state.born_keys.clear();
}

void process_positions_v2(State2& state,const std::vector<u32>& positions,u32 left_token,u32 right_token,u32 merged_token,bool use_parallel) {
    std::unordered_map<u32, std::vector<u32>> by_word;
    by_word.reserve(positions.size());
    for (u32 position : positions) {
        by_word[state.word_of[position]].push_back(position);
    }
    std::vector<u32> active_words;
    active_words.reserve(by_word.size());
    for (const auto& entry : by_word) {
        active_words.push_back(entry.first);
    }

    auto process_word = [&](u32 word) {
        const u64 frequency = state.word_frequencies[word];
        for (u32 position : by_word.at(word)) {
            if (!pair_is_at(state, position, left_token, right_token)) {
                continue;
            }
            const u32 right_position=state.next[position];
            const u32 left_position=state.previous[position];
            const u32 after_position= state.next[right_position];
            const bool has_left=is_live(state,left_position);
            const bool has_after=is_live(state,after_position);
            const u32 left_neighbor=has_left ? state.token[left_position] : no_position;
            const u32 after_neighbor = has_after ? state.token[after_position] : no_position;

            #pragma omp critical(pair_registry)
            {
                if (has_left) {
                    remove_edge_v2(state, left_neighbor, left_token, word, frequency);
                }
                remove_edge_v2(state,left_token,right_token, word, frequency);
                if (has_after) {
                    remove_edge_v2(state, right_token, after_neighbor, word, frequency);
                }
                state.token_count[left_token] -= frequency;
                state.token_count[right_token] -= frequency;
                state.token_count[merged_token] += frequency;
            }

            state.token[position]=merged_token;
            state.alive[right_position]=0;
            state.next[position] = after_position;
            if (after_position != no_position) {
                state.previous[after_position] = position;
            }
            state.previous[right_position] = no_position;
            state.next[right_position]=no_position;

            #pragma omp critical(pair_registry)
            {
                if (has_left) {
                    add_edge_v2(state,left_neighbor,merged_token, word,left_position,frequency);
                }
                if (has_after) {
                    add_edge_v2(state,merged_token,after_neighbor,word,position,frequency);
                }
            }
        }
    };

    if (use_parallel) {
        #pragma omp parallel for schedule(dynamic, 64)
        for (std::size_t idx=0; idx < active_words.size(); ++idx) {
            process_word(active_words[idx]);
        }
    } else {
        for (u32 word : active_words) {
            process_word(word);
        }
    }
}

void run_merge_loop_v2(State2& state) {
    QueueHeap queue;
    queue.comp.state=&state;
    for (const auto& entry : state.pairs) {
        const PairRecord& rec = entry.second;
        if (rec.word_count >= 2) {
            queue.push(QueueEntry{rec.count, rec.fingerprint, entry.first});
        }
    }

    for (;;) {
        const u64 best_key = pop_best_key(state,queue);
        if (best_key == kNoKey) {
            break;
        }
        std::vector<u32> positions=std::move(state.pairs[best_key].positions);
        const u32 left_token=pair_left(best_key);
        const u32 right_token=pair_right(best_key);
        const u32 merged_token = static_cast<u32>(state.vocabulary.size());
        std::string merged_text;
        merged_text.reserve(state.vocabulary[left_token].size()+state.vocabulary[right_token].size());
        merged_text.append(state.vocabulary[left_token]);
        merged_text.append(state.vocabulary[right_token]);
        state.vocabulary.push_back(std::move(merged_text));
        state.token_count.push_back(0);
        const bool use_parallel =positions.size() >= kParallelThreshold && omp_get_max_threads() > 1;
        process_positions_v2(state,positions,left_token,right_token,merged_token, use_parallel);
        push_born_states(state, queue);
    }
}

void finalize_results_v2(const State2& state,Results& results) {
    std::vector<u32> live_tokens;
    live_tokens.reserve(state.token_count.size());
    for (u32 token_id = 1; token_id < state.token_count.size(); ++token_id) {
        if (state.token_count[token_id] != 0) {
            live_tokens.push_back(token_id);
        }
    }
    std::sort(live_tokens.begin(), live_tokens.end(),
              [&state](u32 left, u32 right) {
                  if (state.token_count[left] !=state.token_count[right]) {
                      return state.token_count[left]>state.token_count[right];
                  }
                  return std::strcmp(state.vocabulary[left].c_str(),state.vocabulary[right].c_str()) < 0;
              });

    results.tokens.clear();
    results.tokens.reserve(live_tokens.size());
    for (u32 token_id : live_tokens) {
        const std::string& text=state.vocabulary[token_id];
        results.tokens.push_back(TokenCount{std::vector<Byte>(text.begin(), text.end()),static_cast<std::size_t>(state.token_count[token_id])});
    }
}

}  // namespace

void parallel_task2_v2(const std::vector<CharSplit>& splits, Results& results) {
    State2 state;
    build_state_v2(splits,state);
    run_merge_loop_v2(state);
    finalize_results_v2(state,results);
}

}  // namespace bpe
