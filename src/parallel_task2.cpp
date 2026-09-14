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
#include <utility>
#include <vector>

namespace bpe {
    namespace {
        using u8 = std::uint8_t;
        using u32 = std::uint32_t;
        using u64=std::uint64_t;
        constexpr u32 no_position = std::numeric_limits<u32>::max();
        constexpr u32 byte_value_count = 256;
        constexpr std::size_t kParallelThreshold = 2000;
    }
u64 pack_pair(u32 left, u32 right) {
    return (static_cast<u64>(left) << 32) | static_cast<u64>(right);
}

u32 pair_left(u64 key) { return static_cast<u32>(key >> 32); }

u32 pair_right(u64 key) { return static_cast<u32>(key); }

u64 text_fingerprint(const std::string& left, const std::string& right) {
    u64 fingerprint = 0;
    std::size_t i = 0;
    const std::size_t n = std::min<std::size_t>(left.size(), 8);
    for (; i < n; ++i) {
        fingerprint |= static_cast<u64>(static_cast<u8>(left[i]))
                       << (8 * (7 - i));
    }
    const std::size_t m = std::min<std::size_t>(right.size(), 8 - i);
    for (std::size_t j = 0; j < m; ++j, ++i) {
        fingerprint |= static_cast<u64>(static_cast<u8>(right[j]))
                       << (8 * (7 - i));
    }
    return fingerprint;
}

struct pair_state {
    u64 key = 0;
    u64 count = 0;
    u32 word_count = 0;
    std::vector<u32> positions;
    u32 last_word = no_position;
    u32 last_group = no_position;
    u64 fingerprint = 0;
};

struct queue_entry {
    u64 count = 0;
    u64 fingerprint = 0;
    u32 state = 0;
};

struct task2_state;

struct queue_compare {
    const task2_state* state = nullptr;

    bool operator()(const queue_entry& left, const queue_entry& right) const;
};

struct queue_heap {
    std::vector<queue_entry> data;
    queue_compare comp;

    bool empty() const { return data.empty(); }

    const queue_entry& top() const { return data.front(); }

    void push(queue_entry entry) {
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
    static std::size_t parent(std::size_t i) { return (i - 1) / 4; }
    static std::size_t first_child(std::size_t i) { return 4 * i + 1; }

    void sift_up(std::size_t i) {
        while (i > 0) {
            const std::size_t p = parent(i);
            if (!comp(data[p], data[i])) {
                break;
            }
            std::swap(data[p], data[i]);
            i = p;
        }
    }

    void sift_down(std::size_t i) {
        for (;;) {
            const std::size_t fc = first_child(i);
            if (fc >= data.size()) {
                break;
            }
            std::size_t best = fc;
            const std::size_t end = std::min(fc + 4, data.size());
            for (std::size_t c = fc + 1; c < end; ++c) {
                if (comp(data[best], data[c])) {
                    best = c;
                }
            }
            if (!comp(data[i], data[best])) {
                break;
            }
            std::swap(data[i], data[best]);
            i = best;
        }
    }
};

struct task2_state {
    std::vector<u64> word_frequencies;
    std::vector<u32> token;
    std::vector<u32> previous;
    std::vector<u32> next;
    std::vector<u32> word_of;
    std::vector<u32> edge_group;
    std::vector<u8> alive;
    std::vector<u64> token_count;
    std::vector<std::string> vocabulary;
    std::vector<pair_state> pair_states;
    std::vector<u32> group_state;
    std::vector<u32> group_count;
    std::vector<u32> left_stamp;
    std::vector<u32> left_state;
    std::vector<u32> right_stamp;
    std::vector<u32> right_state;
    u32 same_stamp = no_position;
    u32 same_state = no_position;
    std::vector<u32> born_states;
    u32 free_head = no_position;
};

bool queue_compare::operator()(const queue_entry& left,
                               const queue_entry& right) const {
    if (left.count != right.count) {
        return left.count < right.count;
    }
    if (left.fingerprint != right.fingerprint) {
        return left.fingerprint > right.fingerprint;
    }
    const pair_state& a = state->pair_states[left.state];
    const pair_state& b = state->pair_states[right.state];
    const std::string left_text = state->vocabulary[pair_left(a.key)] +
                                  state->vocabulary[pair_right(a.key)];
    const std::string right_text = state->vocabulary[pair_left(b.key)] +
                                   state->vocabulary[pair_right(b.key)];
    return std::strcmp(left_text.c_str(), right_text.c_str()) > 0;
}

bool is_live(const task2_state& state, u32 position) {
    return position != no_position && state.alive[position] != 0;
}

u32 create_pair_state(task2_state& state, u64 key) {
    const u32 state_id = static_cast<u32>(state.pair_states.size());
    state.pair_states.push_back(pair_state{});
    pair_state& pair = state.pair_states.back();
    pair.key = key;
    pair.fingerprint = text_fingerprint(state.vocabulary[pair_left(key)],
                                        state.vocabulary[pair_right(key)]);
    state.born_states.push_back(state_id);
    return state_id;
}

u32 get_left_pair_state(task2_state& state, u32 left, u32 merged) {
    if (left == merged) {
        if (state.same_stamp != merged) {
            state.same_stamp = merged;
            state.same_state =
                create_pair_state(state, pack_pair(merged, merged));
        }
        return state.same_state;
    }
    if (state.left_stamp[left] != merged) {
        state.left_stamp[left] = merged;
        state.left_state[left] =
            create_pair_state(state, pack_pair(left, merged));
    }
    return state.left_state[left];
}

u32 get_right_pair_state(task2_state& state, u32 merged, u32 right) {
    if (right == merged) {
        if (state.same_stamp != merged) {
            state.same_stamp = merged;
            state.same_state =
                create_pair_state(state, pack_pair(merged, merged));
        }
        return state.same_state;
    }
    if (state.right_stamp[right] != merged) {
        state.right_stamp[right] = merged;
        state.right_state[right] =
            create_pair_state(state, pack_pair(merged, right));
    }
    return state.right_state[right];
}

u32 get_word_group(task2_state& state, u32 state_id, u32 word) {
    pair_state& pair = state.pair_states[state_id];
    if (pair.last_word == word) {
        return pair.last_group;
    }
    u32 group;
    if (state.free_head != no_position) {
        group = state.free_head;
        state.free_head = state.group_count[group];
        state.group_count[group] = 0;
        state.group_state[group] = state_id;
    } else {
        group = static_cast<u32>(state.group_state.size());
        state.group_state.push_back(state_id);
        state.group_count.push_back(0);
    }
    pair.last_word = word;
    pair.last_group = group;
    return group;
}

void remove_edge(task2_state& state, u32 start, u64 frequency) {
    const u32 group = state.edge_group[start];
    if (group == no_position || state.group_count[group] == 0) {
        std::abort();
    }
    pair_state& pair = state.pair_states[state.group_state[group]];
    --state.group_count[group];
    if (pair.count < frequency) {
        std::abort();
    }
    pair.count -= frequency;
    if (state.group_count[group] == 0) {
        if (pair.word_count == 0) {
            std::abort();
        }
        --pair.word_count;
        if (pair.last_group == group) {
            pair.last_group = no_position;
            pair.last_word = no_position;
        }
        state.group_count[group] = state.free_head;
        state.free_head = group;
    }
    state.edge_group[start] = no_position;
}

void add_edge(task2_state& state, u32 start, u32 state_id, u32 word,
              u64 frequency) {
    const u32 group = get_word_group(state, state_id, word);
    pair_state& pair = state.pair_states[state_id];
    if (state.group_count[group]++ == 0) {
        ++pair.word_count;
    }
    pair.count += frequency;
    pair.positions.push_back(start);
    state.edge_group[start] = group;
}

bool pair_is_at(const task2_state& state, u32 position, u32 left, u32 right) {
    if (!is_live(state, position) || state.token[position] != left) {
        return false;
    }
    const u32 next_position = state.next[position];
    return is_live(state, next_position) && state.token[next_position] == right;
}

void push_born_states(task2_state& state, queue_heap& queue) {
    for (u32 state_id : state.born_states) {
        const pair_state& pair = state.pair_states[state_id];
        if (pair.word_count >= 2 && pair.count != 0) {
            queue.push(queue_entry{pair.count, pair.fingerprint, state_id});
        }
    }
    state.born_states.clear();
}

u32 pop_best_state(task2_state& state, queue_heap& queue) {
    while (!queue.empty()) {
        const queue_entry entry = queue.top();
        queue.pop();
        const pair_state& pair = state.pair_states[entry.state];
        if (pair.word_count < 2 || pair.count == 0) {
            continue;
        }
        if (entry.count != pair.count) {
            queue.push(queue_entry{pair.count, pair.fingerprint, entry.state});
            continue;
        }
        return entry.state;
    }
    return no_position;
}

void build_state(const std::vector<CharSplit>& splits, task2_state& state) {
    if (splits.size() >= no_position) {
        throw std::length_error("too many distinct words");
    }

    state.vocabulary.resize(byte_value_count);
    state.token_count.assign(byte_value_count, 0);
    for (u32 value = 1; value < byte_value_count; ++value) {
        state.vocabulary[value].assign(1, static_cast<char>(value));
    }

    state.word_frequencies.reserve(splits.size());
    std::size_t slot_count = 0;
    for (const CharSplit& split : splits) {
        slot_count += split.chars.size() + 1;
    }
    if (slot_count >= no_position) {
        throw std::length_error("input has too many byte positions");
    }
    state.token.reserve(slot_count);
    state.previous.reserve(slot_count);
    state.next.reserve(slot_count);
    state.word_of.reserve(slot_count);
    state.edge_group.reserve(slot_count);
    state.alive.reserve(slot_count);
    state.group_state.reserve(slot_count);
    state.group_count.reserve(slot_count);

    std::vector<u32> initial_state(byte_value_count * byte_value_count,
                                   no_position);
    for (u32 word = 0; word < splits.size(); ++word) {
        const CharSplit& split = splits[word];
        state.word_frequencies.push_back(split.count);
        const u32 first = static_cast<u32>(state.token.size());

        for (std::size_t index = 0; index < split.chars.size(); ++index) {
            const u32 position = static_cast<u32>(state.token.size());
            const u32 value = split.chars[index];
            if (value == 0) {
                throw std::invalid_argument("word contains a NUL byte");
            }
            state.token.push_back(value);
            state.previous.push_back(index == 0 ? no_position : position - 1);
            state.next.push_back(position + 1);
            state.word_of.push_back(word);
            state.edge_group.push_back(no_position);
            state.alive.push_back(1);
            state.token_count[value] += split.count;
        }

        const u32 sentinel = static_cast<u32>(state.token.size());
        state.token.push_back(0);
        state.previous.push_back(split.chars.empty() ? no_position
                                                     : sentinel - 1);
        state.next.push_back(no_position);
        state.word_of.push_back(word);
        state.edge_group.push_back(no_position);
        state.alive.push_back(0);

        if (split.chars.empty()) {
            continue;
        }
        for (u32 position = first; is_live(state, position);
             position = state.next[position]) {
            const u32 next_position = state.next[position];
            if (!is_live(state, next_position)) {
                break;
            }
            const u32 left = state.token[position];
            const u32 right = state.token[next_position];
            u32& state_id = initial_state[left * byte_value_count + right];
            if (state_id == no_position) {
                state_id = create_pair_state(state, pack_pair(left, right));
            }

            pair_state& pair = state.pair_states[state_id];
            u32 group = no_position;
            if (pair.last_word == word) {
                group = pair.last_group;
            } else {
                group = static_cast<u32>(state.group_state.size());
                state.group_state.push_back(state_id);
                state.group_count.push_back(0);
                pair.last_word = word;
                pair.last_group = group;
                ++pair.word_count;
            }
            ++state.group_count[group];
            pair.count += split.count;
            pair.positions.push_back(position);
            state.edge_group[position] = group;
        }
    }

    state.born_states.clear();
    const std::size_t cache_size = state.vocabulary.size() + 1024;
    state.left_stamp.assign(cache_size, no_position);
    state.left_state.assign(cache_size, no_position);
    state.right_stamp.assign(cache_size, no_position);
    state.right_state.assign(cache_size, no_position);
}

struct MergeEvent{
    u32 word = no_position;
    u64 frequency = 0;
    u32 left_position = no_position;
    u32 position = no_position;
    u32 right_position = no_position;
    u32 after_position = no_position;
    u32 left_token = no_position;
    u32 after_token = no_position;
};

void process_positions_sequential(task2_state& state, const std::vector<u32>& positions, u32 left_token, u32 right_token, u32 merged_token) {
    u32 current_word = no_position;
    u64 frequency = 0;
    for (u32 position : positions) {
        if (!pair_is_at(state, position, left_token, right_token)) {
            continue;
        }
        const u32 right_position = state.next[position];
        const u32 word = state.word_of[position];
        if (word != current_word) {
            current_word = word;
            frequency = state.word_frequencies[word];
        }
        const u32 left_position = state.previous[position];
        const u32 after_position = state.next[right_position];

        if (is_live(state, left_position)) {
            remove_edge(state, left_position, frequency);
        }
        remove_edge(state, position, frequency);
        if (is_live(state, after_position)) {
            remove_edge(state, right_position, frequency);
        }

        state.token_count[left_token] -= frequency;
        state.token_count[right_token] -= frequency;
        state.token_count[merged_token] += frequency;
        state.token[position] = merged_token;
        state.alive[right_position] = 0;
        state.next[position] = after_position;
        if (after_position != no_position) {
            state.previous[after_position] = position;
        }
        state.previous[right_position] = no_position;
        state.next[right_position] = no_position;
        state.edge_group[right_position] = no_position;

        if (is_live(state, left_position)) {
            const u32 state_id = get_left_pair_state(
                state, state.token[left_position], merged_token);
            add_edge(state, left_position, state_id, word, frequency);
        }
        if (is_live(state, after_position)) {
            const u32 state_id = get_right_pair_state(
                state, merged_token, state.token[after_position]);
            add_edge(state, position, state_id, word, frequency);
        }
    }
}

void process_positions_parallel(task2_state& state, const std::vector<u32>& positions, u32 left_token, u32 right_token, u32 merged_token) {
    std::unordered_map<u32, std::vector<u32>> by_word;
    by_word.reserve(positions.size());
    for (u32 position : positions) {
        by_word[state.word_of[position]].push_back(position);
    }
    std::vector<u32> active_words;
    active_words.reserve(by_word.size());
    for(const auto& entry:by_word){
        active_words.push_back(entry.first);
    }
    const int num_gthreads=omp_get_max_threads();
    std::vector<std::vector<MergeEvent>> local_events(num_threads);

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        std::vector<MergeEvent>& events = local_events[tid];

    }

}




}
