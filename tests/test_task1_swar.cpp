// Tests for the deep branches of Task 1's chunked SWAR hash/equality, reached
// through the public task1() entry point.
//
// The counting map keys words by buffer pointer and compares them with a
// chunked hash/equality pair (see src/task1.cpp). Several of their branches
// only fire on specific inputs:
//   * words longer than 8 bytes (multi-chunk hashing),
//   * the same word at the same pointer (pointer-identity equality),
//   * a word duplicated at two offsets far from the buffer end (the equality
//     loop's full-chunk advance, then its tail),
//   * two *different* words whose 64-bit FNV-1a hashes collide. This last one
//     matters: unordered_map only calls the equality functor on keys in the
//     same bucket, so the functor's "bytes differ in a full chunk" branch is
//     only reachable through a genuine hash collision.
//
// The collision pair below is derived, not brute-forced. With P = 0x100000001b3
// and h0 the FNV-1a offset, the word c1 || c2 where
//     c2 = (T * P^-1) ^ ((h0 ^ c1) * P)      (P^-1 the inverse of P mod 2^64)
// hashes to exactly T ^ P, for any chosen 64-bit target T and 8-byte prefix
// c1. The two words below use different prefixes "ABCDEFGH" / "IJKLMNOP" and
// the same T = 0x0123456789abcdef, so they share a hash while their first
// eight bytes differ -- precisely the input that reaches the functor's
// full-chunk mismatch branch. The words contain no NUL bytes (words are
// NUL-terminated, so an interior NUL would truncate them).

#include "bpe.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace {

// A corpus of words backed by one contiguous NUL-terminated buffer, the same
// shape test_task1.cpp uses.
struct Corpus {
    std::vector<bpe::Byte> buffer;
    std::vector<bpe::Word> words;
};

Corpus make_corpus(const std::vector<std::string>& word_list) {
    Corpus c;
    for (const std::string& w : word_list) {
        c.buffer.insert(c.buffer.end(), w.begin(), w.end());
        c.buffer.push_back(static_cast<bpe::Byte>('\0'));
    }
    std::size_t pos = 0;
    for (const std::string& w : word_list) {
        c.words.push_back(bpe::Word{&c.buffer[pos]});
        pos += w.size() + 1;
    }
    return c;
}

using StrCount = std::pair<std::string, std::size_t>;

// The specification does not prescribe the order of the Task 1 results, so the
// helper sorts before comparing.
std::vector<StrCount> word_counts(const bpe::Results& r) {
    std::vector<StrCount> out;
    for (const bpe::WordCount& wc : r.word_counts) {
        out.emplace_back(std::string(wc.word.begin(), wc.word.end()), wc.count);
    }
    std::sort(out.begin(), out.end());
    return out;
}

// The two colliding words derived above (16 bytes each, NUL-free).
std::string colliding_w() {
    static const char bytes[] = {'A',
                                 'B',
                                 'C',
                                 'D',
                                 'E',
                                 'F',
                                 'G',
                                 'H',
                                 static_cast<char>(0x73),
                                 static_cast<char>(0xb0),
                                 static_cast<char>(0xbb),
                                 static_cast<char>(0x87),
                                 'Z',
                                 'i',
                                 '?',
                                 'm'};
    return std::string(bytes, sizeof bytes);
}
std::string colliding_v() {
    static const char bytes[] = {'I',
                                 'J',
                                 'K',
                                 'L',
                                 'M',
                                 'N',
                                 'O',
                                 'P',
                                 static_cast<char>(0xeb),
                                 static_cast<char>(0x6e),
                                 static_cast<char>(0x0d),
                                 static_cast<char>(0x10),
                                 static_cast<char>(0xa4),
                                 static_cast<char>(0xef),
                                 static_cast<char>(0xad),
                                 static_cast<char>(0xab)};
    return std::string(bytes, sizeof bytes);
}

}  // namespace

// A word longer than 8 bytes whose first chunk holds no NUL: the chunked hash
// must mix the whole first chunk before it finds the end in a later one.
TEST(Task1, LongWordMultiChunkHash) {
    bpe::Results r;
    Corpus c = make_corpus({"abcdefghij", "pad"});
    bpe::task1(c.words, r);
    EXPECT_EQ(word_counts(r),
              std::vector<StrCount>({{"abcdefghij", 1}, {"pad", 1}}));
}

// The same word at the same pointer twice: the map must count it twice and the
// equality functor's pointer-identity fast path decides it.
TEST(Task1, SamePointerWordCountedTwice) {
    bpe::Results r;
    Corpus c = make_corpus({"xx"});
    const std::vector<bpe::Word> words{c.words[0], c.words[0]};
    bpe::task1(words, r);
    EXPECT_EQ(word_counts(r), std::vector<StrCount>({{"xx", 2}}));
}

// One long word duplicated at two offsets far from the buffer end: the equality
// loop must advance through a full NUL-free chunk (comparing both copies byte
// for byte) and then finish through its tail path.
TEST(Task1, LongDuplicateChunkAdvanceAndTail) {
    bpe::Results r;
    Corpus c = make_corpus({"abcdefghij", "abcdefghij", "pad"});
    bpe::task1(c.words, r);
    EXPECT_EQ(word_counts(r),
              std::vector<StrCount>({{"abcdefghij", 2}, {"pad", 1}}));
}

// Two different words whose FNV-1a hashes collide: the counting map must keep
// them as separate entries (the equality functor's full-chunk mismatch branch
// distinguishes them), despite them landing in the same hash bucket.
TEST(Task1, HashCollisionKeepsWordsDistinct) {
    bpe::Results r;
    const std::string w = colliding_w();
    const std::string v = colliding_v();
    Corpus c = make_corpus({w, v, "pad"});
    bpe::task1(c.words, r);
    EXPECT_EQ(word_counts(r),
              std::vector<StrCount>({{w, 1}, {v, 1}, {"pad", 1}}));
}
