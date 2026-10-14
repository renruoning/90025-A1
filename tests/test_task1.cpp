// Tests for Task 1: word frequency counting and character splitting.

#include "bpe.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

namespace {

// A corpus of words backed by one contiguous NUL-terminated buffer. Words are
// pointers into the buffer, so the buffer must stay alive while they are used.
struct Corpus {
    std::vector<bpe::Byte> buffer;
    std::vector<bpe::Word> words;
};

// Build a corpus from a list of words. The buffer must be complete and stable
// before the word-start pointers are taken, so this is a two-pass build.
Corpus make_corpus(std::initializer_list<std::string> word_list) {
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
// helpers sort before comparing: a correct implementation may emit the distinct
// words in any order.
std::vector<StrCount> word_counts(const bpe::Results& r) {
    std::vector<StrCount> out;
    for (const bpe::WordCount& wc : r.word_counts) {
        out.emplace_back(std::string(wc.word.begin(), wc.word.end()), wc.count);
    }
    std::sort(out.begin(), out.end());
    return out;
}

std::vector<StrCount> char_splits(const bpe::Results& r) {
    std::vector<StrCount> out;
    for (const bpe::CharSplit& cs : r.char_splits) {
        out.emplace_back(std::string(cs.chars.begin(), cs.chars.end()),
                         cs.count);
    }
    std::sort(out.begin(), out.end());
    return out;
}

}  // namespace

TEST(Task1, EmptyCorpus) {
    bpe::Results r;
    bpe::task1({}, r);
    EXPECT_TRUE(r.word_counts.empty());
    EXPECT_TRUE(r.char_splits.empty());
}

TEST(Task1, SingleWordOnce) {
    bpe::Results r;
    Corpus c = make_corpus({"hug"});
    bpe::task1(c.words, r);
    EXPECT_EQ(word_counts(r), std::vector<StrCount>({{"hug", 1}}));
    EXPECT_EQ(char_splits(r), std::vector<StrCount>({{"hug", 1}}));
}

TEST(Task1, CountsWordFrequencies) {
    bpe::Results r;
    Corpus c = make_corpus({"hug", "hug", "pug", "hug", "pun"});
    bpe::task1(c.words, r);
    EXPECT_EQ(word_counts(r),
              std::vector<StrCount>({{"hug", 3}, {"pug", 1}, {"pun", 1}}));
}

TEST(Task1, CharSplitsKeepByteOrder) {
    bpe::Results r;
    Corpus c = make_corpus({"pun"});
    bpe::task1(c.words, r);
    ASSERT_EQ(r.char_splits.size(), 1u);
    EXPECT_EQ(char_splits(r), std::vector<StrCount>({{"pun", 1}}));
}
