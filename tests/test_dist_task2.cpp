// The one-worker 256-pair loop has to agree with the reference task2.

#include "bpe.h"
#include "dist.h"

#include "gtest/gtest.h"

#include <random>
#include <string>
#include <vector>

namespace {

std::vector<bpe::CharSplit> make_splits(unsigned seed, const std::string& alpha,
                                        int words) {
    std::mt19937 rng(seed);
    std::vector<bpe::CharSplit> out;
    for (int i = 0; i < words; ++i) {
        bpe::CharSplit s;
        const int len = 1 + rng() % 12;
        for (int j = 0; j < len; ++j) s.chars.push_back(alpha[rng() % alpha.size()]);
        s.count = 1 + rng() % 50;
        out.push_back(s);
    }
    return out;
}

void expect_same(const std::vector<bpe::CharSplit>& splits) {
    bpe::Results ref, mine;
    bpe::task2(splits, ref);
    bpe::single_worker_task2(splits, mine);
    ASSERT_EQ(ref.tokens.size(), mine.tokens.size());
    for (std::size_t i = 0; i < ref.tokens.size(); ++i) {
        EXPECT_EQ(ref.tokens[i].token, mine.tokens[i].token) << "line " << i;
        EXPECT_EQ(ref.tokens[i].count, mine.tokens[i].count) << "line " << i;
    }
}

}  // namespace

TEST(SingleWorkerTask2, TinyAlphabet) {
    for (unsigned seed = 0; seed < 20; ++seed)
        expect_same(make_splits(seed, "ab", 200));
}

TEST(SingleWorkerTask2, ManyRounds) {
    expect_same(make_splits(99, "abcdefgh", 4000));
}

TEST(SingleWorkerTask2, OverlappingRuns) {
    std::vector<bpe::CharSplit> s;
    for (int n = 2; n < 30; ++n)
        s.push_back({std::vector<bpe::Byte>(n, 'a'), std::size_t(n)});
    expect_same(s);
}

TEST(SingleWorkerTask2, Empty) { expect_same({}); }
