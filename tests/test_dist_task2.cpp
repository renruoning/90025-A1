// The worker-based pipeline has to agree with the reference task1 + task2.

#include "bpe.h"
#include "dist.h"

#include "gtest/gtest.h"

#include <omp.h>
#include <random>
#include <string>
#include <vector>

namespace {

std::vector<bpe::Byte> make_text(unsigned seed, const std::string& alpha,
                                 int vocab, int words) {
    std::mt19937 rng(seed);
    std::vector<std::string> pool;
    for (int i = 0; i < vocab; ++i) {
        std::string w;
        const int len = 1 + rng() % 12;
        for (int j = 0; j < len; ++j) w += alpha[rng() % alpha.size()];
        pool.push_back(w);
    }
    std::string text;
    for (int i = 0; i < words; ++i) text += pool[rng() % pool.size()] + " ";
    return std::vector<bpe::Byte>(text.begin(), text.end());
}

void expect_same(std::vector<bpe::Byte> text, int threads) {
    bpe::Results ref, mine;
    {
        std::vector<bpe::Byte> copy = text;
        const std::vector<bpe::Word> words = bpe::split_words(copy);
        bpe::task1(words, ref);
        bpe::task2(ref.char_splits, ref);
    }
    const int old = omp_get_max_threads();
    omp_set_num_threads(threads);
    bpe::distributed_task1(text, mine);
    bpe::distributed_task2(mine.char_splits, mine);
    omp_set_num_threads(old);
    ASSERT_EQ(ref.tokens.size(), mine.tokens.size());
    for (std::size_t i = 0; i < ref.tokens.size(); ++i) {
        EXPECT_EQ(ref.tokens[i].token, mine.tokens[i].token) << "line " << i;
        EXPECT_EQ(ref.tokens[i].count, mine.tokens[i].count) << "line " << i;
    }
}

}  // namespace

TEST(DistTask2, OneWorker) { expect_same(make_text(1, "abc", 200, 3000), 1); }

TEST(DistTask2, TinyAlphabet) {
    for (unsigned seed = 0; seed < 10; ++seed)
        expect_same(make_text(seed, "ab", 100, 2000), 4);
}

TEST(DistTask2, ManyRounds) {
    expect_same(make_text(99, "abcdefgh", 4000, 40000), 7);
}

TEST(DistTask2, OverlappingRuns) {
    std::string s;
    for (int n = 2; n < 30; ++n) s += std::string(n, 'a') + " ";
    expect_same(std::vector<bpe::Byte>(s.begin(), s.end()), 3);
}

TEST(DistTask2, Empty) { expect_same({}, 4); }
