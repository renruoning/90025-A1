// Tests for the corpus split that decides each rank's share.

#include "dist.h"

#include "gtest/gtest.h"

#include <random>
#include <string>
#include <vector>

namespace {

std::vector<bpe::Byte> bytes(const std::string& s) {
    return std::vector<bpe::Byte>(s.begin(), s.end());
}

bool is_sep(bpe::Byte b) {
    return b == ' ' || b == '\t' || b == '\n' || b == '\r' || b == 0;
}

}  // namespace

TEST(SplitPoints, EmptyInput) {
    const std::vector<bpe::Byte> in;
    const auto cuts = bpe::split_points(in.data(), 0, {2, 4, 10});
    EXPECT_EQ(cuts, (std::vector<std::size_t>{0, 0, 0, 0}));
}

TEST(SplitPoints, SingleRankGetsEverything) {
    const auto in = bytes("hug pug");
    const auto cuts = bpe::split_points(in.data(), in.size(), {4});
    EXPECT_EQ(cuts, (std::vector<std::size_t>{0, 7}));
}

TEST(SplitPoints, CutNeverSplitsAWord) {
    std::mt19937 rng(7);
    std::string s;
    for (int i = 0; i < 5000; ++i) {
        s.append(1 + rng() % 12, char('a' + rng() % 26));
        s.push_back(" \t\n\r"[rng() % 4]);
        if (rng() % 5 == 0) s.push_back(' ');
    }
    const auto in = bytes(s);
    for (const std::vector<int>& w :
         {std::vector<int>{4, 4, 4, 4}, {4, 4, 8}, {2, 6, 8}, {2, 4, 10}}) {
        const auto cuts = bpe::split_points(in.data(), in.size(), w);
        ASSERT_EQ(cuts.size(), w.size() + 1);
        EXPECT_EQ(cuts.front(), 0u);
        EXPECT_EQ(cuts.back(), in.size());
        for (std::size_t r = 1; r + 1 < cuts.size(); ++r) {
            EXPECT_LE(cuts[r - 1], cuts[r]);
            const std::size_t c = cuts[r];
            EXPECT_TRUE(c == 0 || c == in.size() || is_sep(in[c]) ||
                        is_sep(in[c - 1]));
        }
    }
}

TEST(SplitPoints, SharesFollowTheWeights) {
    std::string text;
    for (int i = 0; i < 20000; ++i) text += "word ";
    const auto in = bytes(text);
    const auto cuts = bpe::split_points(in.data(), in.size(), {2, 4, 10});
    const double n = static_cast<double>(in.size());
    EXPECT_NEAR((cuts[1] - cuts[0]) / n, 2.0 / 16, 0.01);
    EXPECT_NEAR((cuts[2] - cuts[1]) / n, 4.0 / 16, 0.01);
    EXPECT_NEAR((cuts[3] - cuts[2]) / n, 10.0 / 16, 0.01);
}

TEST(SplitPoints, OneHugeWordStaysWhole) {
    const auto in = bytes(std::string(1000, 'x'));
    const auto cuts = bpe::split_points(in.data(), in.size(), {2, 4, 10});
    EXPECT_EQ(cuts, (std::vector<std::size_t>{0, 1000, 1000, 1000}));
}
