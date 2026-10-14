// Tests for Task 2: the BPE merge loop.
//
// These cover the spec's rules that are easy to get wrong:
//   R2.1/R2.2  a pair is eligible only if it appears in >= 2 distinct words;
//   R3.2       pair counts are weighted by word frequency;
//   R5.1       ties are broken by strcmp of the merged string;
//   R5.4       overlapping occurrences merge left to right.

#include "bpe.h"

#include "gtest/gtest.h"

#include <string>
#include <utility>
#include <vector>

namespace {

bpe::CharSplit cs(const std::string& word, std::size_t count) {
    return {std::vector<bpe::Byte>(word.begin(), word.end()), count};
}

using StrCount = std::pair<std::string, std::size_t>;

std::vector<StrCount> tokens_of(const bpe::Results& r) {
    std::vector<StrCount> out;
    for (const bpe::TokenCount& t : r.tokens) {
        out.emplace_back(std::string(t.token.begin(), t.token.end()), t.count);
    }
    return out;
}

}  // namespace

TEST(Task2, EmptyCorpusHasNoTokens) {
    bpe::Results r;
    bpe::task2({}, r);
    EXPECT_TRUE(r.tokens.empty());
}

TEST(Task2, SingleWordIsNeverMerged) {
    // R2.2: a pair occurring in a single distinct word is never merged, no
    // matter how frequent the word is.
    bpe::Results r;
    bpe::task2({cs("abb", 100)}, r);
    EXPECT_EQ(tokens_of(r), std::vector<StrCount>({{"b", 200}, {"a", 100}}));
}

TEST(Task2, HuggingFaceExampleByteExact) {
    // The worked example, corpus ("hug",10), ("pug",5), ("pun",12), ("bun",4),
    // ("hugs",5). Under Project 1's one-pair-at-a-time rule the answer is
    // p 17, un 16, hug 15, s 5, ug 5, b 4. Under Project 2's rule it is not:
    // all four eligible pairs fit in the first round, so (u,g), (p,u), (u,n)
    // and (h,u) are merged before any of the others is re-counted, and (p,u)
    // and (u,n) meet in "pun" instead of one consuming the other. That is the
    // difference the round rule makes, in the specification's own example.
    bpe::Results r;
    bpe::task2({cs("hug", 10), cs("pug", 5), cs("pun", 12), cs("bun", 4),
                cs("hugs", 5)},
               r);
    EXPECT_EQ(tokens_of(r), std::vector<StrCount>({{"hug", 15},
                                                   {"n", 12},
                                                   {"pu", 12},
                                                   {"p", 5},
                                                   {"s", 5},
                                                   {"ug", 5},
                                                   {"b", 4},
                                                   {"un", 4}}));
}

TEST(Task2, TieBrokenByLexicographicallySmallerMergedString) {
    // Corpus {ab, abc, bc}: pairs (a,b) and (b,c) both occur in two distinct
    // words with total count 2. Both fit in the first round, and the order is
    // count, then merged text, then key: strcmp("ab","bc") < 0, so (a,b) gets
    // the lower token id and merges first. Its merge takes the (b,c) occurrence
    // out of "abc", so (b,c) is left with "bc" alone -- but the round's list is
    // a snapshot and (b,c) is merged anyway, which is why "bc" appears with 1.
    bpe::Results r;
    bpe::task2({cs("ab", 1), cs("abc", 1), cs("bc", 1)}, r);
    EXPECT_EQ(tokens_of(r),
              std::vector<StrCount>({{"ab", 2}, {"bc", 1}, {"c", 1}}));
}

TEST(Task2, OverlappingOccurrencesMergeLeftToRight) {
    // R5.4: a word (a,a,a) under the selected pair (a,a) becomes (aa,a), not
    // (a,aa). Corpus {aaa, aaab} -> {aaa:2, b:1}. A right-to-left merge would
    // instead give {a:2, aa:2, b:1}.
    bpe::Results r;
    bpe::task2({cs("aaa", 1), cs("aaab", 1)}, r);
    EXPECT_EQ(tokens_of(r), std::vector<StrCount>({{"aaa", 2}, {"b", 1}}));
}

TEST(Task2, WeightedCountsIncludeWordFrequency) {
    // (a,b) occurs once in "ab" and once in "abx"; weighted by word frequency
    // its count is 3 + 2 = 5, so it is eligible and merges.
    bpe::Results r;
    bpe::task2({cs("ab", 3), cs("abx", 2)}, r);
    EXPECT_EQ(tokens_of(r), std::vector<StrCount>({{"ab", 5}, {"x", 2}}));
}

TEST(Task2, LeftToRightMergeInsideOneWordStopsAfterExhaustion) {
    // (a,a) merges left to right in "aaaa" -> (aa,aa). The resulting pair
    // (aa,aa) occurs in a single word, so training stops: the token "aaaa" is
    // never created and the final token "aa" counts 2+1 = 3 occurrences.
    bpe::Results r;
    bpe::task2({cs("aaaa", 1), cs("aa", 1)}, r);
    EXPECT_EQ(tokens_of(r), std::vector<StrCount>({{"aa", 3}}));
}

TEST(Task2, PrefixTokensTieSortedLexicographically) {
    // After merging (a,b) (count 2, the only eligible pair), the tokens are
    // a(2), ab(2), c(1). "a" and "ab" tie on count; strcmp("a","ab") < 0
    // because "a" is a prefix of "ab", so "a" is listed first. This also
    // exercises the prefix-length branch of the byte comparator.
    bpe::Results r;
    bpe::task2({cs("a", 2), cs("ab", 1), cs("abc", 1)}, r);
    EXPECT_EQ(tokens_of(r),
              std::vector<StrCount>({{"a", 2}, {"ab", 2}, {"c", 1}}));
}

TEST(Task2, MultipleMergeRoundsRecountFromScratch) {
    // Corpus {abc, abcd}: (a,b) merges first (count 2, strcmp tie with (b,c)
    // broken in favour of "ab"), then the newly created token participates in
    // (ab,c), which also counts 2 across the two words and merges. This
    // exercises the recount-from-scratch rule across iterations.
    bpe::Results r;
    bpe::task2({cs("abc", 1), cs("abcd", 1)}, r);
    EXPECT_EQ(tokens_of(r), std::vector<StrCount>({{"abc", 2}, {"d", 1}}));
}

TEST(Task2, RewrittenWordWithRepeatedPairIncrementsFlatCount) {
    // Merging (a,b) in "ababab" rewrites the word to (ab,ab,ab): the rewritten
    // word contains the same adjacent pair twice, exercising the "already
    // present" branch of the flat pair-count builder (a repeated pair is
    // incremented, not re-inserted).
    bpe::Results r;
    bpe::task2({cs("ababab", 1), cs("abx", 1)}, r);
    EXPECT_EQ(tokens_of(r), std::vector<StrCount>({{"ab", 4}, {"x", 1}}));
}

TEST(Task2, StalePairIndexEntryIsSkipped) {
    // A word that gains pair p and then loses it through a *different* merge
    // must not be merged again when p is selected later. Corpus {bx:3, abx:1,
    // ab:1, zab:1}: (b,x) counts 4 (the most) and merges first, turning "abx"
    // into (a,bx) and so removing (a,b) from that word. (a,b) remains eligible
    // in "ab" and "zab" (count 2) and is selected next; the stale index entry
    // for "abx" must be skipped, not re-merged.
    bpe::Results r;
    bpe::task2({cs("bx", 3), cs("abx", 1), cs("ab", 1), cs("zab", 1)}, r);
    EXPECT_EQ(tokens_of(r), std::vector<StrCount>(
                                {{"bx", 4}, {"ab", 2}, {"a", 1}, {"z", 1}}));
}

TEST(Task2, PairSurvivingInWordKeepsItsCount) {
    // Merging (x,b) in "axaxb" destroys the second (a,x) but leaves the first,
    // so (a,x) keeps its remaining occurrence count in that word (the "already
    // present, still positive" branch of the per-word patch application).
    bpe::Results r;
    bpe::task2({cs("axaxb", 1), cs("xb", 1)}, r);
    EXPECT_EQ(tokens_of(r),
              std::vector<StrCount>({{"a", 2}, {"xb", 2}, {"x", 1}}));
}

TEST(Task2, StalePositionListIsCompressedInPassing) {
    // A pair that loses most of its occurrences to *other* merges accumulates
    // stale position entries. Corpus: 68 words "Pabx" (distinct prefixes P),
    // 2 words "Qbx" and one "ab". (b,x) out-counts (a,b) at the census (70 vs
    // 69), so it merges first within the round and destroys the (a,b)
    // occurrence in every "Pabx" word. When (a,b) is merged next, 68 of its 69
    // positions are stale and are skipped as the list is walked -- the
    // compression this test is named for -- leaving one live position, the
    // "ab" word. The surviving pairs then merge to "abx" and "bx".
    std::vector<bpe::CharSplit> splits;
    for (int i = 0; i < 68; ++i) {
        std::string w;
        w.push_back(static_cast<char>(0x80 + i));
        w += "abx";
        splits.push_back(cs(w, 1));
    }
    for (int i = 0; i < 2; ++i) {
        std::string w;
        w.push_back(static_cast<char>(0xE0 + i));
        w += "bx";
        splits.push_back(cs(w, 1));
    }
    splits.push_back(cs("ab", 1));
    bpe::Results r;
    bpe::task2(splits, r);
    std::vector<StrCount> expected;
    expected.push_back({"abx", 68});
    expected.push_back({"bx", 2});
    expected.push_back({"ab", 1});
    for (int i = 0; i < 68; ++i) {
        expected.push_back({std::string(1, static_cast<char>(0x80 + i)), 1});
    }
    for (int i = 0; i < 2; ++i) {
        expected.push_back({std::string(1, static_cast<char>(0xE0 + i)), 1});
    }
    EXPECT_EQ(tokens_of(r), expected);
}
