// Tests for the pipeline, the CLI, and output writing.

#include "bpe.h"

#include "gtest/gtest.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;

namespace {

class ScopedTempDir {
   public:
    ScopedTempDir() {
        static int counter = 0;
        path_ = fs::temp_directory_path() /
                ("bpe_test_" + std::to_string(::getpid()) + "_" +
                 std::to_string(counter++));
        fs::create_directories(path_);
    }
    ~ScopedTempDir() {
        std::error_code ec;
        fs::remove_all(path_, ec);
    }
    const fs::path& path() const { return path_; }

   private:
    fs::path path_;
};

}  // namespace

TEST(RunPipeline, ProducesTask1AndTask2Results) {
    const std::string input = "hug hug\npug";
    std::vector<bpe::Byte> buffer(input.begin(), input.end());
    const bpe::Results r = bpe::run_pipeline(buffer);

    // Task 1.1 (word_counts order is unspecified): hug x2, pug x1.
    ASSERT_EQ(r.word_counts.size(), 2u);
    auto find_count = [&](const std::string& word) -> std::size_t {
        for (const auto& wc : r.word_counts) {
            if (std::string(wc.word.begin(), wc.word.end()) == word) {
                return wc.count;
            }
        }
        return 0;
    };
    EXPECT_EQ(find_count("hug"), 2u);
    EXPECT_EQ(find_count("pug"), 1u);

    // Task 1.2 feeds Task 2; tokens must be non-empty and internally
    // consistent (each token's bytes occur in the corpus).
    ASSERT_FALSE(r.tokens.empty());
    for (const bpe::TokenCount& t : r.tokens) {
        EXPECT_GE(t.count, 1u);
    }
}

TEST(RunCli, UsageErrorWithoutInputFile) {
    char arg0[] = "bpe";
    char* argv[] = {arg0, nullptr};
    EXPECT_EQ(bpe::run_cli(1, argv), 2);
}

TEST(RunCli, MissingInputFileReturnsError) {
    char arg0[] = "bpe";
    char arg1[] = "/tmp/bpe_no_such_input_file_xyz.txt";
    char* argv[] = {arg0, arg1};
    EXPECT_EQ(bpe::run_cli(2, argv), 1);
}

TEST(RunCli, WritesByteExactOutputTxt) {
    ScopedTempDir dir;
    const fs::path input = dir.path() / "input.txt";
    {
        std::ofstream out(input);
        out << "hug hug\npug\n";
    }
    const fs::path old = fs::current_path();
    fs::current_path(dir.path());
    char arg0[] = "bpe";
    std::string arg1 = input.string();
    char* argv[] = {arg0, arg1.data()};
    const int rc = bpe::run_cli(2, argv);
    fs::current_path(old);

    EXPECT_EQ(rc, 0);
    std::ifstream out(dir.path() / "output.txt", std::ios::binary);
    std::ostringstream buffer;
    buffer << out.rdbuf();
    // Corpus {hug:2, pug:1}: (u,g) merges first (count 3, both words), so the
    // tokens are ug(3), h(2), p(1) in that order.
    EXPECT_EQ(buffer.str(), std::string("ug 3\nh 2\np 1\n"));
}

TEST(WriteOutput, WritesTokenLines) {
    ScopedTempDir dir;
    bpe::Results r;
    r.tokens.push_back(
        {std::vector<bpe::Byte>{bpe::Byte('u'), bpe::Byte('g')}, 3});
    r.tokens.push_back({std::vector<bpe::Byte>{bpe::Byte('p')}, 1});
    const fs::path out = dir.path() / "output.txt";
    bpe::write_output(r, out.string());

    std::ifstream f(out, std::ios::binary);
    std::ostringstream buffer;
    buffer << f.rdbuf();
    EXPECT_EQ(buffer.str(), std::string("ug 3\np 1\n"));
}

TEST(WriteOutput, ThrowsOnUnwritablePath) {
    bpe::Results r;
    EXPECT_THROW(bpe::write_output(r, "/nonexistent_dir_xyz_1234/out.txt"),
                 std::runtime_error);
}

TEST(WriteOutput, ThrowsOnWriteError) {
    // /dev/full is a Linux device that succeeds on open but fails every write
    // (ENOSPC). A token larger than the stream buffer forces the buffered
    // write to reach the device, which sets the fail bit; the mid-write check
    // in write_output then throws.
    bpe::Results r;
    std::vector<bpe::Byte> big(65536, bpe::Byte('a'));
    r.tokens.push_back({big, 1});
    EXPECT_THROW(bpe::write_output(r, "/dev/full"), std::runtime_error);
}

TEST(PrintTask1, PrintsWordCountsThenCharSplits) {
    bpe::Results r;
    const std::vector<bpe::Byte> hug = {bpe::Byte('h'), bpe::Byte('u'),
                                        bpe::Byte('g')};
    r.word_counts.push_back({hug, 2});
    r.char_splits.push_back({hug, 2});

    testing::internal::CaptureStdout();
    bpe::print_task1(r);
    const std::string out = testing::internal::GetCapturedStdout();
    EXPECT_EQ(out, std::string("hug 2\nh u g 2\n"));
}
