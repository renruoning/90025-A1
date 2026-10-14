#include "bpe.h"
#include "md5.h"
#include "absl/log/log.h"
#include <chrono>
#include <cstdint>
#include <omp.h>
#include <stdexcept>
#include <sys/resource.h>
namespace bpe {
namespace {

std::int64_t elapsed_ms(const std::chrono::steady_clock::time_point& start,
                        const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
        .count();
}
}
Results run_pipeline(std::vector<Byte>& input) {
    const std::chrono::steady_clock::time_point t0 =
        std::chrono::steady_clock::now();
    Results results;
    parallel_task1(input, results);   // split + word counts + char splits
    std::vector<Byte>().swap(input);  // task 1 copied every word; input is dead
    const std::chrono::steady_clock::time_point t1 =
        std::chrono::steady_clock::now();
    parallel_task2(results.char_splits, results);  // BPE merge loop
    const std::chrono::steady_clock::time_point t2 =
        std::chrono::steady_clock::now();
    LOG(INFO) << "task1 (split + word count + char split): "
              << elapsed_ms(t0, t1) << " ms";
    LOG(INFO) << "task2 (BPE merge): " << elapsed_ms(t1, t2) << " ms";
    LOG(INFO) << "task1 + task2: " << elapsed_ms(t0, t2) << " ms";
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
        LOG(INFO) << "peak memory: " << (ru.ru_maxrss / 1024.0) << " MB";
    }
    return results;
}
int run_cli(int argc, char** argv) {
    if (argc != 2) {
        LOG(ERROR) << "usage: " << (argc > 0 ? argv[0] : "bpe")
                   << " <input-file>";
        return 2;
    }
    try {
        LOG(INFO) << "OpenMP threads: " << omp_get_max_threads()
                  << "; MPI rank " << mpi_rank() << " of " << mpi_size();
        // Only rank 0 reads the corpus (specification, Data Specification).
        // TODO(Project 2): send each rank its share of `input` over MPI. Rank 0
        // holds the whole corpus and the other ranks start empty; a word must
        // never be split across two ranks, so choose the boundaries of each
        // share before sending.
        const std::chrono::steady_clock::time_point t_load0 =
            std::chrono::steady_clock::now();
        std::vector<Byte> input;
        if (mpi_rank() == 0) {
            input = read_file(argv[1]);
        }
        const std::chrono::steady_clock::time_point t_load1 =
            std::chrono::steady_clock::now();
        LOG(INFO) << "read " << input.size() << " bytes from " << argv[1];
        LOG(INFO) << "load file: " << elapsed_ms(t_load0, t_load1) << " ms";
        // TODO(Project 2): the merge decision is global, so every rank must
        // reach the same answer as a single process would. See parallel_task1
        // and parallel_task2 in src/parallel.cpp.
        const Results results = run_pipeline(input);
        LOG(INFO) << "task 1: " << results.word_counts.size()
                  << " distinct words; task 2: " << results.tokens.size()
                  << " tokens";
        const std::chrono::steady_clock::time_point t_save0 =
            std::chrono::steady_clock::now();
        // Only rank 0 writes the output (specification, Data Specification).
        if (mpi_rank() == 0) {
            write_output(results, "output.txt");
        }
        const std::chrono::steady_clock::time_point t_save1 =
            std::chrono::steady_clock::now();
        LOG(INFO) << "save output: " << elapsed_ms(t_save0, t_save1) << " ms";
        if (mpi_rank() == 0) {
            LOG(INFO) << "wrote output.txt (md5 "
                      << md5_hex(read_file("output.txt")) << ")";
        }
        return 0;
    } catch (const std::exception& e) {
        LOG(ERROR) << e.what();
        return 1;
    }
}

}
