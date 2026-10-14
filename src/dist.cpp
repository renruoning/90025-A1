#include "dist.h"
#include "bpe.h"
#include "absl/log/log.h"
#include <cstdlib>
#include <omp.h>
#include <sstream>
#include <string>
#ifdef BPE_HAVE_MPI
#include <mpi.h>
#endif
namespace bpe {
namespace {

std::vector<int> g_threads;  // threads per rank
std::vector<int> g_first;    // first worker id per rank, plus total at the end

// "4,4,8" -> {4,4,8}; anything odd gives an empty vector
std::vector<int> parse_tuple(const char* s) {
    std::vector<int> out;
    if (s == nullptr) return out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        char* end = nullptr;
        const long v = std::strtol(tok.c_str(), &end, 10);
        if (end == tok.c_str() || *end != '\0' || v < 1) return {};
        out.push_back(static_cast<int>(v));
    }
    return out;
}
}
void setup_threads() {
    const int rank = mpi_rank();
    const int size = mpi_size();
    int mine = omp_get_max_threads();

    const char* env = std::getenv("BPE_THREADS");
    const std::vector<int> tuple = parse_tuple(env);
    if (env != nullptr && tuple.empty()) {
        LOG(WARNING) << "ignoring malformed BPE_THREADS=\"" << env << "\"";
    } else if (!tuple.empty()) {
        if (static_cast<int>(tuple.size()) != size) {
            LOG(WARNING) << "BPE_THREADS has " << tuple.size()
                         << " entries but there are " << size << " ranks";
        }
        if (rank < static_cast<int>(tuple.size())) mine = tuple[rank];
    }
    omp_set_num_threads(mine);

    g_threads.assign(size, mine);
#ifdef BPE_HAVE_MPI
    if (size > 1) {
        MPI_Allgather(&mine, 1, MPI_INT, g_threads.data(), 1, MPI_INT,
                      MPI_COMM_WORLD);
    }
#endif
    g_first.assign(size + 1, 0);
    for (int r = 0; r < size; ++r) g_first[r + 1] = g_first[r] + g_threads[r];

    if (rank == 0) {
        std::ostringstream os;
        for (int r = 0; r < size; ++r) os << (r ? "," : "") << g_threads[r];
        LOG(INFO) << "thread layout {" << os.str() << "}, "
                  << g_first[size] << " workers";
    }
}
int num_workers() { return g_first.empty() ? 1 : g_first.back(); }
int threads_of(int rank) { return g_threads[rank]; }
int first_worker(int rank) { return g_first[rank]; }
int worker_rank(int worker) {
    int r = 0;
    while (g_first[r + 1] <= worker) ++r;
    return r;
}

}
