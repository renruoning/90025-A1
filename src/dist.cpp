#include "dist.h"
#include "bpe.h"
#include "absl/log/log.h"
#include <algorithm>
#include <chrono>
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

bool is_sep(Byte b) {
    return b == 0x20 || b == 0x09 || b == 0x0A || b == 0x0D || b == 0x00;
}
const std::size_t kChunk = std::size_t(1) << 28;  // MPI counts are ints
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

std::vector<std::size_t> split_points(const Byte* buf, std::size_t n,const std::vector<int>& weights) {
    const std::size_t parts = weights.size();
    std::vector<std::size_t> cuts(parts+1,0);
    cuts[parts]=n;
    std::size_t total =0, cum =0;
    for (int w :weights) total +=static_cast<std::size_t>(w);
    for (std::size_t r=1; r < parts;++r) {
        cum += static_cast<std::size_t>(weights[r -1]);
        std::size_t b = std::max(n/total * cum +n%total*cum/total,cuts[r - 1]);
        // slide forward until the cut sits next to whitespace
        while (b < n && b > 0 && !is_sep(buf[b]) && !is_sep(buf[b - 1])) ++b;
        cuts[r] = b;
    }
    return cuts;
}

void distribute_input(std::vector<Byte>& input) {
#ifdef BPE_HAVE_MPI
    const int size = mpi_size();
    if (size == 1) return;
    const auto t0 = std::chrono::steady_clock::now();
    if (mpi_rank() == 0) {
        const std::vector<std::size_t> cuts =split_points(input.data(), input.size(), g_threads);
        std::vector<MPI_Request> reqs;
        for (int r = 1; r < size; ++r) {
            unsigned long long len = cuts[r + 1] - cuts[r];
            MPI_Send(&len, 1,MPI_UNSIGNED_LONG_LONG,r,0,MPI_COMM_WORLD);
        }
        for (int r = 1; r < size;++r) {
            for (std::size_t off=cuts[r]; off<cuts[r+1]; off+=kChunk) {
                const int len = static_cast<int>(std::min(kChunk, cuts[r + 1] - off));
                reqs.emplace_back();
                MPI_Isend(input.data() + off,len,MPI_BYTE,r,1,MPI_COMM_WORLD, &reqs.back());
            }
        }
        MPI_Waitall(static_cast<int>(reqs.size()),reqs.data(),MPI_STATUSES_IGNORE);
        input.resize(cuts[1]);
        input.shrink_to_fit();
    } else {
        unsigned long long len = 0;
        MPI_Recv(&len, 1, MPI_UNSIGNED_LONG_LONG, 0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        input.resize(len);
        for (std::size_t off = 0; off < len; off+=kChunk) {
            const int n = static_cast<int>(std::min<std::size_t>(kChunk, len - off));
            MPI_Recv(input.data()+off,n,MPI_BYTE, 0, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        }
    }
    const auto t1 = std::chrono::steady_clock::now();
    LOG(INFO) << "rank " << mpi_rank() << " holds " << input.size()
              << " bytes (distribute: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
              << " ms)";
#else
    (void)input;
#endif
}

}
