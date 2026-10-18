#include "dist.h"
#include "bpe.h"
#include "absl/log/log.h"
#include <algorithm>
#include <atomic>
#include <thread>
#include <chrono>
#include <cstdlib>
#include <omp.h>
#include <fcntl.h>
#include <stdexcept>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>
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
// without setup_threads (e.g. unit tests) behave like one rank, default threads
int num_workers() { return g_first.empty() ? omp_get_max_threads() : g_first.back(); }
int threads_of(int rank) {
    return g_threads.empty() ? omp_get_max_threads() : g_threads[rank];
}
int first_worker(int rank) { return g_first.empty() ? 0 : g_first[rank]; }
int worker_rank(int worker) {
    if (g_first.empty()) return 0;
    int r = 0;
    while (g_first[r + 1] <= worker) ++r;
    return r;
}

namespace {
// the cut logic, for any way of looking at byte i (buffer or file)
template <class At>
std::vector<std::size_t> cuts_by(At at, std::size_t n, const std::vector<int>& weights) {
    const std::size_t parts = weights.size();
    std::vector<std::size_t> cuts(parts+1,0);
    cuts[parts]=n;
    std::size_t total =0, cum =0;
    for (int w :weights) total +=static_cast<std::size_t>(w);
    for (std::size_t r=1; r < parts;++r) {
        cum += static_cast<std::size_t>(weights[r -1]);
        std::size_t b = std::max(n/total * cum +n%total*cum/total,cuts[r - 1]);
        // slide forward until the cut sits next to whitespace
        while (b < n && b > 0 && !is_sep(at(b)) && !is_sep(at(b - 1))) ++b;
        cuts[r] = b;
    }
    return cuts;
}
}

std::vector<std::size_t> split_points(const Byte* buf, std::size_t n,const std::vector<int>& weights) {
    return cuts_by([&](std::size_t i) { return buf[i]; }, n, weights);
}

// like read_file, but several threads pread different pieces of the file
std::vector<Byte> read_file_parallel(const std::string& path) {
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open file: " + path);
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); throw std::runtime_error("cannot stat file: " + path); }
    const std::size_t n = static_cast<std::size_t>(st.st_size);
    std::vector<Byte> data(n);
    const std::size_t piece = std::size_t(8) << 20;
    const long pieces = static_cast<long>((n + piece - 1) / piece);
    bool failed = false;
    #pragma omp parallel for schedule(dynamic, 1)
    for (long i = 0; i < pieces; ++i) {
        std::size_t off = std::size_t(i) * piece;
        const std::size_t end = std::min(n, off + piece);
        while (off < end) {
            const ssize_t got = pread(fd, data.data() + off, end - off, static_cast<off_t>(off));
            if (got <= 0) {
                #pragma omp atomic write
                failed = true;
                break;
            }
            off += static_cast<std::size_t>(got);
        }
    }
    close(fd);
    if (failed) throw std::runtime_error("short read on " + path);
    return data;
}

#ifdef BPE_HAVE_MPI
namespace {
const std::size_t kPiece = std::size_t(16) << 20;   // one MPI message while loading
const int kSlots = 12;                              // staging buffers on rank 0
struct Piece { int rank; std::size_t off, len; };
void spin() { std::this_thread::yield(); }
}
#endif

// Rank 0 reads the file and sends every other rank its share while it reads:
// reader threads fill staging buffers, thread 0 sends them in order.
std::vector<Byte> load_input(const std::string& path) {
#ifdef BPE_HAVE_MPI
    const int ranks = mpi_size();
    if (ranks > 1) {
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<Byte> input;
        if (mpi_rank() == 0) {
            const int fd = open(path.c_str(), O_RDONLY);
            if (fd < 0) throw std::runtime_error("cannot open file: " + path);
            struct stat st;
            if (fstat(fd, &st) != 0) { close(fd); throw std::runtime_error("cannot stat file: " + path); }
            const std::size_t n = static_cast<std::size_t>(st.st_size);
            // cuts are found by peeking at the bytes around each one
            const std::vector<std::size_t> cuts = cuts_by([&](std::size_t i) {
                Byte b = 0;
                if (pread(fd, &b, 1, static_cast<off_t>(i)) != 1) b = 0;
                return b;
            }, n, g_threads);
            for (int r = 1; r < ranks; ++r) {
                unsigned long long len = cuts[r + 1] - cuts[r];
                MPI_Send(&len, 1, MPI_UNSIGNED_LONG_LONG, r, 0, MPI_COMM_WORLD);
            }
            std::vector<Piece> pieces;
            for (int r = 1; r < ranks; ++r)
                for (std::size_t off = cuts[r]; off < cuts[r + 1]; off += kPiece)
                    pieces.push_back(Piece{r, off, std::min(kPiece, cuts[r + 1] - off)});
            const long np = static_cast<long>(pieces.size());

            std::vector<std::vector<Byte>> stage(kSlots, std::vector<Byte>(kPiece));
            std::vector<std::atomic<long>> filled(kSlots), freed(kSlots);
            for (int i = 0; i < kSlots; ++i) { filled[i] = -1; freed[i] = -1; }
            std::atomic<long> next_piece{0}, next_own{0};
            std::atomic<bool> sized{false}, failed{false};
            const std::size_t own = cuts[1];
            const std::size_t own_piece = std::size_t(8) << 20;
            const long own_n = static_cast<long>((own + own_piece - 1) / own_piece);

            auto read_at = [&](Byte* dst, std::size_t off, std::size_t len) {
                std::size_t done = 0;
                while (done < len) {
                    const ssize_t got = pread(fd, dst + done, len - done, static_cast<off_t>(off + done));
                    if (got <= 0) { failed = true; return; }
                    done += static_cast<std::size_t>(got);
                }
            };
            const int nt = std::max(2, threads_of(0));
#pragma omp parallel num_threads(nt)
            {
                const int tid = omp_get_thread_num();
                if (tid == 1) {
                    input.resize(own);
                    sized.store(true, std::memory_order_release);
                }
                if (tid != 0) {
                    for (;;) {
                        const long c = next_piece.fetch_add(1);
                        if (c >= np) break;
                        const int sl = int(c % kSlots);
                        const long g = c / kSlots;
                        while (freed[sl].load(std::memory_order_acquire) != g - 1) spin();
                        read_at(stage[sl].data(), pieces[c].off, pieces[c].len);
                        filled[sl].store(g, std::memory_order_release);
                    }
                } else {
                    for (long c = 0; c < np; ++c) {
                        const int sl = int(c % kSlots);
                        const long g = c / kSlots;
                        while (filled[sl].load(std::memory_order_acquire) != g) spin();
                        MPI_Send(stage[sl].data(), int(pieces[c].len), MPI_BYTE,
                                 pieces[c].rank, 1, MPI_COMM_WORLD);
                        freed[sl].store(g, std::memory_order_release);
                    }
                }
                // everyone then reads rank 0's own share
                while (!sized.load(std::memory_order_acquire)) spin();
                for (;;) {
                    const long i = next_own.fetch_add(1);
                    if (i >= own_n) break;
                    const std::size_t off = std::size_t(i) * own_piece;
                    read_at(input.data() + off, off, std::min(own_piece, own - off));
                }
            }
            close(fd);
            if (failed) throw std::runtime_error("short read on " + path);
        } else {
            unsigned long long len = 0;
            MPI_Recv(&len, 1, MPI_UNSIGNED_LONG_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            input.resize(len);
            for (std::size_t off = 0; off < len; off += kPiece) {
                const int n = static_cast<int>(std::min<std::size_t>(kPiece, len - off));
                MPI_Recv(input.data() + off, n, MPI_BYTE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        const auto t1 = std::chrono::steady_clock::now();
        LOG(INFO) << "rank " << mpi_rank() << " holds " << input.size()
                  << " bytes (load+distribute: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
                  << " ms)";
        return input;
    }
#endif
    return read_file_parallel(path);
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
