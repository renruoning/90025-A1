#include "bpe.h"
#include "absl/log/log.h"
#ifdef BPE_HAVE_MPI
#include <mpi.h>
#endif
namespace bpe {
namespace {

int g_rank = 0;
int g_size = 1;
bool g_initialised = false;
}
void mpi_init(int* argc, char*** argv) {
#ifdef BPE_HAVE_MPI
    if (!g_initialised) {
        MPI_Init(argc, argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &g_size);
        g_initialised = true;
        LOG(INFO) << "MPI rank " << g_rank << " of " << g_size;
    }
#else
    (void)argc;
    (void)argv;
#endif
}
void mpi_finalize() {
    if (!g_initialised) {
        return;
    }
#ifdef BPE_HAVE_MPI
    MPI_Finalize();
#endif
    g_initialised = false;
}
int mpi_rank() { return g_rank; }

int mpi_size() { return g_size; }

}
