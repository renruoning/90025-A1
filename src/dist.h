#ifndef BPE_DIST_H_
#define BPE_DIST_H_
#include <vector>
namespace bpe {

// Read BPE_THREADS (e.g. "2,4,10"), set this rank's OpenMP thread count from
// its rank index, then share every rank's count with all the others.
// Call once after mpi_init. Without BPE_THREADS the OpenMP default is kept.
void setup_threads();

// A "worker" is one OpenMP thread of one rank. Workers are numbered rank by
// rank: rank 0's threads first, then rank 1's, ...
int num_workers();
int threads_of(int rank);
int first_worker(int rank);
int worker_rank(int worker);

}

#endif
