#ifndef BPE_DIST_H_
#define BPE_DIST_H_
#include "bpe.h"
#include <cstddef>
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

// Cut a corpus into weights.size() consecutive shares, roughly proportional to
// the weights. Returns weights.size()+1 offsets (first 0, last n); a cut never
// lands inside a word.
std::vector<std::size_t> split_points(const Byte* buf, std::size_t n,
                                      const std::vector<int>& weights);

// Rank 0 sends every rank its share of `input` over MPI (shares are sized by
// thread count) and keeps its own. The other ranks pass an empty vector and
// get their share back. Does nothing on a single rank.
void distribute_input(std::vector<Byte>& input);

}

#endif
