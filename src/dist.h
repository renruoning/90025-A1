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


// Which worker owns a word (hash of its bytes, same on every rank).
int worker_of(const Byte* s, std::size_t n);

// Step 3: count the words of this rank's share, send each word to the worker
// that owns it, and merge there. Afterwards `results` holds only the words this
// rank owns (sorted per worker), not the whole corpus.
void distributed_task1(std::vector<Byte>& input, Results& results);


// Where each of this rank's workers starts in results.char_splits after
// distributed_task1 (threads+1 offsets).
const std::vector<std::size_t>& task1_bounds();

// Steps 4-5: the BPE rounds over all workers. Needs distributed_task1 to have
// run first. Only rank 0 gets results.tokens.
void distributed_task2(const std::vector<CharSplit>& splits, Results& results);

}

#endif
