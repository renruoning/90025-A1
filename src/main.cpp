#include "bpe.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
int main(int argc, char** argv) {
    absl::InitializeLog();
    absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
    bpe::mpi_init(&argc, &argv);
    const int status = bpe::run_cli(argc, argv);
    bpe::mpi_finalize();
    return status;
}
