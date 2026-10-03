// cuSOLVERMp validation: gathers the distributed factors outside the timed interval.
// No refactorization and no gathering in the timed QR interval.
inline int mp_local_length(int n, int block, int peers, int coordinate) {
    int result = 0;
    for (int start = coordinate * block; start < n; start += peers * block)
        result += std::min(block, n - start);
    return result;
}
#include "reference_mp_streamed.inc"
template <class T>
json validate_mp_gathered(int m, int n, int block, int pr, int pc, int rank, int nr, int ld,
                          const T *a, const T *tau, ncclComm_t comm, Stream &st) {
    const double started = seconds();
    const int width = 256, p = pr * pc, rr = rank % pr, rc = rank / pr, k = std::min(m, n);
    Buffer<T> gathered, staging;
    int admission = 1;
    std::string failure;
    const bool force_stream = std::getenv("TQR_MP_VALIDATION_STREAM") &&
                              std::string(std::getenv("TQR_MP_VALIDATION_STREAM")) == "1";
    if (rank == 0) {
        size_t free = 0, total = 0;
        CU(cudaMemGetInfo(&free, &total));
        const size_t words = checked_mul(size_t(m), size_t(n));
        const size_t extra =
            checked_mul(size_t(m), size_t(width + 2 * reference_validation_width(n)));
        if (force_stream) {
            admission = 0;
            failure = "explicit streamed-validation qualification";
        } else if (checked_mul(checked_add(words, extra), sizeof(T)) > free * 9 / 10) {
            admission = 0;
            failure = "independent validation gather exceeds free HBM";
        } else
            try {
                gathered.alloc(words);
                staging.alloc(size_t(m) * width);
            } catch (const std::exception &e) {
                admission = 0;
                failure = e.what();
            }
    }
    MPI_Bcast(&admission, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!admission) {
        gathered = Buffer<T>{};
        staging = Buffer<T>{};
        return validate_mp_streamed<T>(m, n, block, pr, pc, rank, nr, ld, a, tau, comm, st,
                                       failure);
    }
    json result;
    if (admission) {
        std::vector<int> rows(pr), offset(pr);
        int total_rows = 0;
        for (int r = 0; r < pr; ++r) {
            rows[r] = mp_local_length(m, block, pr, r);
            offset[r] = total_rows;
            total_rows += rows[r];
        }
        if (total_rows != m || rows[rr] != nr)
            throw std::runtime_error("independent_gather_row_map");
        const auto dtype = sizeof(T) == 8 ? ncclDouble : ncclFloat;
        for (int col = 0; col < n;) {
            // Do not cross a process-column block boundary in one transfer.
            const int q = std::min({width, block - col % block, n - col});
            const int owner_column = col / block % pc,
                      local_col = (col / (block * pc)) * block + col % block;
            nc_check_ref(ncclGroupStart());
            if (rank == 0) {
                for (int r = 0; r < pr; ++r) {
                    const int source = r + owner_column * pr;
                    if (source != 0 && rows[r])
                        nc_check_ref(ncclRecv(staging.p + size_t(offset[r]) * q,
                                              size_t(rows[r]) * q, dtype, source, comm, st));
                }
            } else if (rc == owner_column && nr)
                nc_check_ref(
                    ncclSend(a + size_t(local_col) * ld, size_t(nr) * q, dtype, 0, comm, st));
            nc_check_ref(ncclGroupEnd());
            if (rank == 0)
                for (int r = 0; r < pr; ++r) {
                    const int source_rank = r + owner_column * pr;
                    const T *source = source_rank ? staging.p + size_t(offset[r]) * q
                                                  : a + size_t(local_col) * ld;
                    const int source_ld = source_rank ? rows[r] : ld;
                    int local_row = 0;
                    for (int global = r * block; global < m; global += pr * block) {
                        const int count = std::min(block, m - global);
                        CU(cudaMemcpy2DAsync(
                            gathered.p + global + size_t(col) * m, size_t(m) * sizeof(T),
                            source + local_row, size_t(source_ld) * sizeof(T),
                            size_t(count) * sizeof(T), q, cudaMemcpyDeviceToDevice, st));
                        local_row += count;
                    }
                }
            st.sync();
            col += q;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        const auto mpi_type = sizeof(T) == 8 ? MPI_DOUBLE : MPI_FLOAT;
        const int local_count = mp_local_length(k, block, pc, rc);
        std::vector<T> local_tau(local_count), all_tau;
        std::vector<int> tau_counts(p), tau_offsets(p);
        int tau_total = 0;
        for (int r = 0; r < p; ++r) {
            tau_counts[r] = mp_local_length(k, block, pc, r / pr);
            tau_offsets[r] = tau_total;
            tau_total += tau_counts[r];
        }
        if (local_count)
            CU(cudaMemcpy(local_tau.data(), tau, size_t(local_count) * sizeof(T),
                          cudaMemcpyDeviceToHost));
        if (rank == 0)
            all_tau.resize(tau_total);
        MPI_Gatherv(local_tau.data(), local_count, mpi_type, rank == 0 ? all_tau.data() : nullptr,
                    tau_counts.data(), tau_offsets.data(), mpi_type, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            cusolverDnHandle_t local = nullptr;
            try {
                reference_solver_check(cusolverDnCreate(&local), "independent validation handle");
                reference_solver_check(cusolverDnSetStream(local, st),
                                       "independent validation stream");
                reference_solver_check(cusolverDnSetMathMode(local, CUSOLVER_DEFAULT_MATH),
                                       "independent validation math");
                ReferenceOrmqr<T> apply(local, st);
                Buffer<T> authoritative_tau(k);
                std::vector<T> selected(k);
                uint64_t mismatches = 0;
                int first_mismatch = -1;
                double max_difference = 0;
                json tail = json::array();
                for (int j = 0; j < k; ++j) {
                    const int col_owner = j / block % pc, row_owner = j / block % pr;
                    const int index = (j / (block * pc)) * block + j % block,
                              owner = row_owner + col_owner * pr;
                    const T value = all_tau[tau_offsets[owner] + index];
                    selected[j] = value;
                    json replicas = json::array();
                    for (int r = 0; r < pr; ++r) {
                        const T other = all_tau[tau_offsets[r + col_owner * pr] + index];
                        if (j >= k - 8)
                            replicas.push_back(other);
                        if (other != value) {
                            ++mismatches;
                            if (first_mismatch < 0)
                                first_mismatch = j;
                            max_difference =
                                std::max(max_difference, double(std::abs(other - value)));
                        }
                    }
                    if (j >= k - 8)
                        tail.push_back({{"global_column", j},
                                        {"owner_rank", owner},
                                        {"process_row_replicas", replicas}});
                }
                if (k)
                    CU(cudaMemcpy(authoritative_tau.p, selected.data(), size_t(k) * sizeof(T),
                                  cudaMemcpyHostToDevice));
                result = validate_conventional_reference<T>(
                    m, n, block,
                    [&](T *x, int ldx, int col, int q) {
                        reference_r(gathered.p, m, x, ldx, m, q, block, 1, 1, 0, 0, col, sizeof(T));
                    },
                    [&](T *x, int ldx, int q, cublasOperation_t op) {
                        apply.apply(m, q, k, gathered.p, m, authoritative_tau.p, x, ldx, op);
                    });
                result["Q_application"] =
                    "actual cuSOLVERMp A/tau gathered in global row/column order; independent cuSOLVERDn Ormqr on GPU0";
                result["tau_selection"] =
                    "actual local tau for global column j from the process owning diagonal A(j,j)";
                result["tau_replica_diagnostic"] = {{"unequal_values", mismatches},
                                                    {"first_column", first_mismatch},
                                                    {"max_absolute_difference", max_difference},
                                                    {"last_eight_columns", tail}};
                result["gathered_factor_bytes"] = gathered.n * sizeof(T);
                result["gather_staging_bytes"] = staging.n * sizeof(T);
                result["ormqr_workspace_bytes"] = apply.workspace_bytes();
                result["including_gather_s"] = seconds() - started;
                reference_solver_check(cusolverDnDestroy(local), "independent validation destroy");
                local = nullptr;
            } catch (const std::exception &e) {
                if (local)
                    cusolverDnDestroy(local);
                result = {{"performed", true}, {"pass", false}, {"error", e.what()}};
            }
        }
    } else if (rank == 0)
        result = {{"performed", false}, {"pass", false}, {"error", failure}};
    std::string encoded = rank == 0 ? result.dump() : std::string();
    int length = int(encoded.size());
    MPI_Bcast(&length, 1, MPI_INT, 0, MPI_COMM_WORLD);
    encoded.resize(length);
    MPI_Bcast(encoded.data(), length, MPI_CHAR, 0, MPI_COMM_WORLD);
    return json::parse(encoded);
}
