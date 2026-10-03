// cuSOLVERMp validation: applies Q in streamed column blocks.
// Native factorization and timing do not use this path.
template <class T>
json validate_mp_streamed(int m, int n, int block, int pr, int pc, int rank, int nr, int ld,
                          const T *a, const T *tau, ncclComm_t comm, Stream &st,
                          const std::string &reason) {
    const double started = seconds();
    const int p = pr * pc, rr = rank % pr, rc = rank / pr, k = std::min(m, n), chunk = 256;
    std::vector<int> rows(pr), offset(pr);
    int total = 0;
    for (int r = 0; r < pr; ++r) {
        rows[r] = mp_local_length(m, block, pr, r);
        offset[r] = total;
        total += rows[r];
    }
    if (total != m || rows[rr] != nr)
        throw std::runtime_error("streamed_reference_row_map");
    Buffer<T> panel, staging, device_tau;
    int admission = 1;
    std::string failure;
    if (rank == 0)
        try {
            size_t free = 0, hbm = 0;
            CU(cudaMemGetInfo(&free, &hbm));
            size_t estimate =
                checked_mul(size_t(m), size_t(block + chunk + 2 * reference_validation_width(n)));
            if (checked_mul(checked_add(estimate, size_t(k)), sizeof(T)) > free * 9 / 10)
                throw std::runtime_error("independent_streamed_validation_exceeds_free_HBM");
            panel.alloc(checked_mul(size_t(m), size_t(block)));
            staging.alloc(checked_mul(size_t(m), size_t(chunk)));
            device_tau.alloc(k);
        } catch (const std::exception &e) {
            admission = 0;
            failure = e.what();
        }
    MPI_Bcast(&admission, 1, MPI_INT, 0, MPI_COMM_WORLD);
    json result;
    if (admission) {
        const auto mpi_type = sizeof(T) == 8 ? MPI_DOUBLE : MPI_FLOAT;
        const auto dtype = sizeof(T) == 8 ? ncclDouble : ncclFloat;
        const int count = mp_local_length(k, block, pc, rc);
        std::vector<T> local_tau(count), all_tau;
        std::vector<int> counts(p), offsets(p);
        int sum = 0;
        for (int r = 0; r < p; ++r) {
            counts[r] = mp_local_length(k, block, pc, r / pr);
            offsets[r] = sum;
            sum += counts[r];
        }
        if (count)
            CU(cudaMemcpy(local_tau.data(), tau, size_t(count) * sizeof(T),
                          cudaMemcpyDeviceToHost));
        if (rank == 0)
            all_tau.resize(sum);
        MPI_Gatherv(local_tau.data(), count, mpi_type, rank == 0 ? all_tau.data() : nullptr,
                    counts.data(), offsets.data(), mpi_type, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            std::vector<T> selected(k);
            for (int j = 0; j < k; ++j) {
                int owner = (j / block) % pr + ((j / block) % pc) * pr;
                int local = j / (block * pc) * block + j % block;
                selected[j] = all_tau[offsets[owner] + local];
            }
            device_tau.upload(selected, st);
            st.sync();
        }
        // Every rank services the same ordered gather requests. A request may cross
        // a process-column block boundary; each transfer below never crosses one.
        auto gather = [&](int first, int width, T *destination, int ldd) {
            for (int done = 0; done < width;) {
                int col = first + done, q = std::min({chunk, block - col % block, width - done});
                int owner_column = col / block % pc,
                    local_col = col / (block * pc) * block + col % block;
                nc_check_ref(ncclGroupStart());
                if (rank == 0) {
                    for (int r = 0; r < pr; ++r) {
                        int source = r + owner_column * pr;
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
                        int source_rank = r + owner_column * pr;
                        const T *source = source_rank ? staging.p + size_t(offset[r]) * q
                                                      : a + size_t(local_col) * ld;
                        int source_ld = source_rank ? rows[r] : ld, local_row = 0;
                        for (int global = r * block; global < m; global += pr * block) {
                            int length = std::min(block, m - global);
                            CU(cudaMemcpy2DAsync(
                                destination + global + size_t(done) * ldd, size_t(ldd) * sizeof(T),
                                source + local_row, size_t(source_ld) * sizeof(T),
                                size_t(length) * sizeof(T), q, cudaMemcpyDeviceToDevice, st));
                            local_row += length;
                        }
                    }
                st.sync();
                done += q;
            }
        };
        if (rank == 0) {
            cusolverDnHandle_t local = nullptr;
            uint64_t requests = 0;
            auto request = [&](int first, int width, T *destination, int ldd) {
                int command[2] = {first, width};
                MPI_Bcast(command, 2, MPI_INT, 0, MPI_COMM_WORLD);
                gather(first, width, destination, ldd);
                ++requests;
            };
            try {
                reference_solver_check(cusolverDnCreate(&local), "streamed validation handle");
                reference_solver_check(cusolverDnSetStream(local, st),
                                       "streamed validation stream");
                reference_solver_check(cusolverDnSetMathMode(local, CUSOLVER_DEFAULT_MATH),
                                       "streamed validation math");
                ReferenceOrmqr<T> apply(local, st);
                result = validate_conventional_reference<T>(
                    m, n, block,
                    [&](T *x, int ldx, int col, int q) {
                        request(col, q, x, ldx);
                        for (int j = 0; j < q; ++j) {
                            int first = std::min(m, col + j + 1);
                            if (first < m)
                                CU(cudaMemsetAsync(x + first + size_t(j) * ldx, 0,
                                                   size_t(m - first) * sizeof(T), st));
                        }
                    },
                    [&](T *x, int ldx, int q, cublasOperation_t op) {
                        auto apply_panel = [&](int first) {
                            int width = std::min(block, k - first);
                            request(first, width, panel.p, m);
                            apply.apply(m - first, q, width, panel.p + first, m,
                                        device_tau.p + first, x + first, ldx, op);
                        };
                        if (op == CUBLAS_OP_N)
                            for (int first = (k - 1) / block * block; first >= 0; first -= block)
                                apply_panel(first);
                        else
                            for (int first = 0; first < k; first += block)
                                apply_panel(first);
                        st.sync();
                    });
                result["Q_application"] =
                    "actual distributed cuSOLVERMp A/tau; bounded panel gathers and independent cuSOLVERDn Ormqr on GPU0";
                result["tau_selection"] = "actual tau from process owning diagonal A(j,j)";
                result["validation_storage"] = "streamed retained factors; no second full matrix";
                result["streaming_reason"] = reason;
                result["gather_requests"] = requests;
                result["factor_panel_bytes"] = panel.n * sizeof(T);
                result["gather_staging_bytes"] = staging.n * sizeof(T);
                result["ormqr_workspace_bytes"] = apply.workspace_bytes();
                result["including_gather_s"] = seconds() - started;
                reference_solver_check(cusolverDnDestroy(local), "streamed validation destroy");
                local = nullptr;
            } catch (const std::exception &e) {
                if (local)
                    cusolverDnDestroy(local);
                result = {{"performed", true}, {"pass", false}, {"error", e.what()}};
            }
            int stop[2] = {-1, 0};
            MPI_Bcast(stop, 2, MPI_INT, 0, MPI_COMM_WORLD);
        } else {
            while (true) {
                int command[2];
                MPI_Bcast(command, 2, MPI_INT, 0, MPI_COMM_WORLD);
                if (command[0] < 0)
                    break;
                gather(command[0], command[1], nullptr, 0);
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
