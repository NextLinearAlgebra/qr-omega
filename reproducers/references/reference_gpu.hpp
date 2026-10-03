#pragma once
// GPU helpers of the reference adapters: input generation and numerical checks.
extern "C" void reference_generate(void *a, int nr, int nc, int ld, int m, int nb, int pr, int pc,
                                   int rr, int rc, int first, int word);
extern "C" void reference_r(const void *a, int lda, void *x, int ldx, int nr, int nc, int nb,
                            int pr, int pc, int rr, int rc, int first, int word);
extern "C" double reference_relative(const void *a, const void *b, int nr, int nc, int lda, int ldb,
                                     int word);
