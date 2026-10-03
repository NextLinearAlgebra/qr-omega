// Isolated override of MAGMA's public block-size tuning functions. The native
// geqrf2_mgpu implementation and shared library remain unchanged. Both the
// input distribution and every native QR call must observe the same value.
#include <magma_v2.h>
#include <atomic>
#include <cstdlib>
#include <dlfcn.h>
#include <stdexcept>
#include <string>

namespace {
std::atomic<unsigned long long> calls[2]{};
using Getter = magma_int_t (*)(magma_int_t, magma_int_t);
magma_int_t get_nb(int which, const char *symbol, magma_int_t m, magma_int_t n) {
    ++calls[which];
    if (const char *value = std::getenv("TQR_MAGMA_QR_NB")) {
        size_t used = 0;
        long long nb = std::stoll(value, &used);
        if (used != std::string(value).size() || nb < 1 || nb > 2048)
            throw std::runtime_error("invalid_MAGMA_QR_tuning_block");
        return magma_int_t(nb);
    }
    auto native = reinterpret_cast<Getter>(dlsym(RTLD_NEXT, symbol));
    if (!native)
        throw std::runtime_error("native_MAGMA_tuning_function_unavailable");
    return native(m, n);
}
} // namespace

extern "C" magma_int_t magma_get_sgeqrf_nb(magma_int_t m, magma_int_t n) {
    return get_nb(0, "magma_get_sgeqrf_nb", m, n);
}
extern "C" magma_int_t magma_get_dgeqrf_nb(magma_int_t m, magma_int_t n) {
    return get_nb(1, "magma_get_dgeqrf_nb", m, n);
}
extern "C" unsigned long long tqr_magma_nb_calls(int which) {
    return calls[which].load();
}
