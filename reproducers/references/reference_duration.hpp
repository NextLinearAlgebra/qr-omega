#pragma once
#include "common.hpp"

namespace qr_omega {
// Observe a completed, globally synchronized operation. Never interrupt a
// native library kernel. Stop further repetitions and still validate factors.
class ReferenceDurationLimit {
    double seconds_ = 0, longest_ = 0;
    bool exceeded_ = false;

  public:
    ReferenceDurationLimit() {
        if (const char *value = std::getenv("QR_OMEGA_REFERENCE_MAX_FACTOR_SECONDS")) {
            size_t used = 0;
            seconds_ = std::stod(value, &used);
            if (used != std::string(value).size() || !std::isfinite(seconds_) || seconds_ < 0)
                throw std::runtime_error("invalid_reference_duration_limit");
        }
    }
    bool observe(double elapsed) {
        longest_ = std::max(longest_, elapsed);
        exceeded_ = exceeded_ || (seconds_ > 0 && elapsed > seconds_);
        return exceeded_;
    }
    bool exceeded() const {
        return exceeded_;
    }
    json record() const {
        return {
            {"max_operation_seconds", seconds_},
            {"exceeded", exceeded_},
            {"longest_completed_operation_s", longest_},
            {"policy",
             "stop subsequent repetitions after an over-limit completed operation; preserve and validate completed output"}};
    }
};
} // namespace qr_omega
