#pragma once
// Digest of a schedule, excluding fields that do not affect execution.
#include "common.hpp"
namespace tqr {
inline std::string tiled_schedule_digest(const json& sched) {
  json norm = sched;
  if (norm.contains("refusal_prices") &&
      norm["refusal_prices"].contains("admission_budget_bytes"))
    norm["refusal_prices"].erase("admission_budget_bytes");
  return digest(norm.dump());
}

}
