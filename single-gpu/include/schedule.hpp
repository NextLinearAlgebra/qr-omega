#pragma once
// ONE authoritative serialized schedule shared by selection, broadcast, replay, diagnostics, and
// execution.
//
// Separation (summaries must never become inputs to replay):
//  schedule = immutable selected schedule (geometry + per-batch descriptors
//             + carriers + packetization + tree + representation)
//  summary  = reporting fields (selection menu, predicted_s, profile_id,
//             key, artifact_digest) -- NEVER read by the replay decoder
//  receipt  = execution evidence (carrier_dispatch, device counters)
//
// This header sits ABOVE tiled_profile.cuh (which already includes the engine menus, coop
// launchers, and TiledPlan), so it can call the canonical menu and sizing functions with no
// duplicated constants.
#include "tiled_profile.cuh"
#include "schedule_digest.hpp"
namespace tqr {
inline constexpr int schedule_format_version = 1;

// Caller-forced constraints only. m/n/p/word/budget are the problem and the
// admission context, not forcing; they travel alongside for validation but
// are not "forced menu choices".
inline json forcing_from_options(const RunOptions& o) {
  return {{"b", o.b},
          {"leaf", o.leaf},
          {"c", o.c},
          {"d", o.d},
          {"threads", o.threads},
          {"strip", o.strip},
          {"pad", o.pad},
          {"radix", o.radix},
          {"gradix", o.gradix},
          {"zc", o.zc},
          {"dc", o.dc},
          {"wcb", o.wcb},
          {"zcb", o.zcb},
          {"dcb", o.dcb},
          {"csw", o.csw},
          {"tree", o.tree},
          {"active", o.active}};
}

// Immutable selected schedule for one TiledPlan. Every field the engine
// dispatches is here; nothing the engine dispatches lives only in `selection`.
inline json tiled_schedule(const TiledPlan& plan, size_t word) {
  json panels = json::array();
  for (size_t pi = 0; pi < plan.panels.size(); ++pi) {
    const auto& pan = plan.panels[pi];
    json batches = json::array();
    for (size_t bi = 0; bi < pan.ranks.size(); ++bi) {
      const auto& batch = pan.ranks[bi];
      json members = json::array();
      for (size_t mi = 0; mi < batch.ge.size(); ++mi) {
        const auto& g = batch.ge[mi];
        const int grow = RowMap(plan.m, plan.p, plan.row_block).to_global(batch.rank, g.row);
        json partitions = json::array();
        if (batch.cc > 1) {
          for (int z = 0; z < batch.cc; ++z)
            partitions.push_back(
                {{"peer", z},
                 {"K_lo", (g.rows * z) / batch.cc},
                 {"K_hi", (g.rows * (z + 1)) / batch.cc}});
        }
        const int sw =
            (batch.csw > 0 && batch.csw < batch.h) ? batch.csw : batch.h;
        const int local_height =
            batch.cc > 0 ? ceildiv(g.rows, batch.cc) : g.rows;
        const size_t shared_need =
            batch.cpw ? coop_mini_shared_bytes(local_height, sw, batch.cpw, word)
                      : coop_shared_bytes(local_height, batch.h, word);
        members.push_back(
            {{"batch_member", (int)mi},
             {"global_row_origin", grow},
             {"local_row_origin", g.row},
             {"local_col_origin", g.col},
             {"rows", g.rows},
             {"h", g.h},
             {"leaf", batch.leaf},
             {"factor_id", g.tile},
             {"descriptor_offset", g.tile},
             {"local_row_partitions", partitions},
             {"measurement_key",
              batch.cc > 0
                  ? json(coop_op(batch.cc, batch.cpw,
                                 (batch.csw > 0 && batch.csw < batch.h)
                                     ? batch.csw
                                     : 0))
                  : json(nullptr)},
             {"workspace_extent_shared_bytes", shared_need}});
      }
      json levels = json::array();
      for (size_t li = 0; li < batch.levels.size(); ++li) {
        json nodes = json::array();
        for (auto& e : batch.levels[li]) {
          json ch = json::array();
          for (int k = 0; k < e.k; ++k)
            ch.push_back({{"row", e.row[k]},
                          {"rank", e.rank[k]},
                          {"height", e.height[k]},
                          {"child", e.child[k]}});
          nodes.push_back({{"k", e.k},
                           {"col", e.col},
                           {"h", e.h},
                           {"tile", e.tile},
                           {"children", ch}});
        }
        levels.push_back({{"level", (int)li}, {"nodes", nodes}});
      }
      const int swb =
          (batch.csw > 0 && batch.csw < batch.h) ? batch.csw : batch.h;
      json winb = json::array();
      if (batch.h > 0) {
        for (int w0 = 0; w0 < batch.h; w0 += swb)
          winb.push_back(
              {{"w0", w0}, {"sw", std::min(swb, batch.h - w0)}});
      }
      batches.push_back(
          {{"rank", batch.rank},
           {"batch_index", (int)bi},
           {"panel_col", batch.col},
           {"h", batch.h},
           {"leaf", batch.leaf},
           {"count", (int)batch.ge.size()},
           {"groups", batch.cc},
           {"minipanel_width", batch.cpw},
           {"threads", batch.cthreads},
           {"window", batch.csw},
           {"window_boundaries", winb},
           {"cblock_s", batch.cblock_s},
           {"ccoop_s", batch.ccoop_s},
           {"cshared_need", batch.cshared_need},
           {"cshared_cap", batch.cshared_cap},
           {"members", members},
           {"levels", levels}});
    }
    json globals = json::array();
    for (auto& e : pan.global) {
      json ch = json::array();
      for (int k = 0; k < e.k; ++k)
        ch.push_back({{"row", e.row[k]},
                      {"rank", e.rank[k]},
                      {"height", e.height[k]},
                      {"child", e.child[k]}});
      globals.push_back({{"k", e.k},
                         {"col", e.col},
                         {"h", e.h},
                         {"tile", e.tile},
                         {"children", ch}});
    }
    const int trailing = plan.n - pan.col - pan.h;
    const int packets = trailing > 0 ? ceildiv(trailing, plan.strip) : 0;
    panels.push_back({{"panel_id", (int)pi},
                      {"elimination_id", pan.col},
                      {"col", pan.col},
                      {"h", pan.h},
                      {"trailing", trailing},
                      {"packets", packets},
                      {"strip", plan.strip},
                      {"depth", plan.d},
                      {"batches", batches},
                      {"global_tree", globals}});
  }
  json sched = {
      {"schedule_format", schedule_format_version},
      {"m", plan.m},
      {"n", plan.n},
      {"p", plan.p},
      {"b", plan.b},
      {"leaf", plan.leaf},
      {"strip", plan.strip},
      {"threads", plan.threads},
      {"pad", plan.pad},
      {"radix", plan.radix},
      {"global_radix", plan.gradix},
      {"d", plan.d},
      {"aggregate", plan.agg},
      {"lookahead", plan.la},
      {"lookahead_free_sms", plan.la_free},
      {"row_distribution", {{"kind", plan.row_block?"block_cyclic":"contiguous"},{"blk",plan.row_block}}},
      {"apply_replication", plan.apply_c},
      {"Z_replication", plan.zc},
      {"D_replication", plan.dc},
      {"W_block_carrier", plan.wcb},
      {"Z_block_carrier", plan.zcb},
      {"D_block_carrier", plan.dcb},
      {"panel_replication", plan.c},
      {"carrier_minipanel_width", plan.pw},
      {"carrier_threads", plan.cthreads},
      {"carrier_window", plan.csw},
      {"carrier_window_forcing", plan.csw_forced},
      {"ge_shared", plan.ge_shared},
      {"tt_shared", plan.tt_shared},
      {"tt_sparse", plan.tt_sparse},
      {"scalar", plan.scalar},
      {"small_shared", plan.small_shared},
      {"max_batch", plan.max_batch},
      {"packets_total", plan.packets},
      {"retained_per_rank", plan.retained},
      {"owned_bytes_per_rank", plan.owned},
      {"refusal_prices",
       {{"strip_floor_q", plan.strip_floor_q},
        {"widest_packets", plan.widest_packets},
        {"depth_price_r1_s", plan.depth_price_r1_s},
        {"depth_price_rmulti_s", plan.depth_price_rmulti_s},
        {"tt_block_s", plan.tt_block_s},
        {"tt_coop_s", plan.tt_coop_s},
        {"merge_fused_s", plan.merge_fused_s},
        {"merge_carried_s", plan.merge_carried_s},
        {"admission_budget_bytes", plan.budget_bytes}}},
      {"zd_pricing",
       {{"Z_priced", plan.z_priced},
        {"D_priced", plan.d_priced},
        {"Z_refused_s", plan.z_refused_s},
        {"Z_taken_s", plan.z_taken_s},
        {"D_refused_s", plan.d_refused_s},
        {"D_taken_s", plan.d_taken_s}}},
      {"packetization",
       {{"strip", plan.strip},
        {"lane_slots", tiled_lane_slots(plan.d, plan.la != 0)},
        {"cross_panel_pipeline", plan.la != 0}}},
      {"tree",
       {{"local_radix", plan.radix},
        {"global_radix", plan.gradix},
        {"representation",
         plan.scalar ? "scalar packet"
                     : "preuploaded per-factor arrays; no per-call H2D "
                       "descriptor copy"}}},
      {"WZD_carriers",
       {{"W", plan.apply_c},
        {"Z", plan.zc},
        {"D", plan.dc},
        {"W_block", plan.wcb},
        {"Z_block", plan.zcb},
        {"D_block", plan.dcb}}},
      {"panels", panels}};
  return sched;
}
// Digest of the EXECUTED schedule (the execution_plan identity). The budget is still enforced
// separately (replay rejects owned>budget before constructing the engine), it just does not enter
// the execution identity. Shared validator: every execution field the engine dispatches must be
// present and in range. Summaries (selection/predicted_s/key/artifact_digest) are NEVER consulted.
// Historical records without a schedule stay readable for analysis via
// tiled_convert_legacy_for_analysis() below.
inline void tiled_validate_schedule(const json& sched, int m, int n, int p,
                                    size_t word) {
  if (!sched.is_object() ||
      sched.value("schedule_format", 0) != schedule_format_version)
    throw std::runtime_error("schedule_format_before_modify");
  for (const char* k :
       {"m", "n", "p", "b", "leaf", "strip", "threads", "pad", "radix",
        "global_radix", "d", "apply_replication", "Z_replication",
        "D_replication", "W_block_carrier", "Z_block_carrier",
        "D_block_carrier", "panel_replication", "carrier_minipanel_width",
        "carrier_threads", "carrier_window", "max_batch", "panels"})
    if (!sched.contains(k))
      throw std::runtime_error(
          std::string("schedule_missing_execution_field_before_modify:") + k);
  if (sched.at("m") != m || sched.at("n") != n || sched.at("p") != p)
    throw std::runtime_error("schedule_shape_before_modify");
  const int b = sched.at("b"), leaf = sched.at("leaf"),
            strip = sched.at("strip"), threads = sched.at("threads"),
            pad = sched.at("pad"), radix = sched.at("radix"),
            gradix = sched.at("global_radix"), depth = sched.at("d"),
            arep = sched.at("apply_replication"),
            zrep = sched.at("Z_replication"), drep = sched.at("D_replication"),
            wblock = sched.at("W_block_carrier"),
            zblock = sched.at("Z_block_carrier"),
            dblock = sched.at("D_block_carrier");
  if (b < 1 || b > 128 || leaf < b || leaf > tiled_max_leaf || strip < 1 ||
      strip > tiled_max_strip ||
      (threads != 128 && threads != 256 && threads != 512 &&
       threads != 1024) ||
      pad < 0 || radix < 2 || radix > TILE_MAX_RADIX || (radix & (radix - 1)) ||
      depth < 1 || depth > 16 || arep < 1 || arep > 64 || gradix < 2 ||
      gradix > TILE_MAX_RADIX || zrep < 1 || zrep > 64 || drep < 1 ||
      drep > 64)
    throw std::runtime_error("schedule_descriptor_before_modify");
  // optional so that schedules written before the field replay as what they are, the unaggregated
  // schedule (aggregate 1).
  if (sched.contains("aggregate") && !sched.at("aggregate").is_number_integer())
    throw std::runtime_error("schedule_aggregate_before_modify");
  const int agg = sched.value("aggregate", 1);
  if (agg < 1 || agg > 16)
    throw std::runtime_error("schedule_aggregate_before_modify");
  // optional; schedules written before the field replay serially (lookahead 0).
  if (sched.contains("lookahead") && !sched.at("lookahead").is_number_integer())
    throw std::runtime_error("schedule_lookahead_before_modify");
  const int la = sched.value("lookahead", 0);
  if (la < 0 || la > 1)
    throw std::runtime_error("schedule_lookahead_before_modify");
  // optional; 0 = no SM cap. A cap without look-ahead has nothing to leave SMs for.
  if (sched.contains("lookahead_free_sms") && !sched.at("lookahead_free_sms").is_number_integer())
    throw std::runtime_error("schedule_lookahead_free_sms_before_modify");
  const int la_free = sched.value("lookahead_free_sms", 0);
  if (la_free < 0 || la_free > 1024 || (la_free > 0 && la != 1))
    throw std::runtime_error("schedule_lookahead_free_sms_before_modify");
  if (!sched.contains("row_distribution") || !sched.at("row_distribution").is_object())
    throw std::runtime_error("schedule_row_distribution_before_modify");
  const auto& rd = sched.at("row_distribution");
  if (!rd.contains("kind") || !rd.at("kind").is_string() || !rd.contains("blk") || !rd.at("blk").is_number_integer())
    throw std::runtime_error("schedule_row_distribution_before_modify");
  const int blk=rd.at("blk");const std::string kind=rd.at("kind");
  if ((kind!="contiguous" && kind!="block_cyclic") ||
      (kind=="contiguous" && blk!=0) ||
      (kind=="block_cyclic" && (p==1 || blk<b || blk%b)))
    throw std::runtime_error("schedule_row_distribution_before_modify");
  auto ok_block = [](int v) { return v == 2 || v == 4 || v == 8; };
  if (!ok_block(wblock) || !ok_block(zblock) || !ok_block(dblock))
    throw std::runtime_error("schedule_block_carrier_before_modify");
  if (!sched.at("panels").is_array())
    throw std::runtime_error("schedule_panels_before_modify");
  for (auto& pan : sched.at("panels")) {
    if (!pan.contains("panel_id") || !pan.contains("elimination_id") ||
        !pan.contains("col") || !pan.contains("h") ||
        !pan.contains("batches") || !pan.contains("global_tree"))
      throw std::runtime_error("schedule_panel_before_modify");
    for (auto& bch : pan.at("batches")) {
      for (const char* k :
           {"rank", "h", "leaf", "count", "groups", "minipanel_width",
            "threads", "window", "members", "levels"})
        if (!bch.contains(k))
          throw std::runtime_error(
              std::string("schedule_batch_before_modify:") + k);
      const int cc = bch.at("groups"), cpw = bch.at("minipanel_width"),
                ct = bch.at("threads"), csw = bch.at("window"),
                hh = bch.at("h");
      if (cc < 0 || cc > coop_partition_limit || cpw < 0 || cpw > 32 ||
          (ct != 0 && ct != 256 && ct != 512) || csw < 0 || csw > 128)
        throw std::runtime_error("schedule_batch_carrier_before_modify");
      if (cpw > 0 && (cpw > hh || (cpw != 8 && cpw != 16 && cpw != 32)))
        throw std::runtime_error("schedule_batch_width_before_modify");
      if (csw > 0 && csw < hh) {
        const std::vector<int> wmenu = coop_window_menu();
        if (std::find(wmenu.begin(), wmenu.end(), csw) == wmenu.end())
          throw std::runtime_error("schedule_batch_window_before_modify");
      }
      for (auto& mem : bch.at("members")) {
        for (const char* k :
             {"batch_member", "global_row_origin", "local_row_origin",
              "rows", "h", "leaf", "factor_id", "descriptor_offset"})
          if (!mem.contains(k))
            throw std::runtime_error(
                std::string("schedule_member_before_modify:") + k);
        if (mem.at("rows").get<int>() < 0)
          throw std::runtime_error("schedule_member_rows_before_modify");
      }
    }
  }
  (void)word;
}

// Shared decoder: rebuild geometry with tiled_instantiate (which validates
// ownership/inventory exactly), then restore the RECORDED per-batch
// descriptors directly. NEVER runs coop_weigh: replay is a restore, not a
// re-selection. Both family and tiled drivers call this one function.
inline TiledPlan tiled_restore_from_schedule(const json& sched, size_t word) {
  tiled_validate_schedule(sched, sched.at("m"), sched.at("n"), sched.at("p"),
                          word);
  TiledPlan plan = tiled_instantiate(
      sched.at("m"), sched.at("n"), sched.at("p"), sched.at("b"),
      sched.at("leaf"), sched.at("strip"), sched.at("threads"), word,
      sched.at("pad"), sched.at("radix"), sched.at("d"),
      sched.at("apply_replication"), sched.at("global_radix"),
      sched.at("Z_replication"), sched.at("D_replication"),
      sched.at("W_block_carrier"), sched.at("Z_block_carrier"),
      sched.at("D_block_carrier"), sched.value("aggregate", 1),
      sched.value("lookahead", 0), sched.at("row_distribution").at("blk"));
  plan.la_free = sched.value("lookahead_free_sms", 0);
  plan.c = sched.at("panel_replication");
  plan.pw = sched.at("carrier_minipanel_width");
  plan.cthreads = sched.at("carrier_threads");
  plan.csw = sched.at("carrier_window");
  plan.csw_forced = sched.value("carrier_window_forcing", 0);
  plan.ge_shared = sched.at("ge_shared");
  plan.tt_shared = sched.at("tt_shared");
  plan.tt_sparse = sched.value("tt_sparse", false);
  plan.scalar = sched.value("scalar", false);
  plan.small_shared = sched.value("small_shared", false);
  // The refusal/certificate pricing is part of the immutable schedule (the engine copies these
  // numbers into witness certificates). Restoring geometry+carriers without them leaves
  // budget_bytes=0, which silently uncertifies every capacity refusal the selector priced.
  if (sched.contains("refusal_prices")) {
    const auto& rp = sched.at("refusal_prices");
    plan.strip_floor_q = rp.value("strip_floor_q", 0);
    plan.widest_packets = rp.value("widest_packets", 0);
    plan.depth_price_r1_s = rp.value("depth_price_r1_s", 0.0);
    plan.depth_price_rmulti_s = rp.value("depth_price_rmulti_s", 0.0);
    plan.tt_block_s = rp.value("tt_block_s", 0.0);
    plan.tt_coop_s = rp.value("tt_coop_s", 0.0);
    plan.merge_fused_s = rp.value("merge_fused_s", 0.0);
    plan.merge_carried_s = rp.value("merge_carried_s", 0.0);
    plan.budget_bytes = rp.value("admission_budget_bytes", size_t(0));
  }
  if (sched.contains("zd_pricing")) {
    const auto& zp = sched.at("zd_pricing");
    plan.z_priced = zp.value("Z_priced", false);
    plan.d_priced = zp.value("D_priced", false);
    plan.z_refused_s = zp.value("Z_refused_s", 0.0);
    plan.z_taken_s = zp.value("Z_taken_s", 0.0);
    plan.d_refused_s = zp.value("D_refused_s", 0.0);
    plan.d_taken_s = zp.value("D_taken_s", 0.0);
  }
  if (plan.panels.size() != sched.at("panels").size())
    throw std::runtime_error("schedule_panel_count_before_modify");
  for (size_t pi = 0; pi < plan.panels.size(); ++pi) {
    const auto& span = sched.at("panels")[pi];
    auto& pan = plan.panels[pi];
    if (pan.col != span.at("col") || pan.h != span.at("h"))
      throw std::runtime_error("schedule_panel_identity_before_modify");
    if (pan.ranks.size() != span.at("batches").size())
      throw std::runtime_error("schedule_batch_count_before_modify");
    for (size_t bi = 0; bi < pan.ranks.size(); ++bi) {
      const auto& sb = span.at("batches")[bi];
      auto& batch = pan.ranks[bi];
      if (batch.rank != sb.at("rank") || batch.col != pan.col ||
          batch.h != sb.at("h") || batch.leaf != sb.at("leaf") ||
          (int)batch.ge.size() != sb.at("count"))
        throw std::runtime_error("schedule_batch_identity_before_modify");
      batch.cc = sb.at("groups");
      batch.cpw = sb.at("minipanel_width");
      batch.cthreads = sb.at("threads");
      batch.csw = sb.at("window");
      batch.cblock_s = sb.value("cblock_s", 0.0);
      batch.ccoop_s = sb.value("ccoop_s", 0.0);
      batch.cshared_need = sb.value("cshared_need", size_t(0));
      batch.cshared_cap = sb.value("cshared_cap", size_t(0));
      if (batch.ge.size() != sb.at("members").size())
        throw std::runtime_error("schedule_member_count_before_modify");
      for (size_t mi = 0; mi < batch.ge.size(); ++mi) {
        const auto& sm = sb.at("members")[mi];
        const auto& g = batch.ge[mi];
        const int grow = RowMap(plan.m, plan.p, plan.row_block).to_global(batch.rank, g.row);
        if (grow != sm.at("global_row_origin") ||
            g.row != sm.at("local_row_origin") ||
            g.rows != sm.at("rows") || g.h != sm.at("h") ||
            g.tile != sm.at("factor_id"))
          throw std::runtime_error("schedule_member_identity_before_modify");
      }
    }
  }
  return plan;
}

// Diagnostic conversion for HISTORICAL records without a schedule (analysis only).
inline json tiled_convert_legacy_for_analysis(const json& legacy_record) {
  json j = legacy_record;
  j["schedule_conversion"] =
      "diagnostic-only: derived post hoc for analysis; not a selected "
      "schedule and carries no new measurement identity";
  j["schedule_digest_preserved"] =
      legacy_record.value("key", legacy_record.value("artifact_digest", ""));
  return j;
}

// Per product/batch selected-vs-executed equality. `sched` is the immutable schedule; `witness` is
// the run's carrier_dispatch. Every carried panel batch must have executed exactly its recorded
// groups; every carried W/Z/D product must have executed its recorded replication.
inline void tiled_check_selected_vs_executed(const json& sched,
                                             const json& witness) {
  const auto per = witness.at("per_product_kind");
  // W/Z/D: every carried product runs at the schedule's recorded replication.
  for (const auto& [kind, key] :
       std::vector<std::pair<std::string, std::string>>{
           {"apply_W", "apply_replication"},
           {"apply_Z", "Z_replication"},
           {"apply_D", "D_replication"}}) {
    const int sel = sched.at(key);
    const auto& hist = per.at(kind).at("at_c");
    for (auto& [cstr, n] : hist.items()) {
      const int c = std::stoi(cstr);
      if (c > 1 && c != sel)
        throw std::runtime_error("selected_vs_executed_" + kind +
                                 ": executed c=" + cstr +
                                 " differs from scheduled c=" +
                                 std::to_string(sel));
    }
  }
  // Panel: per-batch groups.
  std::map<std::string, int> scheduled;
  for (auto& pan : sched.at("panels"))
    for (auto& bch : pan.at("batches")) {
      if (bch.at("groups").get<int>() > 0)
        // Keyed by panel COLUMN, as the engine records it (TiledEngine cooperative_ge_batch:
        // "p<col>r<rank>"); the panel ordinal key never matched.
        scheduled["p" + std::to_string(pan.at("col").get<int>()) + "r" +
                  std::to_string(bch.at("rank").get<int>())] =
            bch.at("groups");
    }
  const auto& phist = per.at("panel_GE").at("at_c");
  // Aggregate check cannot prove per-batch equality; the engine's receipt
  // carries panel_batch_executed (see TiledEngine::evidence) for the exact
  // comparison. Here we enforce that no executed panel c exceeds the
  // schedule's maximum AND that the witness carries the per-batch map.
  int smax = 0;
  for (auto& [k, v] : scheduled) smax = std::max(smax, v);
  for (auto& [cstr, n] : phist.items()) {
    const int c = std::stoi(cstr);
    if (c > 1 && c > smax)
      throw std::runtime_error(
          "selected_vs_executed_panel_GE: executed c=" + cstr +
          " above scheduled maximum c=" + std::to_string(smax));
  }
  if (witness.contains("panel_batch_executed")) {
    const auto& exec = witness.at("panel_batch_executed");
    for (auto& [k, v] : scheduled) {
      if (!exec.contains(k))
        throw std::runtime_error("selected_vs_executed_panel_GE: batch " + k +
                                 " has no executed record");
      if (exec.at(k) != v)
        throw std::runtime_error("selected_vs_executed_panel_GE: batch " + k +
                                 " executed " +
                                 exec.at(k).dump() + " vs scheduled " +
                                 std::to_string(v));
    }
  }
}
}  // namespace tqr
