"""Experiment 5: Stage-Wise Empirical-Bayes Funnel Model + Bellman Continuation Surplus Policy."""
import math
import pickle
import statistics
import time
import numpy as np
import kit
from fast_benchmark import VARIANTS, INFORMATIVE_SOFT, N_OPTIONS


GH_Z = np.array([-2.02018287, -0.95857246, 0.0, 0.95857246, 2.02018287])
GH_S = np.sqrt(2.0) * 0.45 * GH_Z
GH_W = np.array([0.01125741, 0.22207592, 0.53333333, 0.22207592, 0.01125741])



def expected_msmi_gh(ba, bb, fit_hat, goal_sec, drift_offset, ra2, rb2, date_qual_prob=0.2808):
    za = -0.25 + ba + fit_hat + drift_offset + GH_S
    zb = -0.25 + bb + fit_hat + drift_offset + GH_S
    sa = 0.15 + goal_sec + GH_S
    sb = 0.15 + goal_sec + GH_S
    pa = 1.0 / (1.0 + np.exp(-np.clip(za, -20.0, 20.0)))
    pb = 1.0 / (1.0 + np.exp(-np.clip(zb, -20.0, 20.0)))
    qa = 1.0 / (1.0 + np.exp(-np.clip(sa, -20.0, 20.0)))
    qb = 1.0 / (1.0 + np.exp(-np.clip(sb, -20.0, 20.0)))
    integral = float(np.dot(GH_W, pa * pb * qa * qb))
    return ra2 * rb2 * date_qual_prob * integral



def evaluate_bellman_policy(cache, seeds, cfg):
    by_var = {v: [] for v in VARIANTS}
    total_intros = total_mutual = total_dates = total_msmi = total_ask_cost = 0
    covs = []

    ask_mode = cfg.get("ask_mode", "fifo_plus_soft")
    jit_goal_asks = cfg.get("jit_goal_asks", 0)
    use_gh = cfg.get("use_gh", True)
    use_eb = cfg.get("use_eb", True)
    use_online_regime = cfg.get("use_online_regime", True)
    eta_alt = cfg.get("eta_alt", 0.0)
    wait_ratio = cfg.get("wait_ratio", 0.0) # wait if busy alternative is > wait_ratio * current
    wait_max_day = cfg.get("wait_max_day", 42)
    tau_abs = cfg.get("tau_abs", 0.0)


    for v in VARIANTS:
        for s in seeds:
            w, feasible_pairs = cache[(s, v)]
            ms = [dict(m, fields=dict(m["fields"]), field_status=dict(m["field_status"])) for m in w["members"]]
            members = {m["member_id"]: m for m in ms}
            hard_complete = {m_id: all(m["fields"][k] is not None for k in kit.HARD) for m_id, m in members.items()}
            has_declined = {m_id: any(m["field_status"][k] == "declined" for k in kit.HARD) for m_id, m in members.items()}
            busy = {}; paused_day = {}; past = set()
            msmi_count = 0; ask_cost = 0



            adj_feasible = {}
            for a_id, b_id in feasible_pairs:
                adj_feasible.setdefault(a_id, []).append(b_id)
                adj_feasible.setdefault(b_id, []).append(a_id)

            pending_fb = [] 
            member_stats = {}
            obs_acc_X = []
            obs_acc_y = []
            obs_acc_days = []

            w_prior = np.array([0.70, 0.40, 0.25, 0.20])
            w_curr = w_prior.copy()
            drift_hat = 0.0

            for day in range(60):
                new_acc_obs = False
                for item in pending_fb:
                    if item[0] <= day:
                        _, ev_type, act_id, r_b, y_b, x_vec, d_ass = item
                        st = member_stats.setdefault(act_id, {"resp_n": 0, "resp_y": 0, "acc_n": 0, "acc_y": 0, "acc_res": 0.0, "sec_n": 0, "sec_res": 0.0})
                        st["resp_n"] += 1
                        if r_b:
                            st["resp_y"] += 1
                            if ev_type == "intro":
                                st["acc_n"] += 1
                                if y_b: st["acc_y"] += 1
                                pred_p = 1.0 / (1.0 + math.exp(-(-0.25 + float(np.dot(w_curr, x_vec)))))
                                st["acc_res"] += (1.0 if y_b else 0.0) - pred_p
                                obs_acc_X.append(x_vec)
                                obs_acc_y.append(1.0 if y_b else 0.0)
                                obs_acc_days.append(d_ass)
                                new_acc_obs = True
                            elif ev_type == "sec":
                                st["sec_n"] += 1
                                st["sec_res"] += (1.0 if y_b else 0.0) - 0.60
                pending_fb = [item for item in pending_fb if item[0] > day]

                if use_online_regime and new_acc_obs and len(obs_acc_y) >= 12:
                    Xa = np.array(obs_acc_X)
                    ya = np.array(obs_acc_y)
                    da = np.array(obs_acc_days)
                    wt = w_prior.copy()
                    b0 = -0.25
                    for _ in range(3):
                        z = b0 + Xa @ wt
                        p = 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))
                        W = p * (1.0 - p) + 1e-5
                        grad = Xa.T @ (p - ya) + 10.0 * (wt - w_prior)
                        H = (Xa.T * W) @ Xa + 10.0 * np.eye(4)
                        wt -= np.linalg.solve(H, grad)
                        b0 -= np.sum(p - ya) / (np.sum(W) + 4.0)
                    w_curr = wt
                    # Estimate post-day-35 drift offset if day >= 38
                    if day >= 38 and np.sum(da >= 35) >= 6:
                        mask35 = (da >= 35)
                        z35 = -0.25 + Xa[mask35] @ w_curr
                        p35 = 1.0 / (1.0 + np.exp(-np.clip(z35, -20.0, 20.0)))
                        res35 = np.sum(ya[mask35] - p35)
                        w35 = np.sum(p35 * (1.0 - p35)) + 4.0
                        drift_hat = float(np.clip(res35 / w35, -0.6, 0.2))

                avail = [m for m in ms if m["arrived_day"] <= day < m["exit_day"] and paused_day.get(m["member_id"], 100000) > day and busy.get(m["member_id"], -1) <= day]
                avail_set = {m["member_id"] for m in avail}





                budget = 12
                if jit_goal_asks > 0:
                    active_hc_deg = {}
                    for a_id, b_id in feasible_pairs:
                        if a_id in avail_set and b_id in avail_set and hard_complete[a_id] and hard_complete[b_id] and (a_id, b_id) not in past:
                            active_hc_deg[a_id] = active_hc_deg.get(a_id, 0) + 1
                            active_hc_deg[b_id] = active_hc_deg.get(b_id, 0) + 1
                    jit_cands = [
                        m for m in avail
                        if active_hc_deg.get(m["member_id"], 0) >= 1
                        and m["fields"]["relationship_goal"] is None
                        and m["field_status"]["relationship_goal"] == "not_asked"
                    ]
                    jit_cands.sort(key=lambda m: -active_hc_deg[m["member_id"]])
                    for m in jit_cands[:jit_goal_asks]:
                        m["fields"]["relationship_goal"] = m["truth"]["relationship_goal"]
                        m["field_status"]["relationship_goal"] = "observed"
                        budget -= 1
                        ask_cost += 1

                inc_avail = [m for m in avail if not hard_complete[m["member_id"]] and not has_declined[m["member_id"]]]
                ask_hard = inc_avail[: (budget // 3)]
                for m in ask_hard:
                    for k in kit.HARD:
                        m["fields"][k] = m["truth"][k]
                    hard_complete[m["member_id"]] = True
                    budget -= 3
                    ask_cost += 3



                if ask_mode == "fifo_plus_soft" and budget > 0:
                    deg_active = {}
                    for a_id, b_id in feasible_pairs:
                        if hard_complete[a_id] and hard_complete[b_id] and (a_id, b_id) not in past:
                            if a_id in avail_set and b_id in avail_set:
                                deg_active[a_id] = deg_active.get(a_id, 0) + 3
                                deg_active[b_id] = deg_active.get(b_id, 0) + 3
                            else:
                                deg_active[a_id] = deg_active.get(a_id, 0) + 1
                                deg_active[b_id] = deg_active.get(b_id, 0) + 1
                    hc_cands = [m for m in avail if hard_complete[m["member_id"]] and deg_active.get(m["member_id"], 0) > 0]
                    hc_cands.sort(key=lambda m: -deg_active.get(m["member_id"], 0))
                    for k in INFORMATIVE_SOFT:
                        if budget <= 0: break
                        for m in hc_cands:
                            if budget <= 0: break
                            if m["fields"][k] is None and m["field_status"][k] == "not_asked":
                                m["fields"][k] = m["truth"][k]
                                m["field_status"][k] = "observed"
                                budget -= 1
                                ask_cost += 1


                def pair_prob(a_id, b_id):
                    ma, mb = members[a_id], members[b_id]
                    keys_4 = ["relationship_goal", "relationship_pace", "lifestyle", "conversations"]
                    x_vec = []
                    for k in keys_4:
                        va, vb = ma["fields"].get(k), mb["fields"].get(k)
                        if va is not None and vb is not None:
                            x_vec.append(1.0 if va == vb else -0.5)
                        else:
                            x_vec.append(1.5 / N_OPTIONS[k] - 0.5)
                    fit_hat = float(np.dot(w_curr, x_vec))
                    ga, gb = ma["fields"].get("relationship_goal"), mb["fields"].get("relationship_goal")
                    goal_sec = (0.40 if ga == gb else 0.0) if (ga is not None and gb is not None) else 0.20

                    if use_eb:
                        sa = member_stats.get(a_id, {"resp_n": 0, "resp_y": 0, "acc_n": 0, "acc_res": 0.0, "sec_n": 0, "sec_res": 0.0})
                        sb = member_stats.get(b_id, {"resp_n": 0, "resp_y": 0, "acc_n": 0, "acc_res": 0.0, "sec_n": 0, "sec_res": 0.0})
                        ra = (sa["resp_y"] + 3.0 * 0.765) / (sa["resp_n"] + 3.0)
                        rb = (sb["resp_y"] + 3.0 * 0.765) / (sb["resp_n"] + 3.0)
                        # Second moment E[R^2] = mu^2 + var
                        va_r = ra * (1.0 - ra) / (sa["resp_n"] + 4.0)
                        vb_r = rb * (1.0 - rb) / (sb["resp_n"] + 4.0)
                        ra2 = ra * ra + va_r
                        rb2 = rb * rb + vb_r
                        ba = sa["acc_res"] / (2.0 + 0.23 * sa["acc_n"])
                        bb = sb["acc_res"] / (2.0 + 0.23 * sb["acc_n"])
                    else:
                        ra2 = rb2 = 0.765 ** 2
                        ba = bb = 0.0

                    if use_gh:
                        return expected_msmi_gh(ba, bb, fit_hat, goal_sec, drift_hat if day >= 35 else 0.0, ra2, rb2)
                    else:
                        p_acc_a = 1.0 / (1.0 + math.exp(-(-0.25 + ba + fit_hat + (drift_hat if day >= 35 else 0.0))))
                        p_acc_b = 1.0 / (1.0 + math.exp(-(-0.25 + bb + fit_hat + (drift_hat if day >= 35 else 0.0))))
                        p_sec_a = 1.0 / (1.0 + math.exp(-(0.15 + goal_sec)))
                        p_sec_b = 1.0 / (1.0 + math.exp(-(0.15 + goal_sec)))
                        return ra2 * rb2 * p_acc_a * p_acc_b * 0.78 * p_sec_a * p_sec_b * 0.36



                alive_hc = {
                    m["member_id"] for m in ms
                    if m["arrived_day"] <= day < m["exit_day"]
                    and paused_day.get(m["member_id"], 100000) > day
                    and hard_complete[m["member_id"]]
                }
                edge_probs = {}
                node_neighbors = {}
                for a_id, b_id in feasible_pairs:
                    if a_id in alive_hc and b_id in alive_hc and (a_id, b_id) not in past:
                        p_val = pair_prob(a_id, b_id)
                        edge_probs[(a_id, b_id)] = p_val
                        node_neighbors.setdefault(a_id, []).append((b_id, p_val))
                        node_neighbors.setdefault(b_id, []).append((a_id, p_val))


                scored_edges = []
                for (a_id, b_id), p_uv in edge_probs.items():
                    if a_id not in avail_set or b_id not in avail_set:
                        continue


                    alt_a = 0.0
                    for w_id, p_uw in node_neighbors.get(a_id, []):
                        if w_id != b_id:
                            wait_d = max(0, busy.get(w_id, -1) - day)
                            if wait_d <= 8:
                                alt_a = max(alt_a, (0.92 ** wait_d) * p_uw)
                    alt_b = 0.0
                    for w_id, p_vw in node_neighbors.get(b_id, []):
                        if w_id != a_id:
                            wait_d = max(0, busy.get(w_id, -1) - day)
                            if wait_d <= 8:
                                alt_b = max(alt_b, (0.92 ** wait_d) * p_vw)



                    if day < wait_max_day:
                        if wait_ratio > 0 and max(alt_a, alt_b) > wait_ratio * p_uv:
                            continue
                        if tau_abs > 0 and p_uv < tau_abs * (1.0 - day / float(wait_max_day)) and (alt_a > 0 or alt_b > 0):
                            continue

                    decay = max(0.0, 1.0 - day / 50.0)
                    surplus = p_uv - eta_alt * decay * (alt_a + alt_b)
                    scored_edges.append((surplus, (a_id, b_id)))

                scored_edges.sort(key=lambda x: (-x[0], x[1]))
                used = set(); pairs = []
                for _, pair in scored_edges:
                    if pair[0] not in used and pair[1] not in used:
                        pairs.append(pair); used.update(pair)


                for a_id, b_id in sorted(pairs):
                    past.add((a_id, b_id)); total_intros += 1
                    ma, mb = members[a_id], members[b_id]
                    keys_4 = ["relationship_goal", "relationship_pace", "lifestyle", "conversations"]
                    x_vec = []
                    for k in keys_4:
                        va, vb = ma["fields"].get(k), mb["fields"].get(k)
                        if va is not None and vb is not None:
                            x_vec.append(1.0 if va == vb else -0.5)
                        else:
                            x_vec.append(1.5 / N_OPTIONS[k] - 0.5)

                    r = kit.rand_for(s, a_id, b_id, day)
                    shared = r.gauss(0, .45); responded = []; accept = []; response_days = []
                    for actor, other in ((ma, mb), (mb, ma)):
                        resp_i = r.random() < actor["response_rate"]
                        responded.append(resp_i)
                        f, g = actor["truth"], other["truth"]
                        weights = {"relationship_goal": .7, "relationship_pace": .4, "lifestyle": .25, "conversations": .2}
                        if v == "shift":
                            weights = {"relationship_goal": .25, "relationship_pace": .8, "lifestyle": -.25, "conversations": .5}
                        fit = sum(w * (1 if f[k] == g[k] else -.5) for k, w in weights.items())
                        drift = -.5 if v == "drift" and day >= 35 else 0
                        z = -.25 + actor["bias"] + fit + shared + drift
                        acc_i = r.random() < (1 / (1 + kit.math.exp(-z)))
                        accept.append(acc_i)
                        delay = r.randint(1, 7); response_days.append(day + delay)
                        obs_d = (day + delay) if resp_i else (day + 7)
                        pending_fb.append((obs_d, "intro", actor["member_id"], resp_i, acc_i, x_vec, day))
                    busy[a_id] = day + 8; busy[b_id] = day + 8
                    if all(responded) and all(accept):
                        total_mutual += 1
                        date_day = max(response_days) + r.randint(1, 14)
                        if v == "delayed": date_day += r.randint(5, 12)
                        happened = r.random() < .78
                        busy[a_id] = date_day + 6; busy[b_id] = date_day + 6
                        if happened:
                            if date_day <= 100: total_dates += 1
                            seconds = []; on_time = []; obs_days = [date_day]
                            for actor, other in ((ma, mb), (mb, ma)):
                                answer = r.random() < actor["response_rate"]; delay = r.randint(1, 5)
                                p = 1 / (1 + kit.math.exp(-(.15 + actor["second_bias"] + shared + .4 * (actor["truth"]["relationship_goal"] == other["truth"]["relationship_goal"]))))
                                yes = r.random() < p; seconds.append(answer and yes); on_time.append(delay <= 3)
                                obs_d2 = date_day + delay if answer else date_day + 5
                                obs_days.append(obs_d2)
                                pending_fb.append((obs_d2, "sec", actor["member_id"], answer, yes, x_vec, day))
                            if all(seconds):
                                retire_day = max(obs_days)
                                paused_day[a_id] = min(paused_day.get(a_id, 100000), retire_day)
                                paused_day[b_id] = min(paused_day.get(b_id, 100000), retire_day)
                                if date_day - day <= 30 and all(on_time) and retire_day <= 100:
                                    msmi_count += 1; total_msmi += 1
            by_var[v].append(100 * msmi_count / 200)
            total_ask_cost += ask_cost
            covs.append(len({x for p in past for x in p}) / 200)
    overall = statistics.mean([statistics.mean(by_var[v]) for v in VARIANTS])
    return {
        "overall": round(overall, 4),
        "by_var": {v: round(statistics.mean(by_var[v]), 3) for v in VARIANTS},
        "intros": total_intros,
        "mutual": total_mutual,
        "dates": total_dates,
        "msmi": total_msmi,
        "ask_cost": round(total_ask_cost / (len(seeds) * 6), 1),
        "cov": round(statistics.mean(covs), 3)
    }


if __name__ == "__main__":
    with open("/working_dir/c_3af6eefb10334df3/repo/cache/worlds.pkl", "rb") as f:
        CACHE = pickle.load(f)


    test_configs = [
        ("B1. FEBF (GH + EB + OnlineRegime + SoftAsks)", {"use_gh": True, "use_eb": True, "use_online_regime": True}),
        ("B2. FEBF + JIT-Goal(3) Asks", {"use_gh": True, "use_eb": True, "use_online_regime": True, "jit_goal_asks": 3}),
        ("B3. FEBF + Bellman Shadow Price (eta=0.25)", {"use_gh": True, "use_eb": True, "use_online_regime": True, "eta_alt": 0.25}),
        ("B4. FEBF + Bellman Shadow Price (eta=0.40)", {"use_gh": True, "use_eb": True, "use_online_regime": True, "eta_alt": 0.40}),
        ("B5. FEBF + Bellman Wait (ratio=1.35, eta=0.25)", {"use_gh": True, "use_eb": True, "use_online_regime": True, "eta_alt": 0.25, "wait_ratio": 1.35}),
        ("B6. FEBF + Degree-Cond Threshold (tau=0.009, eta=0.25)", {"use_gh": True, "use_eb": True, "use_online_regime": True, "eta_alt": 0.25, "tau_abs": 0.009}),
        ("B7. FEBF + JIT-Goal(2) + Bellman (eta=0.25, tau=0.008)", {"use_gh": True, "use_eb": True, "use_online_regime": True, "jit_goal_asks": 2, "eta_alt": 0.25, "tau_abs": 0.008}),
    ]



    for stage_name, seeds in [("DEVELOPMENT (1001-1008)", range(1001, 1009)), ("SELECTION (2001-2012)", range(2001, 2013))]:
        print(f"\n=== {stage_name} ===")
        for name, cfg in test_configs:
            res = evaluate_bellman_policy(CACHE, seeds, cfg)
            print(f"{name:55s} | MSMI/100={res['overall']:.3f} | MutAcc={res['mutual']:4d} | Dates={res['dates']:4d} | MSMI={res['msmi']:3d} | Cost={res['ask_cost']:5.1f} | {res['by_var']}")
