import json
import math
import pickle
import statistics
import sys
import numpy as np


import kit
from run_round2_experiments import paired_stats, run_policy_on_seeds

VARIANTS = ("development", "sparse", "cold_start", "delayed", "shift", "drift")
INFORMATIVE_SOFT = ["relationship_goal", "relationship_pace", "conversations", "lifestyle"]
N_OPTIONS = {"relationship_goal": 2, "relationship_pace": 3, "lifestyle": 3, "conversations": 4}

R_GRID = np.linspace(0.55, 0.98, 44)
ER2_CACHE = {}
for n_obs in range(25):
    for y_obs in range(n_obs + 1):
        lik = (R_GRID ** y_obs) * ((1.0 - R_GRID) ** (n_obs - y_obs))
        post = lik / np.sum(lik)
        ER2_CACHE[(y_obs, n_obs)] = float(np.sum(post * (R_GRID ** 2)))




def run_febf(cache, seeds, kb=2.0408, tau_init=0.006, sparse_guard=True, use_soft_asks=True, use_eb=True, use_regime=True, use_scarcity=True):
    cluster_means = {s: [] for s in seeds}
    by_var = {v: [] for v in VARIANTS}
    intros = both_resp = mutual = dates = both_sec = msmi_tot = ask_cost_tot = 0
    covs = []
    all_preds = []; all_y_resp = []; all_y_mut = []; all_y_date = []; all_y_sec = []; all_y_msmi = []

    for s in seeds:
        for v in VARIANTS:
            w, feasible_pairs = cache[(s, v)]
            ms = [dict(m, fields=dict(m["fields"]), field_status=dict(m["field_status"])) for m in w["members"]]
            members = {m["member_id"]: m for m in ms}
            hard_complete = {m_id: all(m["fields"][k] is not None for k in kit.HARD) for m_id, m in members.items()}
            has_declined = {m_id: any(m["field_status"][k] == "declined" for k in kit.HARD) for m_id, m in members.items()}
            busy = {}; paused_day = {}; past = set()
            msmi_count = 0; ask_cost = 0
            pending_fb = []; member_stats = {}; obs_acc_X = []; obs_acc_y = []
            w_prior = np.array([0.70, 0.40, 0.25, 0.20])
            w_curr = w_prior.copy()
            zones_seen = {m["zone"] for m in ms if m["arrived_day"] == 0}
            is_sparse_pool = len(zones_seen) > 4

            for day in range(60):
                new_obs = False
                for item in pending_fb:
                    if item[0] <= day:
                        _, act_id, oth_id, r_b, a_b, x_vec = item
                        st = member_stats.setdefault(act_id, {"resp_n": 0, "resp_y": 0, "acc_n": 0, "acc_y": 0, "acc_res": 0.0})
                        st["resp_n"] += 1
                        if r_b:
                            st["resp_y"] += 1; st["acc_n"] += 1
                            if a_b: st["acc_y"] += 1
                            pred_p = 1.0 / (1.0 + math.exp(-(-0.25 + float(np.dot(w_curr, x_vec)))))
                            st["acc_res"] += (1.0 if a_b else 0.0) - pred_p
                            obs_acc_X.append(x_vec); obs_acc_y.append(1.0 if a_b else 0.0)
                            new_obs = True
                pending_fb = [item for item in pending_fb if item[0] > day]

                if use_regime and new_obs and len(obs_acc_y) >= 15:
                    Xa = np.array(obs_acc_X); ya = np.array(obs_acc_y)
                    wt = w_prior.copy(); b0 = -0.25
                    for _ in range(3):
                        z = b0 + Xa @ wt
                        p = 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))
                        W = p * (1.0 - p) + 1e-5
                        grad = Xa.T @ (p - ya) + 12.0 * (wt - w_prior)
                        H = (Xa.T * W) @ Xa + 12.0 * np.eye(4)
                        wt -= np.linalg.solve(H, grad)
                        b0 -= np.sum(p - ya) / (np.sum(W) + 5.0)
                    w_curr = wt

                avail = [m for m in ms if m["arrived_day"] <= day < m["exit_day"] and paused_day.get(m["member_id"], 100000) > day and busy.get(m["member_id"], -1) <= day]
                avail_set = {m["member_id"] for m in avail}
                budget = 12
                inc_avail = [m for m in avail if not hard_complete[m["member_id"]] and not has_declined[m["member_id"]]]
                for m in inc_avail[:4]:
                    for k in kit.HARD: m["fields"][k] = m["truth"][k]
                    hard_complete[m["member_id"]] = True
                    budget -= 3; ask_cost += 3

                if use_soft_asks and budget > 0:
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
                                budget -= 1; ask_cost += 1

                active_deg = {}; active_edges = []
                for a_id, b_id in feasible_pairs:
                    if a_id in avail_set and b_id in avail_set and hard_complete[a_id] and hard_complete[b_id] and (a_id, b_id) not in past:
                        active_edges.append((a_id, b_id))
                        active_deg[a_id] = active_deg.get(a_id, 0) + 1
                        active_deg[b_id] = active_deg.get(b_id, 0) + 1

                tau = -999.0 if (sparse_guard and is_sparse_pool) else (tau_init * (1.0 - day / 35.0) if (tau_init > 0 and day < 35) else -999.0)
                scored_edges = []
                edge_pred_map = {}
                for a_id, b_id in active_edges:
                    ma, mb = members[a_id], members[b_id]
                    keys_4 = ["relationship_goal", "relationship_pace", "lifestyle", "conversations"]
                    x_vec = [ (1.0 if ma["fields"].get(k)==mb["fields"].get(k) else -0.5) if (ma["fields"].get(k) is not None and mb["fields"].get(k) is not None) else (1.5/N_OPTIONS[k]-0.5) for k in keys_4 ]
                    fit_hat = float(np.dot(w_curr, x_vec))
                    ga, gb = ma["fields"].get("relationship_goal"), mb["fields"].get("relationship_goal")
                    goal_sec = (0.40 if ga == gb else 0.0) if (ga is not None and gb is not None) else 0.20
                    if use_eb:
                        sa = member_stats.get(a_id, {"resp_n": 0, "resp_y": 0, "acc_n": 0, "acc_res": 0.0})
                        sb = member_stats.get(b_id, {"resp_n": 0, "resp_y": 0, "acc_n": 0, "acc_res": 0.0})
                        ra2 = ER2_CACHE.get((sa["resp_y"], sa["resp_n"]), 0.6014)
                        rb2 = ER2_CACHE.get((sb["resp_y"], sb["resp_n"]), 0.6014)
                        ba = sa["acc_res"] / (kb + 0.23 * sa["acc_n"])
                        bb = sb["acc_res"] / (kb + 0.23 * sb["acc_n"])
                    else:
                        ra2 = rb2 = 0.6014
                        ba = bb = 0.0
                    p_acc_a = 1.0 / (1.0 + math.exp(-(-0.25 + ba + fit_hat)))
                    p_acc_b = 1.0 / (1.0 + math.exp(-(-0.25 + bb + fit_hat)))
                    p_sec_a = 1.0 / (1.0 + math.exp(-(0.15 + goal_sec)))
                    p_sec_b = 1.0 / (1.0 + math.exp(-(0.15 + goal_sec)))
                    p_msmi = ra2 * rb2 * p_acc_a * p_acc_b * 0.78 * p_sec_a * p_sec_b * 0.36
                    edge_pred_map[(a_id, b_id)] = p_msmi
                    sc = p_msmi * (1.0 + 0.03 / math.sqrt(max(1, active_deg[a_id] * active_deg[b_id]))) if use_scarcity else p_msmi
                    if sc >= tau:
                        scored_edges.append((sc, (a_id, b_id)))

                scored_edges.sort(key=lambda x: (-x[0], x[1]))
                used = set(); pairs = []
                for _, pair in scored_edges:
                    if pair[0] not in used and pair[1] not in used:
                        pairs.append(pair); used.update(pair)

                for a_id, b_id in sorted(pairs):
                    past.add((a_id, b_id)); intros += 1
                    ma, mb = members[a_id], members[b_id]
                    keys_4 = ["relationship_goal", "relationship_pace", "lifestyle", "conversations"]
                    x_vec = [ (1.0 if ma["fields"].get(k)==mb["fields"].get(k) else -0.5) if (ma["fields"].get(k) is not None and mb["fields"].get(k) is not None) else (1.5/N_OPTIONS[k]-0.5) for k in keys_4 ]
                    r = kit.rand_for(s, a_id, b_id, day)
                    shared = r.gauss(0, .45); responded = []; accept = []; response_days = []
                    for actor, other in ((ma, mb), (mb, ma)):
                        resp_i = r.random() < actor["response_rate"]
                        responded.append(resp_i)
                        f, g = actor["truth"], other["truth"]
                        weights = {"relationship_goal": .7, "relationship_pace": .4, "lifestyle": .25, "conversations": .2}
                        if v == "shift": weights = {"relationship_goal": .25, "relationship_pace": .8, "lifestyle": -.25, "conversations": .5}
                        fit = sum(w * (1 if f[k] == g[k] else -.5) for k, w in weights.items())
                        drift = -.5 if v == "drift" and day >= 35 else 0
                        z = -.25 + actor["bias"] + fit + shared + drift
                        acc_i = r.random() < (1 / (1 + kit.math.exp(-z)))
                        accept.append(acc_i)
                        delay = r.randint(1, 7); response_days.append(day + delay)
                        pending_fb.append(((day + delay) if resp_i else (day + 7), actor["member_id"], other["member_id"], resp_i, acc_i, x_vec))
                    busy[a_id] = day + 8; busy[b_id] = day + 8
                    b_resp = int(all(responded)); b_mut = int(all(responded) and all(accept)); b_date = 0; b_sec = 0; b_msmi = 0
                    if b_resp: both_resp += 1
                    if b_mut:
                        mutual += 1
                        date_day = max(response_days) + r.randint(1, 14)
                        if v == "delayed": date_day += r.randint(5, 12)
                        happened = r.random() < .78
                        busy[a_id] = date_day + 6; busy[b_id] = date_day + 6
                        if happened:
                            if date_day <= 100: dates += 1; b_date = 1
                            seconds = []; on_time = []; obs_days = [date_day]
                            for actor, other in ((ma, mb), (mb, ma)):
                                answer = r.random() < actor["response_rate"]; delay = r.randint(1, 5)
                                p = 1 / (1 + kit.math.exp(-(.15 + actor["second_bias"] + shared + .4 * (actor["truth"]["relationship_goal"] == other["truth"]["relationship_goal"]))))
                                yes = r.random() < p; seconds.append(answer and yes); on_time.append(delay <= 3)
                                obs_days.append(date_day + delay if answer else date_day + 5)
                            if all(seconds):
                                retire_day = max(obs_days)
                                paused_day[a_id] = min(paused_day.get(a_id, 100000), retire_day)
                                paused_day[b_id] = min(paused_day.get(b_id, 100000), retire_day)
                                if retire_day <= 100: both_sec += 1; b_sec = 1
                                if date_day - day <= 30 and all(on_time) and retire_day <= 100:
                                    msmi_count += 1; msmi_tot += 1; b_msmi = 1
                    all_preds.append(edge_pred_map[(a_id, b_id)])
                    all_y_resp.append(b_resp); all_y_mut.append(b_mut); all_y_date.append(b_date); all_y_sec.append(b_sec); all_y_msmi.append(b_msmi)
            ep_s = 100.0 * msmi_count / 200.0
            cluster_means[s].append(ep_s)
            by_var[v].append(ep_s)
            ask_cost_tot += ask_cost
            covs.append(len({x for p in past for x in p}) / 200.0)
    c_vec = [statistics.mean(cluster_means[s]) for s in seeds]
    return {
        "overall": round(statistics.mean(c_vec), 4),
        "cluster_scores": c_vec,
        "by_var": {v: round(statistics.mean(by_var[v]), 4) for v in VARIANTS},
        "intros": intros,
        "both_resp": both_resp,
        "mutual": mutual,
        "dates": dates,
        "both_sec": both_sec,
        "msmi": msmi_tot,
        "ask_cost": round(ask_cost_tot / (len(seeds) * 6), 1),
        "cov": round(statistics.mean(covs), 3),
        "preds": np.array(all_preds),
        "y_resp": np.array(all_y_resp),
        "y_mut": np.array(all_y_mut),
        "y_date": np.array(all_y_date),
        "y_sec": np.array(all_y_sec),
        "y_msmi": np.array(all_y_msmi),
    }



if __name__ == "__main__":
    with open("/working_dir/c_3af6eefb10334df3/repo/cache/worlds.pkl", "rb") as f:
        CACHE = pickle.load(f)

    for stage_label, srange in [
        ("Development (Seeds 1001-1008, 8 clusters, 48 ep)", list(range(1001, 1009))),
        ("Selection (Seeds 2001-2012, 12 clusters, 72 ep)", list(range(2001, 2013))),
        ("Confirmation A (Seeds 3001-3040, 40 clusters, 240 ep)", list(range(3001, 3041))),
        ("Confirmation B (Seeds 4001-4040, 40 clusters, 240 ep)", list(range(4001, 4041))),
        ("Combined Confirmation (Seeds 3001-3040 + 4001-4040, 80 clusters, 480 ep)", list(range(3001, 3041)) + list(range(4001, 4041))),
    ]:
        g = run_policy_on_seeds(CACHE, srange, {"ask_mode": "fifo_hard_only", "score_mode": "greedy"})
        sf = run_policy_on_seeds(CACHE, srange, {"ask_mode": "fifo_hard_only", "score_mode": "simple_fifo"})
        febf_lap = run_policy_on_seeds(CACHE, srange, {"ask_mode": "fifo_plus_soft", "score_mode": "stagewise_ml", "use_eb": True, "use_online_weights": True, "use_scarcity": True, "tau_init": 0.006, "tau_end_day": 35})
        febf_ex = run_febf(CACHE, srange, 2.0408, 0.006, True)

        st_ex_g = paired_stats(febf_ex["cluster_scores"], g["cluster_scores"])
        st_ex_sf = paired_stats(febf_ex["cluster_scores"], sf["cluster_scores"])
        st_lap_g = paired_stats(febf_lap["cluster_scores"], g["cluster_scores"])

        print(f"\n=== {stage_label} ===")
        print(f"  Greedy       : {g['overall']:.4f} (MSMI={g['msmi']:3d}, Mut={g['mutual']:4d}, Dates={g['dates']:4d}, Cov={g['cov']:.3f}, Cost={g['ask_cost']:.1f}) | {g['by_var']}")
        print(f"  simple_fifo  : {sf['overall']:.4f} (MSMI={sf['msmi']:3d}, Mut={sf['mutual']:4d}, Dates={sf['dates']:4d}, Cov={sf['cov']:.3f}, Cost={sf['ask_cost']:.1f}) | {sf['by_var']}")
        print(f"  FEBF-Laplace : {febf_lap['overall']:.4f} (MSMI={febf_lap['msmi']:3d}, Mut={febf_lap['mutual']:4d}, Dates={febf_lap['dates']:4d}, Cov={febf_lap['cov']:.3f}, Cost={febf_lap['ask_cost']:.1f}) | {febf_lap['by_var']}")
        print(f"    vs Greedy  : diff={st_lap_g['mean_diff']:+.4f} CI={st_lap_g['ci_95']} t={st_lap_g['t_stat']:.3f} p={st_lap_g['p_val']:.4f} dz={st_lap_g['effect_dz']:.3f} WTL={st_lap_g['wtl']}")
        print(f"  FEBF-Exact   : {febf_ex['overall']:.4f} (MSMI={febf_ex['msmi']:3d}, Mut={febf_ex['mutual']:4d}, Dates={febf_ex['dates']:4d}, Cov={febf_ex['cov']:.3f}, Cost={febf_ex['ask_cost']:.1f}) | {febf_ex['by_var']}")
        print(f"    vs Greedy  : diff={st_ex_g['mean_diff']:+.4f} CI={st_ex_g['ci_95']} t={st_ex_g['t_stat']:.3f} p={st_ex_g['p_val']:.4f} dz={st_ex_g['effect_dz']:.3f} WTL={st_ex_g['wtl']}")
        print(f"    vs Round-1 : diff={st_ex_sf['mean_diff']:+.4f} CI={st_ex_sf['ci_95']} t={st_ex_sf['t_stat']:.3f} p={st_ex_sf['p_val']:.4f} dz={st_ex_sf['effect_dz']:.3f} WTL={st_ex_sf['wtl']}")
