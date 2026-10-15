import json
import math
import pickle
import random
import statistics
import time
from pathlib import Path
import numpy as np
import kit

VARIANTS = ("development", "sparse", "cold_start", "delayed", "shift", "drift")
INFORMATIVE_SOFT = ["relationship_goal", "relationship_pace", "conversations", "lifestyle"]
N_OPTIONS = {"relationship_goal": 2, "relationship_pace": 3, "lifestyle": 3, "conversations": 4}




def run_policy_on_seeds(cache, seeds, cfg):
    by_var = {v: [] for v in VARIANTS}
    cluster_means = {s: [] for s in seeds}
    total_intros = total_both_resp = total_mutual = total_dates = total_both_sec = total_msmi = total_ask_cost = 0
    covs = []
    all_preds = []
    all_y_resp = []
    all_y_mut = []
    all_y_date = []
    all_y_sec = []
    all_y_msmi = []

    ask_mode = cfg.get("ask_mode", "fifo_hard_only")
    score_mode = cfg.get("score_mode", "greedy")
    tau_init = cfg.get("tau_init", -999.0)
    tau_end_day = cfg.get("tau_end_day", 35)
    use_eb = cfg.get("use_eb", False)
    use_online_weights = cfg.get("use_online_weights", False)
    use_scarcity = cfg.get("use_scarcity", False)



    for s in seeds:
        for v in VARIANTS:
            w, feasible_pairs = cache[(s, v)]
            ms = [dict(m, fields=dict(m["fields"]), field_status=dict(m["field_status"])) for m in w["members"]]
            members = {m["member_id"]: m for m in ms}
            hard_complete = {m_id: all(m["fields"][k] is not None for k in kit.HARD) for m_id, m in members.items()}
            has_declined = {m_id: any(m["field_status"][k] == "declined" for k in kit.HARD) for m_id, m in members.items()}
            busy = {}; paused_day = {}; past = set()
            msmi_count = 0; ask_cost = 0

            pending_fb = []
            member_stats = {}
            obs_acc_X = []
            obs_acc_y = []
            w_prior = np.array([0.70, 0.40, 0.25, 0.20])
            w_curr = w_prior.copy()

            for day in range(60):
                new_obs = False
                for item in pending_fb:
                    if item[0] <= day:
                        _, act_id, oth_id, r_b, a_b, x_vec = item
                        st = member_stats.setdefault(act_id, {"resp_n": 0, "resp_y": 0, "acc_n": 0, "acc_y": 0, "acc_res": 0.0})
                        st["resp_n"] += 1
                        if r_b:
                            st["resp_y"] += 1
                            st["acc_n"] += 1
                            if a_b: st["acc_y"] += 1
                            st["acc_res"] += (1.0 if a_b else 0.0) - 0.45
                            obs_acc_X.append(x_vec)
                            obs_acc_y.append(1.0 if a_b else 0.0)
                            new_obs = True
                pending_fb = [item for item in pending_fb if item[0] > day]


                if use_online_weights and new_obs and len(obs_acc_y) >= 15:
                    Xa = np.array(obs_acc_X)
                    ya = np.array(obs_acc_y)
                    wt = w_prior.copy()
                    b0 = -0.20
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
                if ask_mode == "degree_hard_plus_soft":
                    hc_avail = [m for m in avail if hard_complete[m["member_id"]]]
                    def pot_deg(m):
                        c = 0
                        for o in hc_avail:
                            if kit.eligibility(m, o)["status"] != "infeasible":
                                c += 1
                        return -c
                    inc_avail.sort(key=pot_deg)

                ask_hard = inc_avail[:4]
                for m in ask_hard:
                    for k in kit.HARD:
                        m["fields"][k] = m["truth"][k]
                    hard_complete[m["member_id"]] = True
                    budget -= 3
                    ask_cost += 3

                if ask_mode in ("fifo_plus_soft", "degree_hard_plus_soft") and budget > 0:
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

                active_deg = {}
                active_edges = []
                for a_id, b_id in feasible_pairs:
                    if a_id in avail_set and b_id in avail_set and hard_complete[a_id] and hard_complete[b_id] and (a_id, b_id) not in past:
                        active_edges.append((a_id, b_id))
                        active_deg[a_id] = active_deg.get(a_id, 0) + 1
                        active_deg[b_id] = active_deg.get(b_id, 0) + 1

                if tau_init > -100 and day < tau_end_day:
                    tau = tau_init * (1.0 - day / float(tau_end_day))
                else:
                    tau = -999.0

                scored_edges = []
                edge_pred_map = {}
                for a_id, b_id in active_edges:
                    ma, mb = members[a_id], members[b_id]
                    if score_mode == "greedy":
                        sc = sum(ma["fields"].get(k) is not None and ma["fields"].get(k) == mb["fields"].get(k) for k in kit.SOFT)
                        pred_p = 0.0088 + 0.002 * sc
                    elif score_mode == "simple_fifo":
                        sc = 0.0
                        pred_p = 0.0115
                    elif score_mode == "stagewise_ml":
                        keys_4 = ["relationship_goal", "relationship_pace", "lifestyle", "conversations"]
                        x_vec = []
                        for k in keys_4:
                            va, vb = ma["fields"].get(k), mb["fields"].get(k)
                            if va is not None and vb is not None:
                                x_vec.append(1.0 if va == vb else -0.5)
                            else:
                                mk = N_OPTIONS[k]
                                x_vec.append(1.5 / mk - 0.5)
                        fit_hat = float(np.dot(w_curr, x_vec))
                        ga, gb = ma["fields"].get("relationship_goal"), mb["fields"].get("relationship_goal")
                        if ga is not None and gb is not None:
                            goal_sec = 0.40 if ga == gb else 0.0
                        else:
                            goal_sec = 0.20

                        if use_eb:
                            sa = member_stats.get(a_id, {"resp_n": 0, "resp_y": 0, "acc_n": 0, "acc_y": 0, "acc_res": 0.0})
                            sb = member_stats.get(b_id, {"resp_n": 0, "resp_y": 0, "acc_n": 0, "acc_y": 0, "acc_res": 0.0})
                            ra = (sa["resp_y"] + 3.0 * 0.765) / (sa["resp_n"] + 3.0)
                            rb = (sb["resp_y"] + 3.0 * 0.765) / (sb["resp_n"] + 3.0)
                            ba = sa["acc_res"] / (2.0 + 0.25 * sa["acc_n"])
                            bb = sb["acc_res"] / (2.0 + 0.25 * sb["acc_n"])
                        else:
                            ra = rb = 0.765
                            ba = bb = 0.0

                        p_acc_a = 1.0 / (1.0 + math.exp(-(-0.25 + ba + fit_hat)))
                        p_acc_b = 1.0 / (1.0 + math.exp(-(-0.25 + bb + fit_hat)))
                        p_sec_a = 1.0 / (1.0 + math.exp(-(0.15 + goal_sec)))
                        p_sec_b = 1.0 / (1.0 + math.exp(-(0.15 + goal_sec)))
                        p_msmi = (ra * rb) ** 2 * p_acc_a * p_acc_b * 0.78 * p_sec_a * p_sec_b * 0.36
                        pred_p = p_msmi
                        if use_scarcity:
                            scarcity_bonus = 1.0 + 0.03 / math.sqrt(max(1, active_deg[a_id] * active_deg[b_id]))
                            p_msmi = p_msmi * scarcity_bonus
                        sc = p_msmi
                    edge_pred_map[(a_id, b_id)] = pred_p
                    if sc >= tau:
                        scored_edges.append((sc, (a_id, b_id)))

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
                        pending_fb.append((obs_d, actor["member_id"], other["member_id"], resp_i, acc_i, x_vec))
                    busy[a_id] = day + 8; busy[b_id] = day + 8

                    b_resp = int(all(responded))
                    b_mut = int(all(responded) and all(accept))
                    b_date = 0; b_sec = 0; b_msmi = 0
                    if b_resp: total_both_resp += 1
                    if b_mut:
                        total_mutual += 1
                        date_day = max(response_days) + r.randint(1, 14)
                        if v == "delayed": date_day += r.randint(5, 12)
                        happened = r.random() < .78
                        busy[a_id] = date_day + 6; busy[b_id] = date_day + 6
                        if happened:
                            if date_day <= 100:
                                total_dates += 1
                                b_date = 1
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
                                if retire_day <= 100:
                                    total_both_sec += 1
                                    b_sec = 1
                                if date_day - day <= 30 and all(on_time) and retire_day <= 100:
                                    msmi_count += 1; total_msmi += 1
                                    b_msmi = 1
                    all_preds.append(edge_pred_map[(a_id, b_id)])
                    all_y_resp.append(b_resp)
                    all_y_mut.append(b_mut)
                    all_y_date.append(b_date)
                    all_y_sec.append(b_sec)
                    all_y_msmi.append(b_msmi)

            ep_score = 100.0 * msmi_count / 200.0
            by_var[v].append(ep_score)
            cluster_means[s].append(ep_score)
            total_ask_cost += ask_cost
            covs.append(len({x for p in past for x in p}) / 200.0)


    c_vec = [statistics.mean(cluster_means[s]) for s in seeds]
    overall = statistics.mean(c_vec)
    return {
        "overall": round(overall, 4),
        "by_var": {v: round(statistics.mean(by_var[v]), 4) for v in VARIANTS},
        "cluster_scores": c_vec,
        "intros": total_intros,
        "both_resp": total_both_resp,
        "mutual": total_mutual,
        "dates": total_dates,
        "both_sec": total_both_sec,
        "msmi": total_msmi,
        "ask_cost": round(total_ask_cost / (len(seeds) * 6), 2),
        "cov": round(statistics.mean(covs), 4),
        "preds": np.array(all_preds),
        "y_mut": np.array(all_y_mut),
        "y_sec": np.array(all_y_sec),
        "y_msmi": np.array(all_y_msmi),
    }




def paired_stats(cand_scores, base_scores):
    diffs = np.array(cand_scores) - np.array(base_scores)
    n = len(diffs)
    mean_d = float(np.mean(diffs))
    sd_d = float(np.std(diffs, ddof=1))
    se_d = sd_d / math.sqrt(n) if n > 1 else 0.0
    t_stat = mean_d / se_d if se_d > 1e-12 else 0.0
    # two-sided normal/t p-value approx
    # exact t critical for df=39 is 2.0227, for df=79 is 1.9905
    t_crit = 2.0227 if n <= 40 else 1.9905
    ci_low = mean_d - t_crit * se_d
    ci_high = mean_d + t_crit * se_d
    # normal approx p-value
    p_val = math.erfc(abs(t_stat) / math.sqrt(2.0))
    wins = int(np.sum(diffs > 1e-9))
    ties = int(np.sum(np.abs(diffs) <= 1e-9))
    losses = int(np.sum(diffs < -1e-9))
    return {
        "mean_diff": round(mean_d, 4),
        "sd_diff": round(sd_d, 4),
        "se_diff": round(se_d, 4),
        "t_stat": round(t_stat, 3),
        "p_val": round(p_val, 4),
        "ci_95": [round(ci_low, 4), round(ci_high, 4)],
        "effect_dz": round(mean_d / sd_d, 3) if sd_d > 1e-12 else 0.0,
        "wtl": f"{wins}/{ties}/{losses}"
    }

if __name__ == "__main__":
    with open("/working_dir/c_3af6eefb10334df3/repo/cache/worlds.pkl", "rb") as f:
        CACHE = pickle.load(f)

    policies = [
        ("greedy", {"ask_mode": "fifo_hard_only", "score_mode": "greedy"}),
        ("simple_fifo", {"ask_mode": "fifo_hard_only", "score_mode": "simple_fifo"}),
        ("stagewise_fifo_hard", {"ask_mode": "fifo_hard_only", "score_mode": "stagewise_ml"}),
        ("stagewise_soft_asks", {"ask_mode": "fifo_plus_soft", "score_mode": "stagewise_ml"}),
        ("stagewise_eb_soft", {"ask_mode": "fifo_plus_soft", "score_mode": "stagewise_ml", "use_eb": True}),
        ("stagewise_eb_regime", {"ask_mode": "fifo_plus_soft", "score_mode": "stagewise_ml", "use_eb": True, "use_online_weights": True}),
        ("stagewise_eb_regime_scarcity", {"ask_mode": "fifo_plus_soft", "score_mode": "stagewise_ml", "use_eb": True, "use_online_weights": True, "use_scarcity": True}),
        ("stagewise_full_tau006", {"ask_mode": "fifo_plus_soft", "score_mode": "stagewise_ml", "use_eb": True, "use_online_weights": True, "use_scarcity": True, "tau_init": 0.006, "tau_end_day": 35}),
        ("stagewise_full_tau008", {"ask_mode": "fifo_plus_soft", "score_mode": "stagewise_ml", "use_eb": True, "use_online_weights": True, "use_scarcity": True, "tau_init": 0.008, "tau_end_day": 35}),
    ]

    for stage_label, seed_range in [
        ("Confirmation A (Seeds 3001-3040, 240 ep)", range(3001, 3041)),
        ("Confirmation B (Seeds 4001-4040, 240 ep)", range(4001, 4041)),
        ("Combined Confirmation (80 clusters, 480 ep)", list(range(3001, 3041)) + list(range(4001, 4041))),
    ]:
        print(f"\n================ {stage_label} ================")
        stage_res = {}
        for pname, pcfg in policies:
            stage_res[pname] = run_policy_on_seeds(CACHE, seed_range, pcfg)
        base_c = stage_res["greedy"]["cluster_scores"]
        for pname, _ in policies:
            r = stage_res[pname]
            st = paired_stats(r["cluster_scores"], base_c)
            print(f"{pname:30s} | MSMI/100={r['overall']:.3f} | diff={st['mean_diff']:+.3f} {st['ci_95']} p={st['p_val']:.3f} WTL={st['wtl']} | Mut={r['mutual']:4d} Dates={r['dates']:4d} MSMI={r['msmi']:3d} | {r['by_var']}")
