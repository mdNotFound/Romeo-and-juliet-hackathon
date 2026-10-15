import json
import math
import pickle
import statistics
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import kit
from eval_final_febf import run_febf, VARIANTS
from run_round2_experiments import paired_stats, run_policy_on_seeds



def roc_auc(y_true, y_score):
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    _, inv, counts = np.unique(y_score, return_inverse=True, return_counts=True)
    sum_ranks = np.bincount(inv, weights=ranks)
    avg_ranks = sum_ranks / counts
    r_pos = avg_ranks[inv][y_true == 1].sum()
    n_pos = len(pos); n_neg = len(neg)
    return (r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)




def ece_score(y_true, y_prob, n_bins=5):
    bins = np.linspace(0.0, np.max(y_prob) + 1e-6, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if m.sum() > 0:
            ece += (m.sum() / len(y_true)) * abs(y_true[m].mean() - y_prob[m].mean())
    return ece



if __name__ == "__main__":
    with open("/working_dir/c_3af6eefb10334df3/repo/cache/worlds.pkl", "rb") as f:
        CACHE = pickle.load(f)

    s_conf_a = list(range(3001, 3041))
    s_conf_b = list(range(4001, 4041))
    s_comb = s_conf_a + s_conf_b

    print("=== RUNNING ABLATION SUITE ON CONF A (3001-3040), CONF B (4001-4040), AND COMBINED 80 CLUSTERS ===")
    ablations = [
        ("FEBF-Exact (Full Round 2 Policy)", dict(kb=2.0408, tau_init=0.006, sparse_guard=True, use_soft_asks=True, use_eb=True, use_regime=True, use_scarcity=True)),
        ("w/o Sparse Pool Guard", dict(kb=2.0408, tau_init=0.006, sparse_guard=False, use_soft_asks=True, use_eb=True, use_regime=True, use_scarcity=True)),
        ("w/o Reservation Threshold (tau=0)", dict(kb=2.0408, tau_init=-999.0, sparse_guard=True, use_soft_asks=True, use_eb=True, use_regime=True, use_scarcity=True)),
        ("w/o Scarcity Tie-Breaker", dict(kb=2.0408, tau_init=0.006, sparse_guard=True, use_soft_asks=True, use_eb=True, use_regime=True, use_scarcity=False)),
        ("w/o Online Regime Adaptation", dict(kb=2.0408, tau_init=0.006, sparse_guard=True, use_soft_asks=True, use_eb=True, use_regime=False, use_scarcity=True)),
        ("w/o Online Empirical-Bayes (EB)", dict(kb=2.0408, tau_init=0.006, sparse_guard=True, use_soft_asks=True, use_eb=False, use_regime=False, use_scarcity=True)),
        ("w/o Leftover Soft-Field Asks", dict(kb=2.0408, tau_init=0.006, sparse_guard=True, use_soft_asks=False, use_eb=True, use_regime=True, use_scarcity=True)),
        ("Stage-Wise Static (FIFO Hard Only)", dict(kb=2.0408, tau_init=-999.0, sparse_guard=True, use_soft_asks=False, use_eb=False, use_regime=False, use_scarcity=False)),
    ]

    g_all = run_policy_on_seeds(CACHE, s_comb, {"ask_mode": "fifo_hard_only", "score_mode": "greedy"})
    sf_all = run_policy_on_seeds(CACHE, s_comb, {"ask_mode": "fifo_hard_only", "score_mode": "simple_fifo"})

    ablation_summary = []
    full_res_all = None



    for name, kw in ablations:
        rall = run_febf(CACHE, s_comb, **kw)
        if full_res_all is None:
            full_res_all = rall
        st_all = paired_stats(rall["cluster_scores"], g_all["cluster_scores"])
        ca_vec = rall["cluster_scores"][:40]; cb_vec = rall["cluster_scores"][40:]
        ga_vec = g_all["cluster_scores"][:40]; gb_vec = g_all["cluster_scores"][40:]
        sta = paired_stats(ca_vec, ga_vec); stb = paired_stats(cb_vec, gb_vec)
        ca_m = statistics.mean(ca_vec); cb_m = statistics.mean(cb_vec)
        da = sta["mean_diff"]; db = stb["mean_diff"]
        dall = st_all["mean_diff"]; pall = st_all["p_val"]
        ov = rall["overall"]; ms = rall["msmi"]; mut = rall["mutual"]
        print(f"{name:36s} | ConfA={ca_m:.4f} ({da:+.4f}) | ConfB={cb_m:.4f} ({db:+.4f}) | Comb80={ov:.4f} (MSMI={ms:3d}, Mut={mut:4d}, diff={dall:+.4f}, p={pall:.4f})")
        ablation_summary.append({
            "name": name,
            "conf_a": round(ca_m, 4),
            "conf_a_diff": da,
            "conf_b": round(cb_m, 4),
            "conf_b_diff": db,
            "comb80": ov,
            "comb80_msmi": ms,
            "comb80_mut": mut,
            "comb80_dates": rall["dates"],
            "comb80_cost": rall["ask_cost"],
            "comb80_cov": rall["cov"],
            "comb80_diff": dall,
            "comb80_ci": st_all["ci_95"],
            "comb80_p": pall,
            "comb80_wtl": st_all["wtl"],
            "by_var": rall["by_var"]
        })

    print("\n=== FUNNEL & CALIBRATION (Combined 80 Clusters, 480 episodes) ===")
    for mname, r in [("Round-1 simple_fifo", sf_all), ("Greedy Baseline", g_all), ("FEBF-Exact (Round 2)", full_res_all)]:
        preds = r["preds"]; ym = r["y_mut"]; ys = r["y_sec"]; yq = r["y_msmi"]
        auc_m = roc_auc(ym, preds); auc_s = roc_auc(ys, preds); auc_q = roc_auc(yq, preds)
        brier = float(np.mean((yq - preds) ** 2))
        signed_err = float(np.mean(preds - yq))
        ece = float(ece_score(yq, preds, n_bins=5))
        q80 = np.percentile(preds, 80); q20 = np.percentile(preds, 20)
        top20_rate = float(yq[preds >= q80].mean()) * 100.0
        bot20_rate = float(yq[preds <= q20].mean()) * 100.0
        lift20 = float(yq[preds >= q80].mean() / max(1e-9, yq.mean()))
        intros = r["intros"]; br = r["both_resp"]; mut = r["mutual"]; dt = r["dates"]; bs = r["both_sec"]; ms = r["msmi"]
        print(f"  {mname:24s} | Intros={intros:5d} BothResp={br:5d} MutAcc={mut:4d} Dates={dt:4d} Both2nd={bs:4d} MSMI={ms:3d}")
        print(f"    Pred={preds.mean():.5f} Obs={yq.mean():.5f} SignedErr={signed_err:+.5f} ECE={ece:.5f} Brier={brier:.6f} | AUC(Mut)={auc_m:.3f} AUC(2nd)={auc_s:.3f} AUC(MSMI)={auc_q:.3f} | Top20%={top20_rate:.2f}% Bot20%={bot20_rate:.2f}% Lift={lift20:.2f}x")

        if "FEBF" in mname:
            qs = np.percentile(preds, [0, 20, 40, 60, 80, 100])
            qs[-1] += 1e-6
            for bi in range(5):
                mask = (preds >= qs[bi]) & (preds < qs[bi+1])
                print(f"      Quintile {bi+1}: n={mask.sum():5d}, pred={preds[mask].mean()*100:.3f}%, obs_msmi={yq[mask].mean()*100:.3f}%, obs_mut={ym[mask].mean()*100:.2f}%, obs_2nd={ys[mask].mean()*100:.2f}%")
