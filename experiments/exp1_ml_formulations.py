import sys, pickle, math, random, json
import numpy as np

sys.path.insert(0, "/working_dir/c_3af6eefb10334df3/repo")
import kit

with open("/working_dir/c_3af6eefb10334df3/repo/cache/worlds.pkl", "rb") as f:
    CACHE = pickle.load(f)

VARIANTS = ("development", "sparse", "cold_start", "delayed", "shift", "drift")



def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))

def fit_logistic_irls(X, y, l2=1.0, max_iter=25):
    n, d = X.shape
    Xb = np.column_stack([np.ones(n), X])
    w = np.zeros(d + 1)
    p_bar = np.clip(y.mean(), 1e-4, 1 - 1e-4)
    w[0] = math.log(p_bar / (1.0 - p_bar))
    reg = np.diag([0.0] + [l2] * d)
    for _ in range(max_iter):
        p = sigmoid(Xb @ w)
        W = np.clip(p * (1.0 - p), 1e-6, None)
        grad = Xb.T @ (p - y) + reg @ w
        H = (Xb.T * W) @ Xb + reg + 1e-6 * np.eye(d + 1)
        step = np.linalg.solve(H, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-5:
            break
    return w


def predict_logistic(X, w):
    Xb = np.column_stack([np.ones(len(X)), X])
    return sigmoid(Xb @ w)



def fit_gbdt_stumps(X, y, n_estimators=40, lr=0.1, min_leaf=25):
    n, d = X.shape
    p_bar = np.clip(y.mean(), 1e-4, 1 - 1e-4)
    f0 = math.log(p_bar / (1.0 - p_bar))
    F = np.full(n, f0)
    trees = []
    for _ in range(n_estimators):
        p = sigmoid(F)
        r = y - p
        w = np.clip(p * (1.0 - p), 1e-5, None)
        # Fit depth-1 or depth-2 split
        best_gain = -1.0
        best_tree = None
        for j in range(d):
            vals = np.unique(X[:, j])
            if len(vals) > 10:
                vals = np.percentile(vals, [20, 40, 50, 60, 80])
            for thr in vals:
                left = X[:, j] <= thr
                right = ~left
                if left.sum() < min_leaf or right.sum() < min_leaf:
                    continue
                gl, wl = r[left].sum(), w[left].sum()
                gr, wr = r[right].sum(), w[right].sum()
                gain = (gl * gl) / (wl + 1.0) + (gr * gr) / (wr + 1.0)
                if gain > best_gain:
                    best_gain = gain
                    best_tree = (j, thr, gl / (wl + 1.0), gr / (wr + 1.0))
        if best_tree is None:
            break
        j, thr, vl, vr = best_tree
        F += lr * np.where(X[:, j] <= thr, vl, vr)
        trees.append(best_tree)
    return (f0, lr, trees)

def predict_gbdt(X, model):
    f0, lr, trees = model
    F = np.full(len(X), f0)
    for j, thr, vl, vr in trees:
        F += lr * np.where(X[:, j] <= thr, vl, vr)
    return sigmoid(F)



def fit_mlp(X, y, hidden=16, lr=0.05, l2=1e-2, epochs=120, seed=42):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    W1 = rng.normal(0, 0.1, size=(d, hidden))
    b1 = np.zeros(hidden)
    W2 = rng.normal(0, 0.1, size=(hidden, 1))
    p_bar = np.clip(y.mean(), 1e-4, 1 - 1e-4)
    b2 = np.array([math.log(p_bar / (1.0 - p_bar))])
    Y = y.reshape(-1, 1)
    for ep in range(epochs):
        H_pre = X @ W1 + b1
        H = np.tanh(H_pre)
        logits = H @ W2 + b2
        P = sigmoid(logits)
        dlogits = (P - Y) / n
        dW2 = H.T @ dlogits + l2 * W2
        db2 = dlogits.sum(axis=0)
        dH = dlogits @ W2.T
        dH_pre = dH * (1.0 - H * H)
        dW1 = X.T @ dH_pre + l2 * W1
        db1 = dH_pre.sum(axis=0)
        W1 -= lr * dW1; b1 -= lr * db1
        W2 -= lr * dW2; b2 -= lr * db2
    return (W1, b1, W2, b2)


def predict_mlp(X, model):
    W1, b1, W2, b2 = model
    H = np.tanh(X @ W1 + b1)
    return sigmoid((H @ W2 + b2).ravel())




def roc_auc(y_true, y_score):
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    # handle ties by average rank
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



def extract_pair_features(ma, mb, day, member_stats, full_obs=False):
    fa = ma["truth"] if full_obs else ma["fields"]
    fb = mb["truth"] if full_obs else mb["fields"]
    feats = []
    for k in kit.SOFT:
        va, vb = fa.get(k), fb.get(k)
        if va is not None and vb is not None:
            feats.extend([1.0 if va == vb else 0.0, 1.0 if va != vb else 0.0])
        else:
            feats.extend([0.0, 0.0])




    ga, gb = fa.get("relationship_goal"), fb.get("relationship_goal")
    pa, pb = fa.get("relationship_pace"), fb.get("relationship_pace")
    ca, cb = fa.get("conversations"), fb.get("conversations")
    feats.append(1.0 if (ga is not None and gb is not None and ga == gb and pa is not None and pb is not None and pa == pb) else 0.0)
    feats.append(1.0 if (pa is not None and pb is not None and pa == pb and ca is not None and cb is not None and ca == cb) else 0.0)
    # Historical member response & acceptance posteriors from past introductions observed BEFORE `day`
    sa = member_stats.get(ma["member_id"], {"resp_n": 0, "resp_y": 0, "acc_n": 0, "acc_y": 0})
    sb = member_stats.get(mb["member_id"], {"resp_n": 0, "resp_y": 0, "acc_n": 0, "acc_y": 0})
    ra_hat = (sa["resp_y"] + 2.0 * 0.765) / (sa["resp_n"] + 2.0)
    rb_hat = (sb["resp_y"] + 2.0 * 0.765) / (sb["resp_n"] + 2.0)
    aa_hat = (sa["acc_y"] + 2.0 * 0.45) / (sa["acc_n"] + 2.0)
    ab_hat = (sb["acc_y"] + 2.0 * 0.45) / (sb["acc_n"] + 2.0)
    feats.extend([ra_hat * rb_hat, aa_hat * ab_hat, day / 60.0, 1.0 if day >= 35 else 0.0])
    return feats



def collect_dataset(seeds, full_soft_obs=False):
    X, y_resp, y_acc_cond, y_mutual_acc, y_second_pos, y_msmi = [], [], [], [], [], []
    for s in seeds:
        for v in VARIANTS:
            w, feasible_pairs = CACHE[(s, v)]
            ms = [ dict(m, fields=dict(m["fields"]), field_status=dict(m["field_status"])) for m in w["members"] ]
            members = {m["member_id"]: m for m in ms}
            hard_complete = {m_id: all(m["fields"][k] is not None for k in kit.HARD) for m_id, m in members.items()}
            has_declined = {m_id: any(m["field_status"][k] == "declined" for k in kit.HARD) for m_id, m in members.items()}
            busy = {}; paused_day = {}; past = set()
            pending_fb = []
            member_stats = {}

            for day in range(60):
                for obs_d, m_id, r_b, a_b in pending_fb:
                    if obs_d <= day:
                        st = member_stats.setdefault(m_id, {"resp_n": 0, "resp_y": 0, "acc_n": 0, "acc_y": 0})
                        st["resp_n"] += 1
                        if r_b:
                            st["resp_y"] += 1
                            st["acc_n"] += 1
                            if a_b: st["acc_y"] += 1
                pending_fb = [item for item in pending_fb if item[0] > day]

                avail = [m for m in ms if m["arrived_day"] <= day < m["exit_day"] and paused_day.get(m["member_id"], 100000) > day and busy.get(m["member_id"], -1) <= day]
                ask_m = [m for m in avail if not hard_complete[m["member_id"]] and not has_declined[m["member_id"]]][:4]
                for m in ask_m:
                    for k in kit.HARD: m["fields"][k] = m["truth"][k]
                    hard_complete[m["member_id"]] = True
                avail_set = {m["member_id"] for m in avail}
                edges = []
                for a_id, b_id in feasible_pairs:
                    if a_id in avail_set and b_id in avail_set and hard_complete[a_id] and hard_complete[b_id] and (a_id, b_id) not in past:
                        ma, mb = members[a_id], members[b_id]
                        sc = sum(ma["fields"].get(k) is not None and ma["fields"].get(k) == mb["fields"].get(k) for k in kit.SOFT)
                        edges.append((-sc, (a_id, b_id)))
                used = set(); pairs = []
                for _, pair in sorted(edges):
                    if pair[0] not in used and pair[1] not in used:
                        pairs.append(pair); used.update(pair)
                for a_id, b_id in sorted(pairs):
                    past.add((a_id, b_id))
                    ma, mb = members[a_id], members[b_id]
                    feat = extract_pair_features(ma, mb, day, member_stats, full_obs=full_soft_obs)
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
                        pending_fb.append((obs_d, actor["member_id"], resp_i, acc_i))
                    busy[a_id] = day + 8; busy[b_id] = day + 8
                    both_resp = int(all(responded))
                    mut_acc = int(all(responded) and all(accept))
                    both_sec = 0; msmi = 0
                    if mut_acc:
                        date_day = max(response_days) + r.randint(1, 14)
                        if v == "delayed": date_day += r.randint(5, 12)
                        happened = r.random() < .78
                        busy[a_id] = date_day + 6; busy[b_id] = date_day + 6
                        if happened:
                            seconds = []; on_time = []; obs_days = [date_day]
                            for actor, other in ((ma, mb), (mb, ma)):
                                answer = r.random() < actor["response_rate"]; delay = r.randint(1, 5)
                                p = 1 / (1 + kit.math.exp(-(.15 + actor["second_bias"] + shared + .4 * (actor["truth"]["relationship_goal"] == other["truth"]["relationship_goal"]))))
                                yes = r.random() < p; seconds.append(answer and yes); on_time.append(delay <= 3)
                                obs_days.append(date_day + delay if answer else date_day + 5)
                            if all(seconds):
                                both_sec = 1
                                retire_day = max(obs_days)
                                paused_day[a_id] = min(paused_day.get(a_id, 100000), retire_day)
                                paused_day[b_id] = min(paused_day.get(b_id, 100000), retire_day)
                                if date_day - day <= 30 and all(on_time) and retire_day <= 100:
                                    msmi = 1
                    X.append(feat)
                    y_resp.append(both_resp)
                    y_acc_cond.append(int(all(accept)) if both_resp else -1)
                    y_mutual_acc.append(mut_acc)
                    y_second_pos.append(both_sec)
                    y_msmi.append(msmi)
    return np.array(X), np.array(y_resp), np.array(y_acc_cond), np.array(y_mutual_acc), np.array(y_second_pos), np.array(y_msmi)




for mode_name, full_obs in [("Regime A: Partial Soft Obs (FIFO Hard-Only)", False), ("Regime B: Clarified Soft Obs (Post-Hard Soft Queries)", True)]:
    X_tr, yr_tr, yac_tr, ym_tr, ys_tr, yq_tr = collect_dataset(range(101, 106), full_soft_obs=full_obs)
    X_va, yr_va, yac_va, ym_va, ys_va, yq_va = collect_dataset(range(1001, 1009), full_soft_obs=full_obs)
    print(f"\n=== {mode_name} ===")
    print(f"Train intros={len(X_tr)} (MutAcc={ym_tr.sum()}, 2ndPos={ys_tr.sum()}, MSMI={yq_tr.sum()})")
    print(f"Val   intros={len(X_va)} (MutAcc={ym_va.sum()}, 2ndPos={ys_va.sum()}, MSMI={yq_va.sum()})")


    p_greedy = X_va[:, 0:14:2].sum(axis=1) / 7.0 * 0.02


    w_term = fit_logistic_irls(X_tr, yq_tr, l2=5.0)
    p_term_lr = predict_logistic(X_va, w_term)


    gbdt_term = fit_gbdt_stumps(X_tr, yq_tr, n_estimators=25, lr=0.1, min_leaf=40)
    p_term_gbdt = predict_gbdt(X_va, gbdt_term)


    mlp_term = fit_mlp(X_tr, yq_tr, hidden=12, lr=0.05, l2=0.02, epochs=150)
    p_term_mlp = predict_mlp(X_va, mlp_term)

    w_s1 = fit_logistic_irls(X_tr, yr_tr, l2=2.0)
    mask_r = (yac_tr >= 0)
    w_s2 = fit_logistic_irls(X_tr[mask_r], yac_tr[mask_r], l2=2.0)
    mask_m = (ym_tr == 1)

    cols_s3 = [0, 1, 16, 17]
    w_s3 = fit_logistic_irls(X_tr[mask_m][:, cols_s3], ys_tr[mask_m], l2=2.0)
    
    p_s1 = predict_logistic(X_va, w_s1)
    p_s2 = predict_logistic(X_va, w_s2)
    p_s3 = predict_logistic(X_va[:, cols_s3], w_s3)
    p_stagewise = p_s1 * p_s2 * p_s3 * 0.36


    gbdt_s2 = fit_gbdt_stumps(X_tr, ym_tr, n_estimators=35, lr=0.1, min_leaf=30)
    p_stage_gbdt = predict_gbdt(X_va, gbdt_s2) * p_s3 * 0.36



    for name, preds in [
        ("Greedy Soft-Count Heuristic", p_greedy),
        ("Direct Terminal LR (y_msmi)", p_term_lr),
        ("Direct Terminal GBDT (y_msmi)", p_term_gbdt),
        ("Direct Terminal MLP (y_msmi)", p_term_mlp),
        ("Stage-Wise Funnel GBDT", p_stage_gbdt),
        ("Stage-Wise Funnel LR (FEBF)", p_stagewise),
    ]:
        auc_mut = roc_auc(ym_va, preds)
        auc_sec = roc_auc(ys_va, preds)
        auc_msmi = roc_auc(yq_va, preds)
        brier = np.mean((yq_va - np.clip(preds, 0, 1)) ** 2)
        ece = ece_score(yq_va, np.clip(preds, 0, 1))
        thresh = np.percentile(preds, 80)
        lift = yq_va[preds >= thresh].mean() / max(1e-9, yq_va.mean())
        print(f"  {name:30s} | AUC(MutAcc)={auc_mut:.3f} | AUC(2ndPos)={auc_sec:.3f} | AUC(MSMI)={auc_msmi:.3f} | Top20%Lift={lift:.2f}x | ECE={ece:.4f}")
