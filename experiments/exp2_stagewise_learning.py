import sys, pickle, math
import numpy as np

sys.path.insert(0, "/working_dir/c_3af6eefb10334df3/repo")
import kit

with open("/working_dir/c_3af6eefb10334df3/repo/cache/worlds.pkl", "rb") as f:
    CACHE = pickle.load(f)

VARIANTS = ("development", "sparse", "cold_start", "delayed", "shift", "drift")



def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))



def fit_logistic(X, y, l2=1.0, max_iter=30):
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
        if np.max(np.abs(step)) < 1e-6:
            break
    return w



X_acc, y_acc = [], []
X_sec, y_sec = [], []



for s in range(101, 106):
    for v in VARIANTS:
        w, feasible_pairs = CACHE[(s, v)]
        ms = [dict(m, fields=dict(m["fields"]), field_status=dict(m["field_status"])) for m in w["members"]]
        members = {m["member_id"]: m for m in ms}
        hard_complete = {m_id: all(m["fields"][k] is not None for k in kit.HARD) for m_id, m in members.items()}
        has_declined = {m_id: any(m["field_status"][k] == "declined" for k in kit.HARD) for m_id, m in members.items()}
        busy = {}; paused_day = {}; past = set()
        for day in range(60):
            avail = [m for m in ms if m["arrived_day"] <= day < m["exit_day"] and paused_day.get(m["member_id"], 100000) > day and busy.get(m["member_id"], -1) <= day]
            budget = 12
            ask_m = [m for m in avail if not hard_complete[m["member_id"]] and not has_declined[m["member_id"]]][:4]
            budget -= 3 * len(ask_m)
            for m in ask_m:
                for k in kit.HARD: m["fields"][k] = m["truth"][k]
                hard_complete[m["member_id"]] = True
            if budget > 0:
                avail_hc = [m for m in avail if hard_complete[m["member_id"]]]
                for m in avail_hc:
                    if budget <= 0: break
                    for k in kit.SOFT:
                        if budget <= 0: break
                        if m["fields"][k] is None and m["field_status"][k] == "not_asked":
                            m["fields"][k] = m["truth"][k]
                            m["field_status"][k] = "observed"
                            budget -= 1
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
                # Encode soft feature match: +1 if observed match, -0.5 if observed mismatch, 0 if unobserved
                x_soft = []
                for k in kit.SOFT:
                    va, vb = ma["fields"].get(k), mb["fields"].get(k)
                    if va is not None and vb is not None:
                        x_soft.append(1.0 if va == vb else -0.5)
                    else:
                        x_soft.append(0.0)
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
                    if resp_i:
                        X_acc.append(x_soft)
                        y_acc.append(int(acc_i))
                busy[a_id] = day + 8; busy[b_id] = day + 8
                if all(responded) and all(accept):
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
                            if answer:
                                X_sec.append(x_soft)
                                y_sec.append(int(yes))
                        if all(seconds):
                            retire_day = max(obs_days)
                            paused_day[a_id] = min(paused_day.get(a_id, 100000), retire_day)
                            paused_day[b_id] = min(paused_day.get(b_id, 100000), retire_day)

X_acc = np.array(X_acc); y_acc = np.array(y_acc)
X_sec = np.array(X_sec); y_sec = np.array(y_sec)
w_acc = fit_logistic(X_acc, y_acc, l2=2.0)
w_sec = fit_logistic(X_sec, y_sec, l2=5.0)
print(f"Directed Acceptance samples={len(y_acc)}, positive rate={y_acc.mean():.3f}")
print("Learned Stage-2 (Acceptance) intercept & soft weights:")
print(f"  Intercept: {w_acc[0]:+.3f}")
for k, val in zip(kit.SOFT, w_acc[1:]):
    print(f"  {k:25s}: {val:+.3f}")
print(f"\nDirected Second-Meeting samples={len(y_sec)}, positive rate={y_sec.mean():.3f}")
print("Learned Stage-3 (Second-Meeting) intercept & soft weights:")
print(f"  Intercept: {w_sec[0]:+.3f}")
for k, val in zip(kit.SOFT, w_sec[1:]):
    print(f"  {k:25s}: {val:+.3f}")
