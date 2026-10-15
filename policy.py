import argparse
import itertools
import json
import math
import random
import sys
from kit import HARD, SOFT, baseline_asks, baseline_match, eligibility

INFORMATIVE_SOFT = ['relationship_goal', 'relationship_pace', 'conversations', 'lifestyle']
KEYS_4 = ['relationship_goal', 'relationship_pace', 'lifestyle', 'conversations']
N_OPTIONS = {'relationship_goal': 2, 'relationship_pace': 3, 'lifestyle': 3, 'conversations': 4}
W_PRIOR = [0.70, 0.40, 0.25, 0.20]

R_GRID = [0.55 + i * 0.01 for i in range(44)]
ER2_CACHE = {}
for _n in range(25):
    for _y in range(_n + 1):
        _lik = [(r ** _y) * ((1.0 - r) ** (_n - _y)) for r in R_GRID]
        _s = sum(_lik)
        ER2_CACHE[(_y, _n)] = sum((l / _s) * (r * r) for l, r in zip(_lik, R_GRID))



def solve_4x4(A, b):
    """Gaussian elimination with partial pivoting for 4x4 linear system."""
    M = [list(A[i]) + [b[i]] for i in range(4)]
    for i in range(4):
        piv = max(range(i, 4), key=lambda r: abs(M[r][i]))
        M[i], M[piv] = M[piv], M[i]
        div = M[i][i] if abs(M[i][i]) > 1e-12 else 1e-12
        for j in range(i, 5):
            M[i][j] /= div
        for r in range(4):
            if r != i:
                factor = M[r][i]
                for j in range(i, 5):
                    M[r][j] -= factor * M[i][j]
    return [M[i][4] for i in range(4)]



def is_hard_complete(m):
    f = m['fields']
    return all(f.get(k) is not None for k in HARD)

def has_declined_hard(m):
    fs = m['field_status']
    return any(fs.get(k) == 'declined' for k in HARD)



def extract_x_vec(ma, mb):
    fa, fb = ma['fields'], mb['fields']
    x = []
    for k in KEYS_4:
        va, vb = fa.get(k), fb.get(k)
        if va is not None and vb is not None:
            x.append(1.0 if va == vb else -0.5)
        else:
            x.append(1.5 / N_OPTIONS[k] - 0.5)
    return x



def update_online_model(state, memory):
    members_by_id = {m['member_id']: m for m in state['members']}
    intros_by_id = {i['introduction_id']: i for i in state['introductions']}
    processed_events = set(memory.get('processed_events', []))
    member_stats = memory.get('member_stats', {})
    obs_X = memory.get('obs_X', [])
    obs_y = memory.get('obs_y', [])
    w_curr = memory.get('w_curr', list(W_PRIOR))

    new_obs = False
    for ev in state.get('feedback', []):
        if ev.get('event') != 'introduction_response':
            continue
        ev_key = f"{ev['introduction_id']}:{ev['member_id']}"
        if ev_key in processed_events:
            continue
        processed_events.add(ev_key)
        act_id = ev['member_id']
        intro = intros_by_id.get(ev['introduction_id'])
        if not intro:
            continue
        oth_id = intro['user_b'] if intro['user_a'] == act_id else intro['user_a']
        ma, mb = members_by_id.get(act_id), members_by_id.get(oth_id)
        if not ma or not mb:
            continue
        x_vec = extract_x_vec(ma, mb)
        st = member_stats.setdefault(act_id, {'resp_n': 0, 'resp_y': 0, 'acc_n': 0, 'acc_y': 0, 'acc_res': 0.0})
        st['resp_n'] += 1
        resp_b = (ev.get('value') is not None)
        if resp_b:
            st['resp_y'] += 1
            st['acc_n'] += 1
            acc_b = (ev.get('value') == 'yes')
            if acc_b:
                st['acc_y'] += 1
            fit_p = sum(w * x for w, x in zip(w_curr, x_vec))
            pred_p = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, -0.25 + fit_p))))
            st['acc_res'] += (1.0 if acc_b else 0.0) - pred_p
            obs_X.append(x_vec)
            obs_y.append(1.0 if acc_b else 0.0)
            new_obs = True

    if new_obs and len(obs_y) >= 15:
        wt = list(W_PRIOR)
        b0 = -0.25
        for _ in range(3):
            grad = [12.0 * (wt[j] - W_PRIOR[j]) for j in range(4)]
            H = [[12.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
            g_b0 = 0.0
            w_b0 = 5.0
            for x_row, y_val in zip(obs_X, obs_y):
                z = max(-20.0, min(20.0, b0 + sum(wt[j] * x_row[j] for j in range(4))))
                p = 1.0 / (1.0 + math.exp(-z))
                w_i = p * (1.0 - p) + 1e-5
                err = p - y_val
                g_b0 += err
                w_b0 += w_i
                for j in range(4):
                    grad[j] += err * x_row[j]
                    for k in range(4):
                        H[j][k] += w_i * x_row[j] * x_row[k]
            step = solve_4x4(H, grad)
            for j in range(4):
                wt[j] -= step[j]
            b0 -= g_b0 / w_b0
        w_curr = wt

    memory['processed_events'] = sorted(processed_events)
    memory['member_stats'] = member_stats
    memory['obs_X'] = obs_X[-400:]
    memory['obs_y'] = obs_y[-400:]
    memory['w_curr'] = w_curr
    return memory



def febf_asks(state, memory):
    avail = [m for m in state['members'] if m['available']]
    avail_set = {m['member_id'] for m in avail}
    budget = state.get('ask_budget_remaining', 12)
    asks = []

    # Tier 1: FIFO hard-constraint bundles on available members with missing, non-declined hard fields
    inc_avail = [m for m in avail if not is_hard_complete(m) and not has_declined_hard(m)]
    for m in inc_avail[: (budget // 3)]:
        asks.append({'member_id': m['member_id'], 'field': 'constraints'})
        budget -= 3
    if budget > 0:
        past = {tuple(sorted((i['user_a'], i['user_b']))) for i in state['introductions']}
        hc_all = [m for m in state['members'] if is_hard_complete(m)]
        deg_active = {}
        for i in range(len(hc_all)):
            a = hc_all[i]
            for j in range(i + 1, len(hc_all)):
                b = hc_all[j]
                pair = tuple(sorted((a['member_id'], b['member_id'])))
                if pair in past:
                    continue
                if eligibility(a, b)['status'] == 'feasible':
                    w_edge = 3 if (a['member_id'] in avail_set and b['member_id'] in avail_set) else 1
                    deg_active[a['member_id']] = deg_active.get(a['member_id'], 0) + w_edge
                    deg_active[b['member_id']] = deg_active.get(b['member_id'], 0) + w_edge

        hc_cands = [m for m in avail if is_hard_complete(m) and deg_active.get(m['member_id'], 0) > 0]
        hc_cands.sort(key=lambda m: (-deg_active.get(m['member_id'], 0), m['member_id']))
        seen_asks = set()
        for k in INFORMATIVE_SOFT:
            if budget <= 0:
                break
            for m in hc_cands:
                if budget <= 0:
                    break
                m_id = m['member_id']
                if m['fields'].get(k) is None and m['field_status'].get(k) == 'not_asked' and (m_id, k) not in seen_asks:
                    asks.append({'member_id': m_id, 'field': k})
                    seen_asks.add((m_id, k))
                    budget -= 1
    return asks



def febf_match(state, memory, use_scarcity=False):
    memory = update_online_model(state, memory)
    avail_hc = [m for m in state['members'] if m['available'] and is_hard_complete(m)]
    past = {tuple(sorted((i['user_a'], i['user_b']))) for i in state['introductions']}
    day = state['day']

    zones_seen = {m['zone'] for m in state['members'] if m['arrived_day'] == 0}
    is_sparse_pool = len(zones_seen) > 4
    tau = -999.0 if is_sparse_pool else (0.006 * (1.0 - day / 35.0) if day < 35 else -999.0)

    active_edges = []
    active_deg = {}
    for a, b in itertools.combinations(avail_hc, 2):
        pair = tuple(sorted((a['member_id'], b['member_id'])))
        if pair in past or eligibility(a, b)['status'] != 'feasible':
            continue
        active_edges.append((a, b, pair))
        active_deg[pair[0]] = active_deg.get(pair[0], 0) + 1
        active_deg[pair[1]] = active_deg.get(pair[1], 0) + 1

    w_curr = memory.get('w_curr', W_PRIOR)
    member_stats = memory.get('member_stats', {})
    scored_edges = []
    for a, b, pair in active_edges:
        a_id, b_id = pair
        ma, mb = (a, b) if a['member_id'] == a_id else (b, a)
        x_vec = extract_x_vec(ma, mb)
        fit_hat = sum(w * x for w, x in zip(w_curr, x_vec))
        ga, gb = ma['fields'].get('relationship_goal'), mb['fields'].get('relationship_goal')
        goal_sec = (0.40 if ga == gb else 0.0) if (ga is not None and gb is not None) else 0.20

        sa = member_stats.get(a_id, {'resp_n': 0, 'resp_y': 0, 'acc_n': 0, 'acc_res': 0.0})
        sb = member_stats.get(b_id, {'resp_n': 0, 'resp_y': 0, 'acc_n': 0, 'acc_res': 0.0})
        ra2 = ER2_CACHE.get((sa['resp_y'], sa['resp_n']), 0.6014)
        rb2 = ER2_CACHE.get((sb['resp_y'], sb['resp_n']), 0.6014)
        ba = sa['acc_res'] / (2.0408 + 0.23 * sa['acc_n'])
        bb = sb['acc_res'] / (2.0408 + 0.23 * sb['acc_n'])

        p_acc_a = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, -0.25 + ba + fit_hat))))
        p_acc_b = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, -0.25 + bb + fit_hat))))
        p_sec_a = 1.0 / (1.0 + math.exp(-(0.15 + goal_sec)))
        p_sec_b = 1.0 / (1.0 + math.exp(-(0.15 + goal_sec)))
        p_msmi = ra2 * rb2 * p_acc_a * p_acc_b * 0.78 * p_sec_a * p_sec_b * 0.36
        sc = p_msmi * (1.0 + 0.03 / math.sqrt(max(1, active_deg[a_id] * active_deg[b_id]))) if use_scarcity else p_msmi
        if sc >= tau:
            scored_edges.append((-sc, pair))

    scored_edges.sort()
    used = set()
    out = []
    for _, pair in scored_edges:
        if not used.intersection(pair):
            out.append(list(pair))
            used.update(pair)
    return out, memory



def decide(request, mode='febf'):
    state = request.get('observation') or request.get('state')
    memory = request.get('memory') or {}
    phase = request['phase']

    if phase == 'ask':
        if mode == 'no_asks':
            asks = []
        elif mode in ('febf', 'febf_scarcity'):
            asks = febf_asks(state, memory)
        else:
            asks = baseline_asks(state)
        return {'asks': asks, 'memory': memory}

    if mode == 'febf':
        pairs, memory = febf_match(state, memory, use_scarcity=False)
    elif mode == 'febf_scarcity':
        pairs, memory = febf_match(state, memory, use_scarcity=True)
    elif mode == 'simple_fifo':
        avail_hc = [m for m in state['members'] if m['available'] and is_hard_complete(m)]
        past = {tuple(sorted((i['user_a'], i['user_b']))) for i in state['introductions']}
        edges = [
            tuple(sorted((a['member_id'], b['member_id'])))
            for a, b in itertools.combinations(avail_hc, 2)
            if tuple(sorted((a['member_id'], b['member_id']))) not in past
            and eligibility(a, b)['status'] == 'feasible'
        ]
        edges.sort()
        pairs, used = [], set()
        for pair in edges:
            if not used.intersection(pair):
                pairs.append(list(pair))
                used.update(pair)
    elif mode == 'random':
        candidates = [m for m in state['members'] if m['available'] and is_hard_complete(m)]
        past = {tuple(sorted((i['user_a'], i['user_b']))) for i in state['introductions']}
        edges = [
            tuple(sorted((a['member_id'], b['member_id'])))
            for a, b in itertools.combinations(candidates, 2)
            if tuple(sorted((a['member_id'], b['member_id']))) not in past
            and eligibility(a, b)['status'] == 'feasible'
        ]
        random.Random(17 + state['day']).shuffle(edges)
        pairs, used = [], set()
        for pair in edges:
            if not used.intersection(pair):
                pairs.append(list(pair))
                used.update(pair)
    else:
        pairs = baseline_match(state)
    return {'pairs': pairs, 'memory': memory}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', choices=['febf', 'febf_scarcity', 'greedy', 'simple_fifo', 'no_asks', 'random'], default='febf')
    args = parser.parse_args()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        print(json.dumps(decide(req, args.baseline), allow_nan=False), flush=True)
