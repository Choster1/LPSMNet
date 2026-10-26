import bisect
import math
import random
from collections import deque

import networkx as nx
import numpy as np

LINKS = {
    "O1a": (0, 1, 1.0, 2, 80, 1900, "access"),
    "O1b": (0, 6, 1.0, 2, 50, 700, "access"),
    "M1": (1, 2, 2.0, 2, 100, 1900, "motorway"),
    "M2": (2, 3, 2.0, 2, 100, 1900, "motorway"),
    "M3": (3, 4, 2.0, 2, 100, 1900, "motorway"),
    "M4": (4, 5, 2.0, 2, 100, 1900, "motorway"),
    "A1": (6, 7, 2.0, 2, 50, 650, "arterial"),
    "A2": (7, 8, 2.0, 2, 50, 650, "arterial"),
    "A3": (8, 9, 2.0, 2, 50, 650, "arterial"),
    "A4": (9, 10, 2.0, 2, 50, 650, "arterial"),
    "A5": (10, 5, 1.0, 2, 50, 700, "arterial"),
    "R1": (2, 7, 1.0, 1, 60, 1500, "ramp"),
    "R2": (7, 2, 1.0, 1, 60, 1500, "ramp"),
    "R3": (4, 9, 1.0, 1, 60, 1500, "ramp"),
    "R4": (9, 4, 1.0, 1, 60, 1500, "ramp"),
    "L1": (13, 7, 1.0, 1, 50, 800, "local"),
    "L2": (8, 12, 1.0, 1, 50, 800, "local"),
    "L3": (11, 8, 1.0, 1, 50, 800, "local"),
}
KJAM = 150.0
ODS = {
    "OD1 long-distance": (0, 5, 3600, 1500),
    "OD2 cross-town": (11, 5, 600, 150),
    "OD3 local": (13, 12, 450, 110),
}

SCENARIOS = {
    "evening": dict(t0=18.0, light_change=19.5, night_after_change=True,
                    incidents=[("M3", 19.0, 19.75, 0.35), ("M2", 23.0, 25.0, 0.30)],
                    demand="falling", ramp=(19.5, 21.0), split=21.0, first_period="peak", tail=0),
    "morning": dict(t0=0.0, light_change=6.5, night_after_change=False,
                    incidents=[("M3", 6.25, 7.0, 0.35), ("M2", 1.0, 3.0, 0.30)],
                    demand="rising", ramp=(5.0, 6.5), split=5.0, first_period="night", tail=4 * 3600),
}

BASE = dict(horizon=8 * 3600, night_factor=1.0, incident_jitter=600, dt_ctrl=60, theta=1 / 120, smooth=0.5,
            compliance=0.7, habit_scale=90.0, route_ratio=2.0, max_routes=4, est_window=180, est_window_max=900)

MATCHING = {
    "traffic-aware": dict(gate="adaptive", use_threshold=True, estimator="median", est_min_samples=5,
                          gate_min_extra=120, gate_factor=2.0),
    "naive": dict(gate="static", gate_static_extra=1800, use_threshold=False, estimator="mean", est_min_samples=1),
}

SENSITIVITY = {"night_factor": [0.6, 1.4], "compliance": [0.4, 1.0], "theta": [1 / 60, 1 / 240], "dt_ctrl": [30, 180]}


def config(scenario, matching="traffic-aware", **overrides):
    cfg = dict(BASE)
    cfg.update(SCENARIOS[scenario])
    cfg.update(MATCHING[matching])
    cfg.update(overrides)
    return cfg


def build_links():
    out = {}
    for k, (u, v, km, lanes, vff, cap, kind) in LINKS.items():
        out[k] = dict(u=u, v=v, tff=int(round(km / vff * 3600)), cap=lanes * cap / 3600.0,
                      storage=max(5, int(km * lanes * KJAM)), kind=kind)
    return out


def enumerate_routes(cfg):
    links = build_links()
    G = nx.DiGraph()
    for k, l in links.items():
        G.add_edge(l["u"], l["v"], name=k, tff=l["tff"])
    routes, nominal = {}, {}
    for od, (o, d, *_r) in ODS.items():
        paths = []
        for p in nx.all_simple_paths(G, o, d, cutoff=10):
            names = [G[a][b]["name"] for a, b in zip(p[:-1], p[1:])]
            paths.append((sum(links[n]["tff"] for n in names), names))
        paths.sort()
        tmin = paths[0][0]
        keep = [p for p in paths if p[0] <= cfg["route_ratio"] * tmin][:cfg["max_routes"]]
        routes[od] = [p[1] for p in keep]
        w = [math.exp(-(p[0] - tmin) / cfg["habit_scale"]) for p in keep]
        nominal[od] = [x / sum(w) for x in w]
    return routes, nominal


def demand_rate(od, t, cfg):
    peak, night = ODS[od][2], ODS[od][3] * cfg["night_factor"]
    h = t / 3600.0 + cfg["t0"]
    a, b = cfg["ramp"]
    if cfg["demand"] == "falling":
        r = peak if h < a else (peak + (h - a) / (b - a) * (night - peak) if h < b else night)
    else:
        r = night if h < a else (night + (h - a) / (b - a) * (peak - night) if h < b else peak)
    return r / 3600.0


def make_arrivals(seed, cfg):
    rng = random.Random(seed)
    arr = []
    for od in ODS:
        rmax = max(ODS[od][2], ODS[od][3] * cfg["night_factor"]) / 3600
        t = 0.0
        while True:
            t += rng.expovariate(rmax)
            if t >= cfg["horizon"]:
                break
            if rng.random() < demand_rate(od, t, cfg) / rmax:
                arr.append((t, od, rng.random(), rng.random()))
    arr.sort()
    shift = rng.uniform(-cfg["incident_jitter"], cfg["incident_jitter"])
    return arr, shift


def simulate(cfg, arrivals, inc_shift, method, seed, guidance=True):
    rng = random.Random(seed)
    links = build_links()
    routes, nominal = enumerate_routes(cfg)
    names = list(links)
    tff = {k: links[k]["tff"] for k in names}
    q = {k: deque() for k in names}
    credit = {k: 0.0 for k in names}
    t0s = cfg["t0"] * 3600
    inc_by_link = {}
    for i, (l, a, b, f) in enumerate(cfg["incidents"]):
        s = inc_shift if i == 0 else 0.0
        inc_by_link.setdefault(l, []).append((a * 3600 - t0s + s, b * 3600 - t0s + s, f))

    nodes = sorted({x for l in links.values() for x in (l["u"], l["v"])})
    v_od, v_route, v_pos, v_dep, v_arr = [], [], [], [], []
    det_t = {n: [] for n in nodes}
    det_at = {}
    link_exits = {k: [] for k in names}
    processed = {k: 0 for k in names}
    samples = {k: [] for k in names}
    est = {k: float(tff[k]) for k in names}
    match_log = []
    shares = {od: list(nominal[od]) for od in ODS}
    buffers = {od: deque() for od in ODS}
    change = cfg["light_change"] * 3600 - t0s
    night_after = cfg["night_after_change"]
    agg = np.median if cfg["estimator"] == "median" else np.mean

    def is_night(t):
        return (t >= change) if night_after else (t < change)

    def cap(k, t):
        c = links[k]["cap"]
        for a, b, f in inc_by_link.get(k, ()):
            if a <= t < b:
                c *= f
        return c

    def detect(vid, node, t):
        det_at[(vid, node)] = len(det_t[node])
        det_t[node].append(t)

    def reid_link(k, now):
        u = links[k]["u"]
        ex, ts = link_exits[k], det_t[u]
        while processed[k] < len(ex) and ex[processed[k]][0] <= now:
            tv, vid = ex[processed[k]]
            processed[k] += 1
            j = det_at[(vid, u)]
            true_tt = tv - ts[j]
            if method is None:
                samples[k].append((tv, true_tt, true_tt))
                continue
            hi = tv - tff[k] + 1e-6
            if cfg["gate"] == "static":
                lo = tv - tff[k] - cfg["gate_static_extra"]
            else:
                lo = tv - max(tff[k] + cfg["gate_min_extra"], cfg["gate_factor"] * est[k])
            i0, i1 = bisect.bisect_left(ts, lo), bisect.bisect_right(ts, hi)
            true_in = i0 <= j < i1
            m = (i1 - i0) - int(true_in)
            qn, gn = is_night(tv), is_night(ts[j])
            pair = "NN" if (qn and gn) else ("DD" if not (qn or gn) else ("DN" if gn else "ND"))
            mp = method[pair]
            gl = mp["gen"]
            g = (gl[rng.randrange(len(gl))] + rng.gauss(0.0, mp["h"])) if true_in else math.inf
            if m > 0:
                qmin = -math.expm1(math.log(rng.random() or 1e-300) / m)
                imp = float(np.interp(qmin, mp["cdf"], mp["grid"]))
            else:
                imp = math.inf
            if cfg["use_threshold"] and min(g, imp) > mp["thr"]:
                match_log.append((tv, 2, m))
                continue
            if g <= imp:
                samples[k].append((tv, true_tt, true_tt))
                match_log.append((tv, 0, m))
            else:
                jj = i0 + rng.randrange(m)
                if true_in and jj >= j:
                    jj += 1
                samples[k].append((tv, tv - ts[jj], true_tt))
                match_log.append((tv, 1, m))

    def estimate(k, now):
        s, W = samples[k], cfg["est_window"]
        while True:
            vals = []
            for x in reversed(s):
                if x[0] <= now - W:
                    break
                vals.append(x[1])
            if len(vals) >= cfg["est_min_samples"] or W >= cfg["est_window_max"]:
                break
            W = min(W * 2, cfg["est_window_max"])
        if vals:
            est[k] = float(agg(vals))

    def control():
        a = cfg["smooth"]
        for od, rs in routes.items():
            if len(rs) == 1:
                continue
            T = [sum(est[k] for k in r) for r in rs]
            tmin = min(T)
            w = [n * math.exp(-cfg["theta"] * (x - tmin)) for n, x in zip(nominal[od], T)]
            s = sum(w)
            shares[od] = [(1 - a) * o + a * x / s for o, x in zip(shares[od], w)]

    def pick(probs, u):
        c = 0.0
        for i, p in enumerate(probs):
            c += p
            if u < c:
                return i
        return len(probs) - 1

    ai, n_arr = 0, len(arrivals)
    order = names[:]
    dt_ctrl = int(cfg["dt_ctrl"])
    for t in range(cfg["horizon"] + 4 * 3600):
        while ai < n_arr and arrivals[ai][0] <= t:
            ta, od, ur, uc = arrivals[ai]
            ai += 1
            follow = guidance and uc < cfg["compliance"]
            r = pick(shares[od] if follow else nominal[od], ur)
            vid = len(v_od)
            v_od.append(od); v_route.append(routes[od][r]); v_pos.append(0)
            v_dep.append(ta); v_arr.append(None)
            buffers[od].append(vid)
        for buf in buffers.values():
            while buf:
                vid = buf[0]
                k0 = v_route[vid][0]
                if len(q[k0]) >= links[k0]["storage"]:
                    break
                buf.popleft()
                detect(vid, links[k0]["u"], t)
                q[k0].append((t + tff[k0], vid))
        rng.shuffle(order)
        for k in order:
            qk = q[k]
            c = cap(k, t)
            credit[k] = min(credit[k] + c, c + 1.0)
            if not qk:
                continue
            while credit[k] >= 1.0 and qk and qk[0][0] <= t:
                vid = qk[0][1]
                route, pos = v_route[vid], v_pos[vid]
                if pos + 1 < len(route):
                    nk = route[pos + 1]
                    if len(q[nk]) >= links[nk]["storage"]:
                        break
                    qk.popleft()
                    v_pos[vid] = pos + 1
                    q[nk].append((t + tff[nk], vid))
                else:
                    qk.popleft()
                    v_arr[vid] = t
                credit[k] -= 1.0
                detect(vid, links[k]["v"], t)
                link_exits[k].append((t, vid))
        if t > 0 and t % dt_ctrl == 0 and t <= cfg["horizon"]:
            for k in names:
                reid_link(k, t)
                estimate(k, t)
            if guidance:
                control()
        if t >= cfg["horizon"] and ai >= n_arr and not any(buffers.values()) and not any(q.values()):
            break

    split = cfg["split"] * 3600 - t0s
    first, second = cfg["first_period"], ("night" if cfg["first_period"] == "peak" else "peak")
    out = {"TTS_all": 0.0, "TTS_peak": 0.0, "TTS_night": 0.0}
    for hb in range(8):
        out[f"TTS_h{hb}"] = 0.0
    for od in ODS:
        out[f"TTS_{od.split()[0]}"] = 0.0
    for vid in range(len(v_od)):
        if v_arr[vid] is None:
            continue
        tt = (v_arr[vid] - v_dep[vid]) / 3600.0
        out["TTS_all"] += tt
        out[f"TTS_h{min(int(v_dep[vid] // 3600), 7)}"] += tt
        out[f"TTS_{first}" if v_dep[vid] < split else f"TTS_{second}"] += tt
        out[f"TTS_{v_od[vid].split()[0]}"] += tt
    out["unfinished"] = sum(a is None for a in v_arr)
    for hb in range(8):
        ml = [mm for mm in match_log if hb * 3600 <= mm[0] < (hb + 1) * 3600]
        out[f"wrong_h{hb}"] = sum(mm[1] == 1 for mm in ml) / max(len(ml), 1)
    for per, (a, b) in {first: (0, split), second: (split, cfg["horizon"] + cfg["tail"])}.items():
        ml = [mm for mm in match_log if a <= mm[0] < b]
        n = max(len(ml), 1)
        out[f"wrong_{per}"] = sum(mm[1] == 1 for mm in ml) / n
        out[f"reject_{per}"] = sum(mm[1] == 2 for mm in ml) / n
        out[f"pool_{per}"] = float(np.mean([mm[2] for mm in ml])) if ml else 0.0
        err = [s_[1] - s_[2] for k in names for s_ in samples[k] if a <= s_[0] < b]
        out[f"tt_mae_{per}"] = float(np.mean(np.abs(err))) if err else 0.0
    return out
