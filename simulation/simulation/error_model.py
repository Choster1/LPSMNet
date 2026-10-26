import math
import os

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODELS = ("DNDM", "SFM", "LPSMNet")
PAIRS = ("DD", "ND", "NN")
FAR = 1e-3
GALLERY_IDS = 148
_trapz = getattr(np, "trapezoid", None) or np.trapz


class KDE1D:
    def __init__(self, x, shift=0.0, ngrid=6000):
        self.x = np.asarray(x, float) + shift
        n = len(self.x)
        sd = self.x.std(ddof=1)
        iqr = np.subtract(*np.percentile(self.x, [75, 25]))
        self.h = 0.9 * min(sd, iqr / 1.34) * n ** (-0.2)
        lo, hi = self.x.min() - 6 * self.h, self.x.max() + 6 * self.h
        self.grid = np.linspace(lo, hi, ngrid)
        dx = self.grid[1] - self.grid[0]
        counts, _ = np.histogram(self.x, bins=np.append(self.grid - dx / 2, self.grid[-1] + dx / 2))
        k = np.arange(-int(6 * self.h / dx) - 1, int(6 * self.h / dx) + 2) * dx
        ker = np.exp(-0.5 * (k / self.h) ** 2)
        ker /= ker.sum()
        pdf = np.convolve(counts / n, ker, mode="same")
        self.pdf = pdf / dx
        self.cdf = np.clip(np.cumsum(pdf), 0, 1)

    def ppf(self, q):
        return np.interp(q, self.cdf, self.grid)


def distance_file(model):
    return os.path.join(DATA_DIR, f"{model}_distances.csv")


def load_distances(model):
    df = pd.read_csv(distance_file(model))
    mi, mj = df.modality_i, df.modality_j
    pair = np.where((mi == "day") & (mj == "day"), "DD", np.where((mi == "night") & (mj == "night"), "NN", "ND"))
    same = (df.label == "same").values
    out = {}
    for p in PAIRS:
        sel = pair == p
        out[p] = (df.distance.values[sel & same], df.distance.values[sel & ~same])
    return out


def matching_model(model):
    spec = {}
    for p, (g, i) in load_distances(model).items():
        gen, imp = KDE1D(g), KDE1D(i)
        spec[p] = dict(gen=gen.x.tolist(), h=gen.h, grid=imp.grid, cdf=imp.cdf, thr=float(imp.ppf(FAR)))
    spec["DN"] = spec["ND"]
    return spec


def p_correct(gen, imp, m):
    surv = np.clip(1 - np.interp(gen.grid, imp.grid, imp.cdf, left=0, right=1), 0, 1)
    return float(_trapz(gen.pdf * surv ** m, gen.grid))


def consistency_table():
    rows = []
    for model in MODELS:
        d = load_distances(model)
        g_all = np.concatenate([d[p][0] for p in PAIRS])
        i_all = np.concatenate([d[p][1] for p in PAIRS])
        ts = np.linspace(min(g_all.min(), i_all.min()), max(g_all.max(), i_all.max()), 3000)
        fnr = np.array([(g_all > t).mean() for t in ts])
        fpr = np.array([(i_all <= t).mean() for t in ts])
        thr = float(ts[np.argmin(np.abs(fnr - fpr))])
        for p in PAIRS:
            g, i = d[p]
            gk, ik = KDE1D(g), KDE1D(i)
            rows.append(dict(model=model, pair=p, n_same=len(g), n_diff=len(i),
                             d_prime=(i.mean() - g.mean()) / math.sqrt(0.5 * (i.var(ddof=1) + g.var(ddof=1))),
                             rank1_147=p_correct(gk, ik, GALLERY_IDS - 1), p_correct_1000=p_correct(gk, ik, 1000),
                             fpr_single_threshold=float((i <= thr).mean()), threshold_far_0p1=float(ik.ppf(FAR))))
    return pd.DataFrame(rows)
