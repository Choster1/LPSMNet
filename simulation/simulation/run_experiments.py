import argparse
import csv
import os
import time
from multiprocessing import Pool

from error_model import MODELS, matching_model
from simulator import MATCHING, SENSITIVITY, config, make_arrivals, simulate

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
_MODELS = {}
_ARRIVALS = {}


def plan(scenario, n_main, n_sens):
    tasks = []
    for rep in range(n_main):
        tasks.append((scenario, "main", "base", "base", "no_guidance", "-", rep))
        tasks.append((scenario, "main", "base", "base", "perfect", "-", rep))
        for matching in MATCHING:
            for model in MODELS:
                tasks.append((scenario, "main", "base", "base", model, matching, rep))
    if scenario == "evening":
        for factor, levels in SENSITIVITY.items():
            for level in levels:
                for rep in range(n_sens):
                    if factor == "night_factor":
                        tasks.append((scenario, "sens", factor, level, "no_guidance", "-", rep))
                    tasks.append((scenario, "sens", factor, level, "perfect", "-", rep))
                    for matching in MATCHING:
                        for model in MODELS:
                            tasks.append((scenario, "sens", factor, level, model, matching, rep))
    return tasks


def key(task):
    return "|".join(str(x) for x in task)


def run_task(task):
    scenario, study, factor, level, cond, matching, rep = task
    overrides = {} if factor == "base" else {factor: level}
    cfg = config(scenario, "traffic-aware" if matching == "-" else matching, **overrides)
    akey = (scenario, cfg["night_factor"], rep)
    if akey not in _ARRIVALS:
        _ARRIVALS.clear()
        _ARRIVALS[akey] = make_arrivals(1000 + rep, cfg)
    arrivals, shift = _ARRIVALS[akey]
    method = None
    if cond in MODELS:
        if cond not in _MODELS:
            _MODELS[cond] = matching_model(cond)
        method = _MODELS[cond]
    res = simulate(cfg, arrivals, shift, method, seed=50_000 + rep, guidance=(cond != "no_guidance"))
    return dict(key=key(task), scenario=scenario, study=study, factor=factor, level=level, condition=cond,
                matching=matching, rep=rep, **res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["evening", "morning", "all"], default="all")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--sens-reps", type=int, default=20)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=RESULTS_DIR)
    a = ap.parse_args()
    if a.quick:
        a.reps, a.sens_reps = 2, 1
    os.makedirs(a.out, exist_ok=True)
    for scenario in (["evening", "morning"] if a.scenario == "all" else [a.scenario]):
        out_csv = os.path.join(a.out, f"{scenario}{'_quick' if a.quick else ''}.csv")
        done = set()
        if os.path.exists(out_csv):
            with open(out_csv) as f:
                done = {row["key"] for row in csv.DictReader(f)}
        todo = [t for t in plan(scenario, a.reps, a.sens_reps) if key(t) not in done]
        todo.sort(key=lambda t: (t[2], str(t[3]), t[6]))
        print(f"{scenario}: {len(todo)} runs to do ({len(done)} already in {out_csv})", flush=True)
        t_start = time.time()
        with Pool(a.workers) as pool, open(out_csv, "a", newline="") as f:
            writer = None
            for i, row in enumerate(pool.imap_unordered(run_task, todo, chunksize=4) if a.workers > 1 else map(run_task, todo)):
                if writer is None:
                    if os.path.getsize(out_csv) > 0:
                        with open(out_csv) as fh:
                            fields = next(csv.reader(fh))
                        writer = csv.DictWriter(f, fieldnames=fields)
                    else:
                        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                        writer.writeheader()
                writer.writerow(row)
                f.flush()
                if (i + 1) % 20 == 0 or i + 1 == len(todo):
                    el = time.time() - t_start
                    print(f"  {i + 1}/{len(todo)} runs, {el / 60:.1f} min elapsed, "
                          f"~{el / (i + 1) * (len(todo) - i - 1) / 60:.1f} min left", flush=True)


if __name__ == "__main__":
    main()
