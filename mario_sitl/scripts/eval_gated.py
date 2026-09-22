#!/usr/bin/env python
"""Compare the gated fusion against the control, per flight regime, and read the gate.

Three questions, in the order they can falsify the idea:

  1. do the headline numbers move at all (seen/unseen ATE, TDE, 4 seeds);
  2. does the error drop specifically in steady cruise, where the accelerometer carries
     no speed information but the drag-induced lean angle does -- and not in the
     acceleration bins, where nothing was predicted;
  3. does the gate actually do what the story says, i.e. lean on the IMU when |a| is
     large and on attitude when it is small. A gate pinned near a constant means the
     layer learned nothing and the first two answers are about something else.

Regime bins follow the brief: below 0.07 m/s is under Blackbird's observability floor
(APPLICATION_VALIDATION.md 3.4), and |a| >= 1 m/s^2 is the "accelerating" bin used in
diagnose_observability.py. Speed and acceleration come from differentiating the Leica
ground-truth position, never from the IMU, so the bins do not depend on the input the
models are being judged on.

Significance is paired across seeds (n=4, scipy ttest_rel): the same four seeds train
both arms, and this project has already had single-seed results reverse.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402
from scipy import stats                               # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mario_sitl" / "scripts"))

from mario.config import Config                       # noqa: E402
from mario.data import load_eval_sequences            # noqa: E402
from mario.model import CausalMambaDispNet            # noqa: E402
from mario.model_gated import GatedMambaDispNet       # noqa: E402
from eval_full_rollout import rollout_full, metrics   # noqa: E402

RESULTS = ROOT / "mario_sitl" / "results" / "gated"


def eval_pairs(dataset: str, cfg, verbose: bool = False) -> dict:
    """Sequences to score, as {split: [(name, seq)]}.

    Blackbird is where the models were trained, and 97 % of its evaluation steps are
    accelerating, so the steady-cruise bins the gate hypothesis is about are empty
    there. These other sets are the ones that actually populate them; the models are
    scored on them zero-shot, without any further training.
    """
    if dataset == "blackbird":
        seen, unseen = load_eval_sequences(cfg.data, verbose=verbose)
        return {"seen": seen, "unseen": unseen}
    if dataset == "euroc":
        import euroc_data
        # the alignment the EuRoC study picked on its own training split; both arms get
        # the same one, so it cannot favour either
        return {"euroc_test": euroc_data.load_split(
            euroc_data.read_lists()["test"], dt=cfg.data.dt, align="gravity",
            yaw_deg=185.0, verbose=verbose)}
    if dataset == "uzhfpv":
        import uzhfpv_data
        return {"uzhfpv_test": uzhfpv_data.load_split(
            uzhfpv_data.read_lists()["test"], dt=cfg.data.dt, align="yaw",
            yaw_deg=310.0, verbose=verbose)}
    if dataset == "sitl1":
        # the first x500 collection: setpoints walked at constant speed, so only
        # 13-26 % of its samples exceed 1 m/s^2 (FINETUNE_METHOD.md 1.1)
        import numpy as _np
        from eval_openloop import build_sequence
        files = sorted((ROOT / "mario_sitl" / "results" / "dataset").glob("flight_*.npz"))
        return {"sitl1": [(f.stem, build_sequence(dict(_np.load(f)), "gt")) for f in files]}
    raise ValueError(dataset)


ARCHS = {"control": CausalMambaDispNet, "gated": GatedMambaDispNet}
#: (label, speed_lo, speed_hi, accel_lo, accel_hi) in m/s and m/s^2
BINS = [
    ("very_slow", 0.0, 0.07, 0.0, np.inf),
    ("slow_steady", 0.07, 0.5, 0.0, 1.0),
    ("mid_steady", 0.5, 3.0, 0.0, 1.0),
    ("accelerating", 0.0, np.inf, 1.0, np.inf),
]


def load_net(arch: str, ckpt: Path, cfg, dev):
    net = ARCHS[arch](d_state=cfg.model.d_state, d_conv=cfg.model.d_conv,
                      expand=cfg.model.expand, num_layers=cfg.model.num_layers).to(dev)
    net.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=True))
    net.eval()
    return net


def step_table(net, arch: str, seq, cfg, dev):
    """Per predicted step: displacement error, GT speed, GT |accel|, and the gate.

    Windows are laid end to end (stride = window_size) so no step is counted twice.
    """
    d = cfg.data
    acc, gyro, rot = seq["acc"], seq["gyro"], seq["gt_orientation"]
    pos = seq["gt_translation"].numpy()
    vel = np.gradient(pos, d.dt, axis=0)
    accel = np.gradient(vel, d.dt, axis=0)
    n_steps = (d.window_size - d.label_start_index) // d.label_stride - 1

    err, speed, amag, gate = [], [], [], []
    pmag, tmag = [], []
    for start in range(0, len(acc) - d.window_size - d.label_stride, d.window_size):
        sl = slice(start, start + d.window_size)
        with torch.no_grad():
            out = net(acc[sl].unsqueeze(0).to(dev), gyro[sl].unsqueeze(0).to(dev),
                      rot[sl].unsqueeze(0).to(dev).Log().tensor().float(),
                      **({"return_gate": True} if arch == "gated" else {}))
        pred = out[0][0].cpu().numpy()[:n_steps]
        idx = start + d.label_start_index + d.label_stride * np.arange(n_steps)
        # true body-frame displacement over the same step
        world = pos[idx + d.label_stride] - pos[idx]
        true = np.einsum("tji,tj->ti", rot[idx].matrix().numpy(), world)
        err.append(np.linalg.norm(pred - true, axis=1))
        pmag.append(np.linalg.norm(pred, axis=1))
        tmag.append(np.linalg.norm(true, axis=1))
        speed.append(np.linalg.norm(vel[idx], axis=1))
        amag.append(np.linalg.norm(accel[idx], axis=1))
        if arch == "gated":
            gate.append(out[2][0].cpu().numpy()[:n_steps].mean(axis=-1))
    if not err:
        return None
    out = {"err": np.concatenate(err), "speed": np.concatenate(speed),
           "accel": np.concatenate(amag), "pred_mag": np.concatenate(pmag),
           "true_mag": np.concatenate(tmag)}
    if gate:
        out["gate"] = np.concatenate(gate)
    return out


def binned(tab) -> dict:
    res = {}
    for name, s_lo, s_hi, a_lo, a_hi in BINS:
        m = ((tab["speed"] >= s_lo) & (tab["speed"] < s_hi)
             & (tab["accel"] >= a_lo) & (tab["accel"] < a_hi))
        e = tab["err"][m]
        res[name] = {"rmse": float(np.sqrt(np.mean(e ** 2))) if len(e) else float("nan"),
                     "n": int(m.sum()),
                     "pred_mag": float(np.mean(tab["pred_mag"][m])) if len(e) else float("nan"),
                     "true_mag": float(np.mean(tab["true_mag"][m])) if len(e) else float("nan")}
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=ROOT / "runs" / "gated")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3])
    ap.add_argument("--timeline-traj", default="sid",
                    help="unseen trajectory drawn in gate_timeline.png")
    ap.add_argument("--dataset", default="blackbird",
                    choices=("blackbird", "euroc", "uzhfpv", "sitl1"),
                    help="where to score the already-trained models; no training happens")
    a = ap.parse_args()

    cfg = Config.load(str(ROOT / "configs" / "trial8.yaml"))
    cfg.data.data_dir = str(Path(cfg.data.data_dir).expanduser())
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    named = eval_pairs(a.dataset, cfg)
    suffix = "" if a.dataset == "blackbird" else f"_{a.dataset}"

    out = {"bins": {n: {"speed": [s_lo, s_hi], "accel": [a_lo, a_hi]}
                    for n, s_lo, s_hi, a_lo, a_hi in BINS},
           "runs": {}, "per_seed": defaultdict(dict)}
    scatter = {"accel": [], "gate": []}
    timelines: dict = {}

    for arch in ARCHS:
        for seed in a.seeds:
            run = a.runs / f"{arch}_s{seed}"
            ckpt = run / "best.pt"
            if not ckpt.exists():
                print(f"missing {ckpt}")
                continue
            net = load_net(arch, ckpt, cfg, dev)
            meta = json.loads((run / "run_meta.json").read_text())
            res = json.loads((run / "results.json").read_text())

            full, per_bin_err = {}, defaultdict(list)
            for split, pairs in named.items():
                ates = []
                for name, seq in pairs:
                    rolled = rollout_full(net, seq, dev, cfg.data.window_size,
                                          cfg.data.label_start_index, cfg.data.label_stride)
                    if rolled is not None:
                        ates.append(metrics(*rolled))
                    tab = step_table(net, arch, seq, cfg, dev)
                    if tab is None:
                        continue
                    for k, v in binned(tab).items():
                        per_bin_err[k].append((v["rmse"], v["n"], v["pred_mag"], v["true_mag"]))
                    if arch == "gated":
                        scatter["accel"].append(tab["accel"])
                        scatter["gate"].append(tab["gate"])
                        if seed == a.seeds[0]:
                            timelines[name.split("/")[0]] = tab
                full[split] = {
                    "full_ate": float(np.mean([m["ATE"] for m in ates])),
                    "full_tde": float(np.mean([m["TDE"] for m in ates])),
                }

            bins = {}
            for k, vals in per_bin_err.items():
                w = np.array([n for _, n, _, _ in vals], dtype=float)
                r = np.array([x for x, _, _, _ in vals])
                pm = np.array([x for _, _, x, _ in vals])
                tm = np.array([x for _, _, _, x in vals])
                ok = np.isfinite(r) & (w > 0)
                bins[k] = {"rmse": float(np.sqrt(np.sum(w[ok] * r[ok] ** 2) / w[ok].sum()))
                           if ok.any() else float("nan"),
                           "n": int(w.sum()),
                           "pred_mag": float(np.sum(w[ok] * pm[ok]) / w[ok].sum()) if ok.any() else float("nan"),
                           "true_mag": float(np.sum(w[ok] * tm[ok]) / w[ok].sum()) if ok.any() else float("nan")}
            out["runs"][f"{arch}_s{seed}"] = {
                "arch": arch, "seed": seed, "n_params": meta["n_params"],
                "windowed_seen_ate": res["seen"]["mean_ate"],
                "windowed_unseen_ate": res["unseen"]["mean_ate"],
                "windowed_seen_tde": res["seen"]["mean_tde"],
                "windowed_unseen_tde": res["unseen"]["mean_tde"],
                **{f"{k}_{m}": v[m] for k, v in full.items() for m in ("full_ate", "full_tde")},
                "bins": bins,
            }
            print(f"{arch:<8} s{seed:<3} seen {res['seen']['mean_ate']:.4f}  "
                  f"unseen {res['unseen']['mean_ate']:.4f}  "
                  + "  ".join(f"{k} {v['rmse']:.5f}" for k, v in bins.items()))

    # ---- aggregate + paired significance across seeds -------------------------------
    def col(arch, key):
        return np.array([out["runs"][f"{arch}_s{s}"][key] for s in a.seeds
                         if f"{arch}_s{s}" in out["runs"]])

    keys = [f"{split}_{m}" for split in named for m in ("full_ate", "full_tde")]
    if a.dataset == "blackbird":   # the windowed numbers in results.json are Blackbird's
        keys = ["windowed_seen_ate", "windowed_unseen_ate", "windowed_seen_tde",
                "windowed_unseen_tde"] + keys
    summary = {}
    for key in keys:
        c, g = col("control", key), col("gated", key)
        summary[key] = {"control_mean": float(c.mean()), "control_std": float(c.std(ddof=1)),
                        "gated_mean": float(g.mean()), "gated_std": float(g.std(ddof=1)),
                        "delta_pct": float(100 * (g.mean() - c.mean()) / c.mean())}
        if len(c) == len(g) > 1:
            t, p = stats.ttest_rel(g, c)
            summary[key].update(t=float(t), p=float(p))

    bin_summary = {}
    for name, *_ in BINS:
        c = np.array([out["runs"][f"control_s{s}"]["bins"][name]["rmse"] for s in a.seeds
                      if f"control_s{s}" in out["runs"]])
        g = np.array([out["runs"][f"gated_s{s}"]["bins"][name]["rmse"] for s in a.seeds
                      if f"gated_s{s}" in out["runs"]])
        n = out["runs"][f"control_s{a.seeds[0]}"]["bins"][name]["n"]
        pm_c = float(np.mean([out["runs"][f"control_s{s_}"]["bins"][name]["pred_mag"]
                              for s_ in a.seeds if f"control_s{s_}" in out["runs"]]))
        pm_g = float(np.mean([out["runs"][f"gated_s{s_}"]["bins"][name]["pred_mag"]
                              for s_ in a.seeds if f"gated_s{s_}" in out["runs"]]))
        tm = float(np.mean([out["runs"][f"control_s{s_}"]["bins"][name]["true_mag"]
                            for s_ in a.seeds if f"control_s{s_}" in out["runs"]]))
        row = {"n_steps": n, "true_mag": tm,
               "control_pred_mag": pm_c, "gated_pred_mag": pm_g,
               "control_mean": float(np.nanmean(c)), "control_std": float(np.nanstd(c, ddof=1)),
               "gated_mean": float(np.nanmean(g)), "gated_std": float(np.nanstd(g, ddof=1)),
               "delta_pct": float(100 * (np.nanmean(g) - np.nanmean(c)) / np.nanmean(c))}
        if len(c) == len(g) > 1:
            t, p = stats.ttest_rel(g, c)
            row.update(t=float(t), p=float(p))
        bin_summary[name] = row
    out["summary"], out["bin_summary"] = summary, bin_summary
    out["per_seed"] = dict(out["per_seed"])

    # ---- gate diagnostics ------------------------------------------------------------
    figs = RESULTS / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    if scatter["accel"]:
        acc_all = np.concatenate(scatter["accel"])
        g_all = np.concatenate(scatter["gate"])
        r, p = stats.pearsonr(acc_all, g_all)
        out["gate"] = {"mean": float(g_all.mean()), "std": float(g_all.std()),
                       "min": float(g_all.min()), "max": float(g_all.max()),
                       "pearson_r_vs_accel": float(r), "pearson_p": float(p),
                       "n": int(len(g_all))}
        print(f"\ngate: mean {g_all.mean():.3f} +- {g_all.std():.3f} "
              f"[{g_all.min():.3f}, {g_all.max():.3f}]   r(|a|, g) = {r:+.3f} (p={p:.2g})")

        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        idx = np.random.default_rng(0).choice(len(g_all), min(20000, len(g_all)), replace=False)
        ax.scatter(acc_all[idx], g_all[idx], s=2, alpha=0.12, color="#2b6cb0", linewidths=0)
        edges = np.quantile(acc_all, np.linspace(0, 1, 13))
        mids = 0.5 * (edges[1:] + edges[:-1])
        med = [np.median(g_all[(acc_all >= lo) & (acc_all < hi)])
               for lo, hi in zip(edges[:-1], edges[1:])]
        ax.plot(mids, med, color="#c0392b", lw=1.8, marker="o", ms=3.5,
                label="median per decile")
        ax.set_xlabel("GT |acceleration| [m/s$^2$]", fontsize=9)
        ax.set_ylabel("gate g  (1 = IMU, 0 = attitude)", fontsize=9)
        ax.set_title(f"Gate vs acceleration   r = {r:+.3f}", fontsize=10)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(figs / f"gate_vs_accel{suffix}.png", dpi=200)
        print(f"wrote {figs / f'gate_vs_accel{suffix}.png'}")

    timeline = timelines.get(a.timeline_traj)
    if timeline is None and timelines:
        timeline = max(timelines.values(), key=lambda t: len(t["gate"]))
    tl_name = next((k for k, v in timelines.items() if v is timeline), a.timeline_traj)
    if timeline is not None:
        t = np.arange(len(timeline["gate"])) * cfg.data.label_stride * cfg.data.dt
        fig, ax = plt.subplots(2, 1, figsize=(7.0, 4.0), sharex=True)
        ax[0].plot(t, timeline["accel"], color="#444", lw=1.0)
        ax[0].axhline(1.0, color="#c0392b", ls=":", lw=1.0)
        ax[0].set_ylabel("GT |a| [m/s$^2$]", fontsize=9)
        ax[1].plot(t, timeline["gate"], color="#2b6cb0", lw=1.0)
        ax[1].axhline(float(np.mean(timeline["gate"])), color="#c0392b", ls=":", lw=1.0)
        ax[1].set_ylabel("gate g", fontsize=9)
        ax[1].set_xlabel("Time [s]", fontsize=9)
        for x in ax:
            x.tick_params(labelsize=8)
            x.spines[["top", "right"]].set_visible(False)
        ax[0].set_title(f"{tl_name} ({a.dataset}) — acceleration and gate over time",
                        fontsize=10)
        fig.tight_layout()
        fig.savefig(figs / f"gate_timeline{suffix}.png", dpi=200)
        print(f"wrote {figs / f'gate_timeline{suffix}.png'}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out["dataset"] = a.dataset
    (RESULTS / f"summary{suffix}.json").write_text(json.dumps(out, indent=2))

    print(f"\n{'metric':<22}{'control':>18}{'gated':>18}{'delta':>9}{'p':>9}")
    for k, v in summary.items():
        print(f"{k:<22}{v['control_mean']:>10.4f}±{v['control_std']:<7.4f}"
              f"{v['gated_mean']:>10.4f}±{v['gated_std']:<7.4f}"
              f"{v['delta_pct']:>+8.1f}%{v.get('p', float('nan')):>9.3f}")
    print(f"\n{'regime':<14}{'n steps':>9}{'control RMSE':>20}{'gated RMSE':>20}{'delta':>9}{'p':>9}")
    for k, v in bin_summary.items():
        print(f"{k:<14}{v['n_steps']:>9}{v['control_mean']:>12.5f}±{v['control_std']:<7.5f}"
              f"{v['gated_mean']:>12.5f}±{v['gated_std']:<7.5f}"
              f"{v['delta_pct']:>+8.1f}%{v.get('p', float('nan')):>9.3f}")
    print(f"\nwrote {RESULTS / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
