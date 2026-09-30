#!/usr/bin/env python
"""When is MARIO accurate, and when does it fail? Step-level error against flight state.

The published numbers are per-trajectory ATE, which averages over everything the drone
was doing. This resolves the same runs by flight condition: every predicted step is
tagged with what the airframe was doing at that instant, and the error is read against
each condition separately.

Two error definitions, because they answer different questions:

  absolute   ||pred - true||, in metres per 90 ms step. What the integrator accumulates.
  relative   ||pred - true|| / ||true||. Fast flight has larger displacements, so the
             absolute error grows with speed almost by construction; the relative form
             asks whether the estimate is proportionally worse.

Conditions are computed from ground truth (position differentiated twice) and from the
IMU only where the quantity IS the IMU reading (gyro magnitude, the drag proxy). Speed
and acceleration never come from the input being judged.

Seeds are kept separate and reported as mean +- std: a single-seed ranking in this
project has reversed three times (PAPER_NOTES.md 7).
"""
from __future__ import annotations

import argparse
import json
import sys
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
from mario.model import CausalMambaDispNet            # noqa: E402
from eval_full_rollout import rollout_full, metrics   # noqa: E402
from eval_gated import eval_pairs                     # noqa: E402

RESULTS = ROOT / "mario_sitl" / "results" / "regimes"
GRAVITY = 9.80665

#: (key, label, unit) — every condition the steps are resolved against
FACTORS = [
    ("speed", "Speed", "m/s"),
    ("accel", "|Acceleration|", "m/s$^2$"),
    ("gyro", "|Angular rate|", "rad/s"),
    ("yaw_rate", "|Yaw rate|", "rad/s"),
    ("climb", "|Vertical speed|", "m/s"),
    ("drag", "Drag signal", "m/s$^2$"),
]


def step_table(net, seq, cfg, dev) -> dict:
    """One row per predicted step: error plus the flight condition at that instant."""
    d = cfg.data
    acc, gyro, rot = seq["acc"], seq["gyro"], seq["gt_orientation"]
    pos = seq["gt_translation"].numpy()
    vel = np.gradient(pos, d.dt, axis=0)
    accel = np.gradient(vel, d.dt, axis=0)
    gyro_np = gyro.numpy()
    acc_np = acc.numpy()
    n_steps = (d.window_size - d.label_start_index) // d.label_stride - 1

    cols: dict[str, list] = {k: [] for k in
                             ("err", "rel", "true_mag", "pred_mag",
                              "speed", "accel", "gyro", "yaw_rate", "climb", "drag")}
    for start in range(0, len(acc) - d.window_size - d.label_stride, d.window_size):
        sl = slice(start, start + d.window_size)
        with torch.no_grad():
            pred, _ = net(acc[sl].unsqueeze(0).to(dev), gyro[sl].unsqueeze(0).to(dev),
                          rot[sl].unsqueeze(0).to(dev).Log().tensor().float())
        p = pred[0].cpu().numpy()[:n_steps]
        idx = start + d.label_start_index + d.label_stride * np.arange(n_steps)
        world = pos[idx + d.label_stride] - pos[idx]
        true = np.einsum("tji,tj->ti", rot[idx].matrix().numpy(), world)

        e = np.linalg.norm(p - true, axis=1)
        tm = np.linalg.norm(true, axis=1)
        # the condition is averaged over the step the prediction covers, not sampled at
        # its first instant -- a 90 ms step can straddle the start of a turn
        win = np.stack([np.arange(i, i + d.label_stride) for i in idx])
        cols["err"].append(e)
        cols["rel"].append(e / np.maximum(tm, 1e-6))
        cols["true_mag"].append(tm)
        cols["pred_mag"].append(np.linalg.norm(p, axis=1))
        cols["speed"].append(np.linalg.norm(vel[win], axis=-1).mean(1))
        cols["accel"].append(np.linalg.norm(accel[win], axis=-1).mean(1))
        cols["gyro"].append(np.linalg.norm(gyro_np[win], axis=-1).mean(1))
        cols["yaw_rate"].append(np.abs(gyro_np[win][..., 2]).mean(1))
        cols["climb"].append(np.abs(vel[win][..., 2]).mean(1))
        # horizontal specific force: at steady speed this is the drag term, the only
        # channel that carries velocity when acceleration is absent
        cols["drag"].append(np.linalg.norm(acc_np[win][..., :2], axis=-1).mean(1))
    if not cols["err"]:
        return {}
    return {k: np.concatenate(v) for k, v in cols.items()}


def deciles(tab: dict, key: str, n_bins: int = 10) -> list:
    """Error inside each decile of one condition, so bins hold equal counts."""
    x = tab[key]
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x <= hi if hi == edges[-1] else x < hi)
        if m.sum() < 10:
            continue
        rows.append({"lo": float(lo), "hi": float(hi), "mid": float(np.median(x[m])),
                     "n": int(m.sum()),
                     "rmse": float(np.sqrt(np.mean(tab["err"][m] ** 2))),
                     "rel_median": float(np.median(tab["rel"][m])),
                     "rel_q25": float(np.quantile(tab["rel"][m], 0.25)),
                     "rel_q75": float(np.quantile(tab["rel"][m], 0.75)),
                     "true_mag": float(np.mean(tab["true_mag"][m])),
                     "pred_mag": float(np.mean(tab["pred_mag"][m]))})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True,
                    help="one checkpoint per seed, e.g. runs/nm_s42/best.pt ...")
    ap.add_argument("--dataset", default="blackbird",
                    choices=("blackbird", "euroc", "uzhfpv", "sitl1"))
    ap.add_argument("--tag", default=None, help="output name; defaults to the dataset")
    a = ap.parse_args()
    tag = a.tag or a.dataset

    cfg = Config.load(str(ROOT / "configs" / "trial8.yaml"))
    cfg.data.data_dir = str(Path(cfg.data.data_dir).expanduser())
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    named = eval_pairs(a.dataset, cfg)

    per_seed, per_traj, corr = [], {}, {k: [] for k, _, _ in FACTORS}
    for ck in a.ckpts:
        net = CausalMambaDispNet(d_state=cfg.model.d_state, d_conv=cfg.model.d_conv,
                                 expand=cfg.model.expand,
                                 num_layers=cfg.model.num_layers).to(dev)
        net.load_state_dict(torch.load(ck, map_location=dev, weights_only=True))
        net.eval()

        tabs = {}
        for split, pairs in named.items():
            for name, seq in pairs:
                short = name.split("/")[0]
                tab = step_table(net, seq, cfg, dev)
                if not tab:
                    continue
                tabs[short] = tab
                rolled = rollout_full(net, seq, dev, cfg.data.window_size,
                                      cfg.data.label_start_index, cfg.data.label_stride)
                m = metrics(*rolled) if rolled is not None else {}
                row = per_traj.setdefault(short, {"split": split, "ate": [], "tde": [],
                                                  "speed": float(np.mean(tab["speed"])),
                                                  "accel": float(np.mean(tab["accel"])),
                                                  "drag": float(np.mean(tab["drag"])),
                                                  "rel_median": []})
                if m:
                    row["ate"].append(m["ATE"])
                    row["tde"].append(m["TDE"])
                row["rel_median"].append(float(np.median(tab["rel"])))

        allt = {k: np.concatenate([t[k] for t in tabs.values()]) for k in next(iter(tabs.values()))}
        seed_row = {"ckpt": str(ck),
                    "n_steps": int(len(allt["err"])),
                    "rmse": float(np.sqrt(np.mean(allt["err"] ** 2))),
                    "rel_median": float(np.median(allt["rel"])),
                    "deciles": {k: deciles(allt, k) for k, _, _ in FACTORS}}
        for k, _, _ in FACTORS:
            r_abs = stats.spearmanr(allt[k], allt["err"]).statistic
            r_rel = stats.spearmanr(allt[k], allt["rel"]).statistic
            corr[k].append((float(r_abs), float(r_rel)))
        per_seed.append(seed_row)
        print(f"{Path(ck).parent.name:<22} steps {seed_row['n_steps']:>6}  "
              f"RMSE {seed_row['rmse']:.5f}  rel median {seed_row['rel_median']:.3f}")

    # ---- report -----------------------------------------------------------------
    print(f"\n{'condition':<18}{'rho vs abs err':>16}{'rho vs rel err':>16}")
    corr_out = {}
    for k, label, _ in FACTORS:
        ab = np.array([c[0] for c in corr[k]])
        re = np.array([c[1] for c in corr[k]])
        corr_out[k] = {"abs_mean": float(ab.mean()), "abs_std": float(ab.std(ddof=1)),
                       "rel_mean": float(re.mean()), "rel_std": float(re.std(ddof=1))}
        print(f"{label:<18}{ab.mean():>+10.3f}±{ab.std(ddof=1):<5.3f}"
              f"{re.mean():>+10.3f}±{re.std(ddof=1):<5.3f}")

    print(f"\n{'trajectory':<14}{'split':<8}{'ATE [m]':>10}{'TDE [%]':>9}"
          f"{'speed':>8}{'|accel|':>9}{'drag':>7}{'rel err':>9}")
    for name, row in sorted(per_traj.items(), key=lambda kv: np.mean(kv[1]["ate"] or [0])):
        print(f"{name:<14}{row['split']:<8}{np.mean(row['ate']):>10.3f}"
              f"{np.mean(row['tde']):>9.3f}{row['speed']:>8.2f}{row['accel']:>9.2f}"
              f"{row['drag']:>7.2f}{np.mean(row['rel_median']):>9.3f}")

    # ---- figures ----------------------------------------------------------------
    figs = RESULTS / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.2))
    for ax, (k, label, unit) in zip(axes.ravel(), FACTORS):
        mids = np.array([d["mid"] for d in per_seed[0]["deciles"][k]])
        rel = np.array([[d["rel_median"] for d in s["deciles"][k]] for s in per_seed])
        rmse = np.array([[d["rmse"] for d in s["deciles"][k]] for s in per_seed])
        ax.plot(mids, rel.mean(0), color="#c0392b", lw=1.8, marker="o", ms=3.5,
                label="relative error (median)")
        ax.fill_between(mids, rel.mean(0) - rel.std(0), rel.mean(0) + rel.std(0),
                        color="#c0392b", alpha=0.15)
        ax.set_xlabel(f"{label} [{unit}]", fontsize=9)
        ax.set_ylabel("relative error", fontsize=9, color="#c0392b")
        ax.tick_params(labelsize=8)
        ax.spines[["top"]].set_visible(False)
        ax2 = ax.twinx()
        ax2.plot(mids, rmse.mean(0), color="#2b6cb0", lw=1.4, ls="--", marker="s", ms=3,
                 label="RMSE")
        ax2.set_ylabel("step RMSE [m]", fontsize=9, color="#2b6cb0")
        ax2.tick_params(labelsize=8)
        ax2.spines[["top"]].set_visible(False)
    fig.suptitle(f"Step error by flight condition — {tag}, {len(per_seed)} seeds "
                 "(deciles, equal counts)", fontsize=11)
    fig.tight_layout()
    fig.savefig(figs / f"error_by_condition_{tag}.png", dpi=180)
    print(f"\nwrote {figs / f'error_by_condition_{tag}.png'}")

    fig2, ax = plt.subplots(figsize=(6.0, 4.4))
    for split, color in (("seen", "#2b6cb0"), ("unseen", "#c0392b"),
                         ("euroc_test", "#2b6cb0"), ("uzhfpv_test", "#c0392b"),
                         ("sitl1", "#2b6cb0")):
        xs = [r["speed"] for r in per_traj.values() if r["split"] == split]
        ys = [np.mean(r["ate"]) for r in per_traj.values() if r["split"] == split]
        if xs:
            ax.scatter(xs, ys, color=color, s=38, label=split, zorder=3)
    for name, r in per_traj.items():
        ax.annotate(name, (r["speed"], np.mean(r["ate"])), fontsize=7,
                    xytext=(3, 3), textcoords="offset points", color="#555")
    ax.set_xlabel("Mean speed [m/s]", fontsize=9)
    ax.set_ylabel("Trajectory ATE [m]", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig2.tight_layout()
    fig2.savefig(figs / f"ate_vs_speed_{tag}.png", dpi=180)
    print(f"wrote {figs / f'ate_vs_speed_{tag}.png'}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"summary_{tag}.json").write_text(json.dumps(
        {"dataset": a.dataset, "checkpoints": [str(c) for c in a.ckpts],
         "spearman": corr_out, "per_seed": per_seed,
         "per_trajectory": {k: {**v, "ate_mean": float(np.mean(v["ate"])),
                                "tde_mean": float(np.mean(v["tde"]))}
                            for k, v in per_traj.items()}}, indent=2))
    print(f"wrote {RESULTS / f'summary_{tag}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
