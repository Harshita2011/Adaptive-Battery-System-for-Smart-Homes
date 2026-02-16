import argparse
import math
from pathlib import Path
from typing import List

import pandas as pd


KEY_METRICS = [
    "peak_temp_c",
    "soh_degradation_rate_per_hour",
    "protection_events",
    "energy_cost_index",
    "load_curtailment_count",
]


def protection_events(series: pd.Series) -> int:
    vals = series.fillna("").tolist()
    count = 0
    prev = None
    for v in vals:
        if prev == "OPTIMAL" and v == "PROTECTED":
            count += 1
        prev = v
    return count


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def welch_pvalue_normal_approx(a: pd.Series, b: pd.Series) -> float:
    # Lightweight fallback without scipy: Welch t with normal approximation.
    a = a.dropna().astype(float)
    b = b.dropna().astype(float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")

    ma, mb = float(a.mean()), float(b.mean())
    va, vb = float(a.var(ddof=1)), float(b.var(ddof=1))
    se = math.sqrt((va / len(a)) + (vb / len(b)))
    if se == 0:
        return 1.0

    z = (ma - mb) / se
    p_two_tailed = 2.0 * (1.0 - _normal_cdf(abs(z)))
    return max(0.0, min(1.0, p_two_tailed))


def scenario_metrics_per_experiment(df: pd.DataFrame) -> pd.DataFrame:
    group_cols: List[str] = ["scenario"]
    if "controller_mode" in df.columns:
        group_cols.append("controller_mode")
    else:
        df = df.copy()
        df["controller_mode"] = "unknown"
        group_cols.append("controller_mode")

    if "experiment_id" in df.columns:
        group_cols.append("experiment_id")
    else:
        df = df.copy()
        df["experiment_id"] = "single_experiment"
        group_cols.append("experiment_id")

    if "repeat_id" in df.columns:
        group_cols.append("repeat_id")
    else:
        df = df.copy()
        df["repeat_id"] = 1
        group_cols.append("repeat_id")

    rows = []
    for keys, g in df.groupby(group_cols):
        g = g.sort_values("timestamp_utc")
        key_map = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))

        first_h = float(g["health"].iloc[0])
        last_h = float(g["health"].iloc[-1])
        duration_hours = max(len(g) / 3600.0, 1e-9)
        soh_deg_rate = (first_h - last_h) / duration_hours

        peak_temp = float(g["temp_c"].max())
        prot_events = protection_events(g["status"])
        energy_cost_idx = float(g["energy_cost_index"].sum())
        curtailments = int(g["turn_off_commands"].sum())
        fallback_events = int(((g["mode"] == "LIVE") & (g["source"] == "SIM")).sum())

        rows.append(
            {
                **key_map,
                "samples": len(g),
                "peak_temp_c": round(peak_temp, 6),
                "soh_start": round(first_h, 6),
                "soh_end": round(last_h, 6),
                "soh_degradation_rate_per_hour": round(soh_deg_rate, 6),
                "protection_events": prot_events,
                "energy_cost_index": round(energy_cost_idx, 6),
                "load_curtailment_count": curtailments,
                "live_to_sim_fallback_events": fallback_events,
            }
        )
    return pd.DataFrame(rows)


def summarize_with_ci(metrics_df: pd.DataFrame) -> pd.DataFrame:
    agg_rows = []
    for (scenario, mode), g in metrics_df.groupby(["scenario", "controller_mode"]):
        base = {"scenario": scenario, "controller_mode": mode, "n": len(g)}

        for m in KEY_METRICS + ["live_to_sim_fallback_events"]:
            vals = g[m].astype(float)
            mean = float(vals.mean())
            std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            ci95 = 1.96 * (std / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
            base[f"{m}_mean"] = round(mean, 6)
            base[f"{m}_std"] = round(std, 6)
            base[f"{m}_ci95"] = round(ci95, 6)

        agg_rows.append(base)
    return pd.DataFrame(agg_rows).sort_values(["scenario", "controller_mode"])


def significance_vs_baseline(metrics_df: pd.DataFrame, baseline_mode: str = "rule_only") -> pd.DataFrame:
    rows = []
    for scenario, g in metrics_df.groupby("scenario"):
        g_base = g[g["controller_mode"] == baseline_mode]
        if g_base.empty:
            continue

        for mode, g_mode in g.groupby("controller_mode"):
            if mode == baseline_mode:
                continue

            for metric in KEY_METRICS:
                p = welch_pvalue_normal_approx(g_mode[metric], g_base[metric])
                rows.append(
                    {
                        "scenario": scenario,
                        "baseline_mode": baseline_mode,
                        "compare_mode": mode,
                        "metric": metric,
                        "mean_baseline": round(float(g_base[metric].mean()), 6),
                        "mean_compare": round(float(g_mode[metric].mean()), 6),
                        "delta_compare_minus_baseline": round(
                            float(g_mode[metric].mean() - g_base[metric].mean()), 6
                        ),
                        "p_value_approx": round(float(p), 8) if not math.isnan(p) else None,
                    }
                )
    return pd.DataFrame(rows).sort_values(["scenario", "metric", "compare_mode"])


def write_report(summary: pd.DataFrame, sig_df: pd.DataFrame, out_md: Path):
    lines = []
    lines.append("# Scenario Evaluation Report")
    lines.append("")
    lines.append("## KPI Summary (Mean ± CI95)")
    lines.append("")
    try:
        lines.append(summary.to_markdown(index=False))
    except Exception:
        lines.append("```text")
        lines.append(summary.to_string(index=False))
        lines.append("```")

    lines.append("")
    lines.append("## Pairwise Significance vs Baseline (rule_only)")
    lines.append("")
    if sig_df.empty:
        lines.append("No baseline-comparison rows available.")
    else:
        try:
            lines.append(sig_df.to_markdown(index=False))
        except Exception:
            lines.append("```text")
            lines.append(sig_df.to_string(index=False))
            lines.append("```")

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- CI95 computed as `1.96 * std / sqrt(n)` across repeats.")
    lines.append("- p-values are Welch-style normal approximations (no scipy dependency).")
    lines.append("- Re-run with higher repeats for stronger statistical power.")
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=False, help="Path to scenario CSV from run_test_scenarios.ps1")
    parser.add_argument("--baseline-mode", default="rule_only", help="Baseline controller mode for significance")
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else None
    if input_path and input_path.is_dir():
        input_path = input_path / "scenario_log.csv"

    if input_path is None or not input_path.exists():
        eval_root = Path("data/eval")
        candidates = (
            sorted(eval_root.glob("*/scenario_log.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
            if eval_root.exists()
            else []
        )
        if not candidates:
            raise FileNotFoundError("No scenario_log.csv found. Run scripts/run_test_scenarios.ps1 first.")
        input_path = candidates[0]
        print(f"[info] Using latest scenario log: {input_path}")

    out_dir = input_path.parent
    per_exp_csv = out_dir / "metrics_per_experiment.csv"
    summary_csv = out_dir / "metrics_summary.csv"
    sig_csv = out_dir / "significance_vs_baseline.csv"
    report_md = out_dir / "evaluation_report.md"

    df = pd.read_csv(input_path)
    per_exp = scenario_metrics_per_experiment(df)
    summary = summarize_with_ci(per_exp)
    sig_df = significance_vs_baseline(per_exp, baseline_mode=args.baseline_mode)

    per_exp.to_csv(per_exp_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    sig_df.to_csv(sig_csv, index=False)
    write_report(summary, sig_df, report_md)

    print(f"[done] per-experiment metrics: {per_exp_csv}")
    print(f"[done] summary with CI: {summary_csv}")
    print(f"[done] significance: {sig_csv}")
    print(f"[done] report: {report_md}")


if __name__ == "__main__":
    main()

