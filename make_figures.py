"""Regenerate all manuscript and supplementary figures from shipped CSV outputs.

Usage:
    python make_figures.py --data-dir . --out .

The script intentionally contains no hard-coded machine-specific paths.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


def _box(ax, x, y, w, h, text, fontsize=9.2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        linewidth=1.2,
        facecolor="white",
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=12, linewidth=1.1,
        shrinkA=0, shrinkB=0,
    ))


def figure_1(out: Path):
    """System architecture, with all edge labels clear of node boxes."""
    fig, ax = plt.subplots(figsize=(12.0, 6.8))
    ax.set_xlim(0, 14.0)
    ax.set_ylim(0, 8.0)
    ax.axis("off")

    # Top process chain.
    _box(ax, 0.45, 5.05, 2.55, 1.15, "Low-carbon electricity\n+ B300-equivalent compute")
    _box(ax, 3.70, 5.05, 2.55, 1.15, "Direct liquid cooling\n85% capture screen")
    _box(ax, 6.95, 5.05, 2.90, 1.15, "Useful low-temperature heat\nwater + dryer-air preheat")
    _box(ax, 10.65, 5.05, 2.75, 1.15, "Host process demand\ncar wash / laundromat")

    # Secondary paths.
    _box(ax, 0.45, 0.75, 2.75, 1.15, "Spectrum-XGS\nscale-across fabric")
    _box(ax, 3.70, 0.75, 2.75, 1.15, "Residual room heat\nair-side dry rejection")
    _box(ax, 7.05, 2.35, 2.70, 1.15, "Thermal storage\n4 h screen")
    _box(ax, 10.65, 2.35, 2.75, 1.15, "PV/T rejection\nthen optional trim")

    # Horizontal top arrows. Labels are deliberately above the boxes rather than on their borders.
    _arrow(ax, 3.00, 5.625, 3.70, 5.625)
    ax.text(3.35, 6.42, "IT electricity", ha="center", va="bottom", fontsize=8.3)
    _arrow(ax, 6.25, 5.625, 6.95, 5.625)
    ax.text(6.60, 6.42, "captured heat", ha="center", va="bottom", fontsize=8.3)
    _arrow(ax, 9.85, 5.625, 10.65, 5.625)
    ax.text(10.25, 6.42, "useful heat", ha="center", va="bottom", fontsize=8.3)

    # Scale-across connection into compute.
    _arrow(ax, 1.83, 1.90, 1.83, 5.05)
    ax.text(2.03, 3.47, "distributed jobs / collectives", rotation=90,
            ha="left", va="center", fontsize=8.2)

    # Uncaptured room-heat path.
    _arrow(ax, 4.98, 5.05, 4.98, 1.90)
    ax.text(5.18, 3.55, "uncaptured heat", ha="left", va="center", fontsize=8.2)

    # Surplus captured heat to storage and rejection.
    _arrow(ax, 8.40, 5.05, 8.40, 3.50)
    ax.text(8.62, 4.25, "surplus", ha="left", va="center", fontsize=8.2)
    _arrow(ax, 9.75, 2.925, 10.65, 2.925)
    ax.text(10.20, 3.72, "rejection duty", ha="center", va="bottom", fontsize=8.2)

    ax.text(0.45, 7.42, "Metropolitan distributed AI heat-reuse module",
            fontsize=12.2, fontweight="bold", va="top")
    ax.text(
        10.70, 1.55,
        "Recovered process water can supply evaporative make-up after treatment;\n"
        "dry-only operation uses no direct cooling water.",
        fontsize=8.3, va="top",
    )
    fig.tight_layout()
    fig.savefig(out / "Figure_1_architecture.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_2(data: Path, out: Path):
    """Metropolitan site geometry and propagation summary."""
    sites = pd.read_csv(data / "network_sites.csv")
    fig, ax = plt.subplots(figsize=(7.7, 7.0))
    ax.scatter(sites.x_km, sites.y_km, s=16, alpha=0.82)
    ax.scatter([0], [0], marker="*", s=170, label="Metropolitan fabric core")
    for rr in [10, 20, 35]:
        ax.add_patch(Circle((0, 0), rr, fill=False, linewidth=0.8, alpha=0.45))
    ax.set_aspect("equal")
    ax.set_xlim(-38, 38)
    ax.set_ylim(-38, 38)
    ax.set_xlabel("Modelled east-west offset (km)")
    ax.set_ylabel("Modelled north-south offset (km)")
    ax.legend(loc="upper right", frameon=False)
    median = float(sites.one_way_route_adjusted_us.median())
    maximum = float(sites.one_way_route_adjusted_us.max())
    gpus = float(sites.gpus.sum())
    endpoint = gpus * 800 / 1e6
    summary = (
        f"120 sites; {gpus:,.0f} B300-equivalent accelerators; {endpoint:.2f} Pb/s aggregate installed endpoint line rate.\n"
        f"Route-adjusted one-way propagation: {median:.0f} µs median; {maximum:.0f} µs maximum. "
        "Application-level training efficiency is not inferred."
    )
    # Put the summary outside the map rather than covering site points.
    fig.text(0.12, 0.012, summary, fontsize=8.4, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / "Figure_2_scale_across_network.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_3(data: Path, out: Path):
    s = pd.read_csv(data / "sensitivity.csv").set_index("scenario")
    order = [
        "profile_heat_following", "profile_no_scheduler", "flat_0.649",
        "flat_0.80", "flat_0.80_no_scheduler", "flat_0.90", "flat_0.95",
    ]
    labels = [
        "Profile +\nplacement", "Profile, no\nplacement", "Flat 64.9%",
        "Flat 80% +\nplacement", "Flat 80%, no\nplacement",
        "Flat 90% +\nplacement", "Flat 95% +\nplacement",
    ]
    vals = [float(s.loc[o, "it_to_useful_heat_pct"]) for o in order]
    fig, ax = plt.subplots(figsize=(9.3, 5.1))
    bars = ax.bar(np.arange(len(vals)), vals)
    ax.set_xticks(np.arange(len(vals)), labels)
    ax.set_ylabel("Useful process heat / IT electricity (%)")
    ax.set_ylim(0, 90)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.8, f"{v:.1f}%", ha="center", fontsize=8.8)
    fig.tight_layout()
    fig.savefig(out / "Figure_3_workload_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_4(data: Path, out: Path):
    h = pd.read_csv(data / "laundry_heat_sensitivity.csv")
    fig, ax = plt.subplots(figsize=(7.7, 5.0))
    ax.plot(h.laundry_kWhth_per_load, h.it_to_useful_heat_pct, marker="o", label="IT to useful heat")
    ax.plot(h.laundry_kWhth_per_load, h.heat_demand_coverage_pct, marker="s", label="Host heat demand covered")
    ax.set_xlabel("Assumed displaceable laundromat heat (kWh$_{th}$ per load)")
    ax.set_ylabel("Thermal metric (%)")
    ax.set_ylim(45, 103)
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "Figure_4_heat_demand_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_5(data: Path, out: Path):
    c = pd.read_csv(data / "carbon_sensitivity.csv")
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(c.electricity_gCO2e_kWh, c.same_supply_reduction_pct, marker="o")
    for x, y in zip(c.electricity_gCO2e_kWh, c.same_supply_reduction_pct):
        ax.annotate(f"{y:.1f}%", (x, y), xytext=(4, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Electricity lifecycle intensity in both architectures (gCO$_2$e kWh$^{-1}$)")
    ax.set_ylabel("Operational GHG reduction from integrated architecture (%)")
    ax.set_xlim(0, 510)
    ax.set_ylim(20, 95)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "Figure_5_carbon_same_supply.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_6(data: Path, out: Path):
    s = pd.read_csv(data / "sensitivity.csv").set_index("scenario")
    order = ["profile_heat_following", "flat_0.649", "flat_0.80", "flat_0.90", "flat_0.95", "dry_only"]
    labels = ["Profile", "Flat 64.9%", "Flat 80%", "Flat 90%", "Flat 95%", "Dry-only"]
    vals = [float(s.loc[o, "cooling_water_m3"]) for o in order]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    bars = ax.bar(np.arange(len(vals)), vals)
    ax.set_xticks(np.arange(len(vals)), labels)
    ax.set_ylabel("Direct evaporative cooling water (m$^3$ yr$^{-1}$)")
    top = max(vals)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v + max(top * 0.015, 8), f"{v:.0f}", ha="center", fontsize=8.8)
    ax.set_ylim(0, top * 1.14)
    fig.tight_layout()
    fig.savefig(out / "Figure_6_direct_water.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_s1(out: Path):
    counts = {
        "Small car wash": 9, "Medium car wash": 15, "Large car wash": 36,
        "Small laundromat": 41, "Medium laundromat": 83, "Large laundromat": 124,
    }
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    x = np.arange(len(counts)); vals = list(counts.values())
    bars = ax.bar(x, vals)
    ax.set_xticks(x, ["Small\ncar wash", "Medium\ncar wash", "Large\ncar wash",
                      "Small\nlaundromat", "Medium\nlaundromat", "Large\nlaundromat"])
    ax.set_ylabel("B300-equivalent accelerators per site (thermal-sizing input)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+2, str(v), ha="center", fontsize=9)
    ax.set_ylim(0, 140)
    fig.tight_layout()
    fig.savefig(out / "Figure_S1_facility_sizing.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_s2(data: Path, out: Path):
    sites = pd.read_csv(data / "network_sites.csv")
    lat = np.sort(sites.one_way_route_adjusted_us.to_numpy(float))
    cdf = np.arange(1, len(lat)+1) / len(lat)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.plot(lat, cdf)
    ax.set_xlabel("Route-adjusted one-way propagation to metropolitan core (µs)")
    ax.set_ylabel("Cumulative fraction of sites")
    ax.set_xlim(0, 260)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "Figure_S2_propagation_cdf.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".", help="Directory containing shipped CSV outputs")
    ap.add_argument("--out", default=".", help="Directory for regenerated PNG figures")
    args = ap.parse_args()
    data = Path(args.data_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    figure_1(out)
    figure_2(data, out)
    figure_3(data, out)
    figure_4(data, out)
    figure_5(data, out)
    figure_6(data, out)
    figure_s1(out)
    figure_s2(data, out)


if __name__ == "__main__":
    main()
