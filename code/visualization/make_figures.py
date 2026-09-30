from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10.5,
        "axes.labelsize": 10.5,
        "axes.titlesize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "savefig.dpi": 300,
        "figure.facecolor": "white",
    }
)

COLORS = {
    "A": "#2878B5",
    "B": "#E39A2D",
    "C": "#2A9D78",
    "direct": "#2F7FB9",
    "relay": "#E9A23B",
    "stock": "#4C9B72",
    "need": "#3C78A8",
    "gap": "#D86A2E",
}


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def safe_payload_figure() -> None:
    values = np.array(
        [
            [25.0, 30.0, 80.0],
            [25.0, 30.0, 68.86037775413683],
            [25.0, 30.0, 68.20455041133383],
            [25.0, 30.0, 63.69649250314165],
            [25.0, 30.0, 80.0],
            [25.0, 30.0, 80.0],
            [25.0, 30.0, 80.0],
            [25.0, 28.800781652904007, 58.90310867739053],
            [25.0, 30.0, 80.0],
            [25.0, 30.0, 80.0],
            [25.0, 30.0, 80.0],
            [25.0, 30.0, 68.11775809489428],
            [25.0, 30.0, 80.0],
            [25.0, 30.0, 80.0],
            [25.0, 30.0, 80.0],
        ]
    )
    capacities = np.array([25.0, 30.0, 80.0])
    ratio = values / capacities
    nodes = [f"S{i:03d}" for i in range(1, 16)]

    transport = json.loads(
        (ROOT / "data/processed" / "transport.json").read_text(
            encoding="utf-8"
        )
    )
    node_map = {node["id"]: node for node in transport["nodes"]}
    dem_path = ROOT / "data/processed" / "dem.tif"

    def sample_segment(destination: str, samples: int = 240):
        origin = node_map["O01"]
        target = node_map[destination]
        lon = np.linspace(origin["lon"], target["lon"], samples)
        lat = np.linspace(origin["lat"], target["lat"], samples)
        if dem_path.exists():
            import rasterio

            with rasterio.open(dem_path) as dataset:
                terrain = np.array(
                    [value[0] for value in dataset.sample(zip(lon, lat))],
                    dtype=float,
                )
                nodata = dataset.nodata
            if nodata is not None:
                terrain[terrain == nodata] = np.nan
        else:
            terrain = np.linspace(origin["elev_m"], target["elev_m"], samples)
        distance_km = np.linspace(
            0,
            haversine_km(origin["lon"], origin["lat"], target["lon"], target["lat"]),
            samples,
        )
        return distance_km, terrain

    distances = []
    terrain_maxima = []
    for node in nodes:
        origin = node_map["O01"]
        target = node_map[node]
        distances.append(
            haversine_km(
                origin["lon"], origin["lat"], target["lon"], target["lat"]
            )
        )
        _, profile = sample_segment(node, samples=160)
        terrain_maxima.append(float(np.nanmax(profile)))

    fig = plt.figure(figsize=(11.2, 8.3))
    grid = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.28)
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[1, 0])
    ax4 = fig.add_subplot(grid[1, 1])

    image = ax1.imshow(ratio, vmin=0.7, vmax=1.0, cmap="Blues", aspect="auto")
    ax1.set_xticks(range(3), ["Type A", "Type B", "Type C"])
    ax1.set_yticks(range(15), nodes)
    ax1.set_xlabel("Aircraft type")
    ax1.set_ylabel("Service area")
    for i in range(15):
        for j in range(3):
            ax1.text(
                j,
                i,
                f"{values[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if ratio[i, j] > 0.88 else "black",
                fontsize=8.5,
            )
    colorbar = fig.colorbar(image, ax=ax1, pad=0.025, fraction=0.055)
    colorbar.set_label("Safe payload / rated payload")
    ax1.set_title("(a) Safe-payload matrix at a 20% reserve")

    markers = {"A": "o", "B": "s", "C": "^"}
    for type_index, aircraft_type in enumerate("ABC"):
        scatter = ax2.scatter(
            distances,
            terrain_maxima,
            c=ratio[:, type_index],
            cmap="viridis",
            vmin=0.68,
            vmax=1.0,
            marker=markers[aircraft_type],
            s=42,
            edgecolor="black",
            linewidth=0.35,
            label=f"Type {aircraft_type}",
        )
    ax2.set_xlabel("Horizontal distance from dispatch center (km)")
    ax2.set_ylabel("Maximum terrain elevation on segment (m)")
    ax2.set_title("(b) Terrain-distance-payload coupling")
    ax2.legend(frameon=False, loc="lower right")
    colorbar2 = fig.colorbar(scatter, ax=ax2, pad=0.025, fraction=0.055)
    colorbar2.set_label("Safe payload / rated payload")

    representative = ["S004", "S008", "S012", "S015"]
    route_colors = ["#2878B5", "#E39A2D", "#2A9D78", "#C44E52"]
    for node, color in zip(representative, route_colors):
        distance_km, profile = sample_segment(node)
        cruise = float(np.nanmax(profile)) + 50.0
        ax3.plot(distance_km, profile, color=color, linewidth=1.45)
        ax3.plot(
            distance_km,
            np.full_like(distance_km, cruise),
            color=color,
            linewidth=1.15,
            linestyle="--",
        )
        ax3.text(
            distance_km[-1],
            cruise + 8,
            f"{node} cruise",
            color=color,
            fontsize=8.3,
            ha="right",
        )
    ax3.set_xlabel("Distance along segment (km)")
    ax3.set_ylabel("Elevation (m)")
    ax3.set_title("(c) Representative DEM and cruise-altitude profiles")
    ax3.legend(
        handles=[
            Line2D([], [], color="black", linewidth=1.4, label="Terrain"),
            Line2D(
                [],
                [],
                color="black",
                linewidth=1.2,
                linestyle="--",
                label="Cruise altitude",
            ),
        ],
        frameon=False,
        loc="upper left",
    )

    reserve = np.array([10, 20, 30])
    reserve_payload = {
        "A": np.array([25.000, 25.000, 23.859]),
        "B": np.array([30.000, 29.909, 28.124]),
        "C": np.array([78.687, 75.099, 68.516]),
    }
    for aircraft_type in "ABC":
        ax4.plot(
            reserve,
            reserve_payload[aircraft_type],
            marker=markers[aircraft_type],
            color=COLORS[aircraft_type],
            linewidth=1.8,
            label=f"Type {aircraft_type}",
        )
        for x_value, y_value in zip(reserve, reserve_payload[aircraft_type]):
            ax4.annotate(
                f"{y_value:.2f}",
                (x_value, y_value),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=7.8,
            )
    ax4.set_xlabel("Return reserve (%)")
    ax4.set_ylabel("Average safe payload (kg)")
    ax4.set_xticks(reserve)
    ax4.set_title("(d) Reserve sensitivity by aircraft type")
    ax4.legend(frameon=False)

    fig.suptitle("Multi-view terrain-aware physical feasibility", fontsize=12.5)
    save(fig, "safe_payload")


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius = 6371.0088
    phi1, phi2 = np.radians([lat1, lat2])
    dphi = phi2 - phi1
    dlambda = np.radians(lon2 - lon1)
    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )
    return float(2.0 * radius * np.arcsin(np.sqrt(a)))


def convergence_figure() -> None:
    path = ROOT / "results" / "experiments" / "algorithm_convergence.csv"
    grouped: dict[str, list[dict[str, str]]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(row["algorithm"], []).append(row)

    order = ["SA", "TS", "GA", "DGWO", "DPSO"]
    labels = {
        "SA": "SA",
        "TS": "TS",
        "GA": "GA",
        "DGWO": "DGWO",
        "DPSO": "DPSO",
    }
    palette = ["#2878B5", "#E39A2D", "#2A9D78", "#C44E52", "#7E57C2"]
    metrics = [
        ("makespan_s", "Makespan (s)"),
        ("energy_kwh", "Transport energy (kWh)"),
        ("sorties", "Sorties"),
        ("weighted_tardiness", "Weighted tardiness"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(11.0, 8.4), layout="constrained")
    trace_axes = axes.flat[:4]
    for ax, (column, ylabel) in zip(trace_axes, metrics):
        for algorithm, color in zip(order, palette):
            rows = grouped[algorithm]
            iterations = np.array([float(row["iteration"]) for row in rows])
            objective = np.array([float(row[column]) for row in rows])
            ax.plot(
                iterations,
                objective,
                color=color,
                linewidth=1.25,
                drawstyle="steps-post",
            )
            ax.scatter(
                iterations[-1],
                objective[-1],
                color=color,
                s=14,
                zorder=4,
            )
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.set_xlim(left=0)
        ax.grid(alpha=0.18)

    final_rows = []
    for algorithm in order:
        row = grouped[algorithm][-1]
        final_rows.append(
            {
                "label": labels[algorithm],
                "makespan": float(row["makespan_s"]),
                "energy": float(row["energy_kwh"]),
                "sorties": float(row["sorties"]),
                "tardiness": float(row["weighted_tardiness"]),
            }
        )
    final_rows.append(
        {
            "label": "CP-SAT",
            "makespan": 6104.093554854697,
            "energy": 69.14247161396887,
            "sorties": 23.0,
            "tardiness": 0.0,
        }
    )

    scatter_ax = axes[2, 0]
    marker_map = {"CP-SAT": "*", "DGWO": "o", "DPSO": "x"}
    for row, color in zip(final_rows, palette + ["#202020"]):
        marker = marker_map.get(row["label"], "o")
        size = 55 + 12.0 * row["sorties"]
        scatter_ax.scatter(
            row["makespan"],
            row["energy"],
            s=size,
            color=color,
            marker=marker,
            edgecolor=None if marker == "x" else "white",
            linewidth=0.6,
            zorder=3,
        )
        if row["label"] == "DPSO":
            label = "DGWO / DPSO\n36 sorties"
        elif row["label"] == "DGWO":
            label = ""
        else:
            label = f"{row['label']}\n{row['sorties']:.0f} sorties"
        if label:
            scatter_ax.annotate(
                label,
                (row["makespan"], row["energy"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8.0,
            )
    scatter_ax.set_xlabel("Final makespan (s)")
    scatter_ax.set_ylabel("Final transport energy (kWh)")
    scatter_ax.set_title("(e) Final objective trade-off", fontsize=10)
    scatter_ax.grid(alpha=0.18)

    heat_ax = axes[2, 1]
    metric_keys = ["makespan", "energy", "sorties", "tardiness"]
    matrix = np.array([[row[key] for key in metric_keys] for row in final_rows])
    normalized = (matrix.max(axis=0) - matrix) / np.maximum(
        matrix.max(axis=0) - matrix.min(axis=0),
        1e-12,
    )
    heat_ax.imshow(normalized, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    heat_ax.set_xticks(
        range(4),
        ["Makespan", "Energy", "Sorties", "Tardiness"],
        rotation=20,
        ha="right",
    )
    heat_ax.set_yticks(range(len(final_rows)), [row["label"] for row in final_rows])
    heat_ax.set_title("(f) Normalized final performance", fontsize=10)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            text_value = f"{value:.0f}" if column_index != 1 else f"{value:.1f}"
            heat_ax.text(
                column_index,
                row_index,
                text_value,
                ha="center",
                va="center",
                fontsize=7.4,
                color="white" if normalized[row_index, column_index] > 0.45 else "black",
            )

    handles = [
        Line2D([], [], color=color, linewidth=1.8, label=labels[algorithm])
        for algorithm, color in zip(order, palette)
    ]
    fig.legend(
        handles,
        [labels[algorithm] for algorithm in order],
        frameon=False,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.006),
    )
    fig.suptitle("Convergence and final-performance atlas of five transport algorithms", y=1.01)
    save(fig, "solver_convergence")


TRANSPORT_SCHEDULE = [
    ("T002", "U05", "B", 0.0, 1555.4),
    ("T003", "U07", "C", 0.0, 1626.2),
    ("T004", "U01", "A", 0.0, 1313.5),
    ("T005", "U06", "B", 0.0, 1510.5),
    ("T008", "U08", "C", 0.0, 1560.2),
    ("T009", "U02", "A", 0.0, 2275.8),
    ("T010", "U03", "A", 0.0, 1727.4),
    ("T017", "U04", "A", 0.0, 2603.4),
    ("T011", "U01", "A", 1313.5, 3005.7),
    ("T007", "U06", "B", 1510.5, 3380.2),
    ("T006", "U05", "B", 1555.4, 3478.1),
    ("T016", "U08", "C", 1560.2, 3668.0),
    ("T018", "U07", "C", 1626.2, 3741.7),
    ("T014", "U03", "A", 1727.4, 3898.6),
    ("T015", "U02", "A", 2275.8, 4003.2),
    ("T020", "U04", "A", 2777.9, 5211.0),
    ("T012", "U06", "B", 3380.2, 5282.1),
    ("T013", "U05", "B", 3478.1, 5666.8),
    ("T001", "U01", "A", 3640.9, 5609.5),
    ("T019", "U08", "C", 3668.0, 6104.1),
    ("T023", "U07", "C", 3741.7, 6042.8),
    ("T022", "U02", "A", 4055.9, 5744.5),
    ("T021", "U03", "A", 4118.5, 5810.7),
]


JOINT_TRANSPORT = [
    ("T002", "U01", "A", 0.0, 2536.7),
    ("T005", "U02", "A", 0.0, 2337.1),
    ("T006", "U03", "A", 0.0, 1667.4),
    ("T007", "U07", "C", 0.0, 1692.2),
    ("T008", "U04", "A", 0.0, 1612.1),
    ("T009", "U05", "B", 0.0, 1869.8),
    ("T010", "U06", "B", 0.0, 1636.3),
    ("T011", "U08", "C", 0.0, 1494.2),
    ("T004", "U08", "C", 1568.8, 3869.8),
    ("T013", "U04", "A", 1612.1, 3267.0),
    ("T012", "U06", "B", 1636.3, 2843.6),
    ("T014", "U03", "A", 1667.4, 2980.9),
    ("T003", "U07", "C", 1692.2, 3800.0),
    ("T001", "U05", "B", 1869.8, 3626.4),
    ("T018", "U02", "A", 3495.7, 5727.7),
    ("T015", "U01", "A", 3525.7, 5217.9),
    ("T021", "U06", "B", 3591.3, 5107.5),
    ("T024", "U05", "B", 3626.4, 5588.3),
    ("T023", "U04", "A", 3722.2, 5833.4),
    ("T017", "U07", "C", 3800.0, 6805.8),
    ("T020", "U08", "C", 3869.8, 6563.4),
    ("T019", "U03", "A", 3884.0, 6146.6),
    ("T022", "U06", "B", 5107.5, 6949.3),
    ("T016", "U01", "A", 5217.9, 6906.5),
]


RELAY_SCHEDULE = [
    ("RE05", "R01", 137.3, 774.4, 2816.7, 3282.1),
    ("RE03", "R02", 281.2, 779.5, 1221.1, 1536.1),
    ("RE01", "R02", 1836.1, 2494.4, 3448.4, 3916.9),
    ("RE02", "R01", 3582.1, 4278.9, 6586.3, 7111.6),
    ("RE04", "R02", 4216.9, 4715.2, 6085.4, 6400.4),
]


def gantt(
    transport_rows: list[tuple[str, str, str, float, float]],
    relay_rows: list[tuple[str, str, float, float, float, float]] | None,
    name: str,
    title: str,
) -> None:
    machines = sorted({row[1] for row in transport_rows})
    if relay_rows:
        machines += sorted({row[1] for row in relay_rows})
    y_index = {machine: idx for idx, machine in enumerate(machines)}
    fig, ax = plt.subplots(figsize=(8.6, 4.8 if relay_rows else 4.35))
    for sortie, machine, aircraft_type, start, end in transport_rows:
        y = y_index[machine]
        ax.barh(
            y,
            (end - start) / 60.0,
            left=start / 60.0,
            height=0.58,
            color=COLORS[aircraft_type],
            edgecolor="white",
            linewidth=0.45,
        )
        ax.text(
            (start + end) / 120.0,
            y,
            sortie[1:].lstrip("0"),
            ha="center",
            va="center",
            color="white",
            fontsize=7.4,
        )
    if relay_rows:
        for mission, machine, start, service_start, service_end, end in relay_rows:
            y = y_index[machine]
            ax.barh(
                y,
                (end - start) / 60.0,
                left=start / 60.0,
                height=0.58,
                color="#D9CFE3",
                edgecolor="white",
                linewidth=0.45,
            )
            ax.barh(
                y,
                (service_end - service_start) / 60.0,
                left=service_start / 60.0,
                height=0.58,
                color=COLORS["relay"],
                edgecolor="white",
                linewidth=0.45,
            )
            ax.text(
                (start + end) / 120.0,
                y,
                mission,
                ha="center",
                va="center",
                fontsize=7.2,
            )
    ax.set_yticks(range(len(machines)), machines)
    ax.invert_yaxis()
    ax.set_xlabel("Time from mission start (min)")
    ax.set_ylabel("Aircraft")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    if relay_rows:
        ax.legend(
            handles=[
                Patch(facecolor=COLORS["A"], label="Type A"),
                Patch(facecolor=COLORS["B"], label="Type B"),
                Patch(facecolor=COLORS["C"], label="Type C"),
                Patch(facecolor=COLORS["relay"], label="Relay service"),
                Patch(facecolor="#D9CFE3", label="Relay flight/turnaround"),
            ],
            frameon=False,
            ncol=5,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
        )
    else:
        ax.legend(
            handles=[
                Patch(facecolor=COLORS["A"], label="Type A"),
                Patch(facecolor=COLORS["B"], label="Type B"),
                Patch(facecolor=COLORS["C"], label="Type C"),
            ],
            frameon=False,
            ncol=3,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
        )
    ax.set_title(title)
    save(fig, name)


def algorithm_comparison_figure() -> None:
    labels = ["SA", "TS", "GA", "DGWO", "DPSO", "CP-SAT"]
    makespan = np.array([9852.58, 7879.93, 9179.75, 11313.83, 11313.83, 6104.09])
    energy = np.array([102.99, 87.15, 94.52, 112.69, 112.69, 69.14])
    sorties = np.array([35, 26, 30, 36, 36, 23])
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 6.0), layout="constrained")
    axes[0].bar(x, makespan, color="#4D8DB8")
    axes[0].set_ylabel("Makespan (s)")
    axes[1].bar(x, energy, color="#E39A2D")
    axes[1].set_ylabel("Energy (kWh)")
    for ax, values in [(axes[0], makespan), (axes[1], energy)]:
        ax.set_xticks(x, labels)
        ax.grid(axis="y")
        for idx, value in enumerate(values):
            ax.text(idx, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8.2)
    for idx, count in enumerate(sorties):
        axes[0].text(idx, makespan[idx] * 0.48, f"{count} sorties", ha="center", color="white", fontsize=8.1)
    fig.suptitle("Algorithm comparison under the on-time delivery objective")
    save(fig, "algorithm_comparison")


def communication_heatmap() -> None:
    path = ROOT / "results/verification" / "joint_nominal" / "communication.csv"
    intervals: dict[str, list[tuple[float, float, str]]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            intervals.setdefault(row["sortie"], []).append(
                (float(row["start_s"]), float(row["end_s"]), row["provider"])
            )

    schedule = {row[0]: row for row in JOINT_TRANSPORT}
    sortie_order = [row[0] for row in sorted(JOINT_TRANSPORT, key=lambda item: (item[3], item[4]))]
    fig, ax = plt.subplots(figsize=(8.8, 7.0))
    for y, sortie in enumerate(sortie_order):
        start = schedule[sortie][3]
        end = schedule[sortie][4]
        ax.barh(y, (end - start) / 60.0, left=start / 60.0, height=0.72, color="#F1F3F4")
        for left, right, provider in sorted(intervals.get(sortie, [])):
            color = COLORS["direct"] if provider == "G01" else COLORS["relay"]
            ax.barh(
                y,
                (right - left) / 60.0,
                left=left / 60.0,
                height=0.72,
                color=color,
                linewidth=0,
            )
    ax.set_yticks(range(len(sortie_order)), sortie_order)
    ax.invert_yaxis()
    ax.set_xlabel("Time from mission start (min)")
    ax.set_ylabel("Transport sortie")
    ax.set_title("Communication mode over each transport sortie")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.legend(
        handles=[
            Patch(facecolor=COLORS["direct"], label="Direct link"),
            Patch(facecolor=COLORS["relay"], label="Relay-assisted"),
            Patch(facecolor="#F1F3F4", label="Inactive/non-certified"),
        ],
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
    )
    save(fig, "communication_heatmap")


def resource_gap_figure() -> None:
    resources = [
        "A UAV",
        "B UAV",
        "C UAV",
        "A battery",
        "B battery",
        "C battery",
        "Relay UAV",
        "Relay energy",
    ]
    stock = np.array([4, 2, 2, 6, 4, 4, 2, 6])
    unpartitioned = np.array([4, 2, 2, 6, 4, 4, 2, 3])
    two_groups = np.array([7, 4, 4, 9, 6, 5, 3, 4])
    three_groups = np.array([7, 4, 5, 9, 6, 6, 3, 4])
    x = np.arange(len(resources))
    width = 0.24

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6), sharey=True, layout="constrained")
    for ax, needs, title in [
        (axes[0], two_groups, "Two independent task groups"),
        (axes[1], three_groups, "Three independent task groups"),
    ]:
        ax.bar(x - width, unpartitioned, width, color="#A7B0B5", label="Unpartitioned peak")
        ax.bar(x, needs, width, color=COLORS["need"], label="Independent-group demand")
        ax.bar(x + width, stock, width, color=COLORS["stock"], label="Available stock")
        for idx, value in enumerate(needs):
            gap = max(0, value - stock[idx])
            if gap:
                ax.text(
                    idx,
                    max(value, stock[idx]) + 0.22,
                    f"+{gap}",
                    ha="center",
                    color=COLORS["gap"],
                    fontsize=8.4,
                )
        ax.set_xticks(x, resources, rotation=35, ha="right")
        ax.set_title(title)
        ax.grid(axis="y")
    axes[0].set_ylabel("Concurrent resource requirement")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
    )
    save(fig, "resource_gaps")


def main() -> None:
    safe_payload_figure()
    convergence_figure()
    gantt(TRANSPORT_SCHEDULE, None, "transport_gantt", "Transport schedule with 23 sorties")
    algorithm_comparison_figure()
    gantt(JOINT_TRANSPORT, RELAY_SCHEDULE, "joint_gantt", "Joint delivery and relay schedule")
    communication_heatmap()
    resource_gap_figure()
    print(f"Wrote English figures to {OUT}")


if __name__ == "__main__":
    main()
