#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""问题四独立修正版。

本程序不读取问题三结果，使用当前可核验的问题二运输方案作为固定时间轴，
重新计算 2 组和 3 组服务区划分下的运输无人机、电池峰值需求。

关键修正：
1. 电池占用区间使用运输方案中的“充电完成时刻_s”；
2. 不把每个架次都强制追加一次完整充电时间；
3. 多服务区运输架次涉及的服务区必须放在同一任务组；
4. 中继资源不在问题三未确认时强行估计，结果只报告运输侧资源。
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


ROOT = Path(__file__).resolve().parent
DEFAULT_ROUTE_FILE = (
    ROOT
    / "问题二_21架次零迟到最短时间"
    / "问题二第一问_CP-SAT强化方案运输架次.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "问题四独立修正版结果"

SERVICE_PATTERN = re.compile(r"S\d{3}")
BOX_PATTERN = re.compile(r"S\d{3}-[A-Z]{3}-\d{2,3}")
SERVICE_AREAS = tuple(f"S{i:03d}" for i in range(1, 16))
KINDS = ("A", "B", "C")

INVENTORY = {
    "运输无人机-A": 4,
    "运输无人机-B": 2,
    "运输无人机-C": 2,
    "共享电池-A": 6,
    "共享电池-B": 4,
    "共享电池-C": 4,
}


@dataclass(frozen=True)
class Route:
    route_id: str
    kind: str
    services: tuple[str, ...]
    start_s: float
    return_s: float
    battery_ready_s: float
    energy_kwh: float
    boxes: int


class DisjointSet:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            nxt = self.parent[value]
            self.parent[value] = root
            value = nxt
        return root

    def union(self, first: str, second: str) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            self.parent[second_root] = first_root


def number(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    if value in (None, ""):
        raise ValueError(f"运输架次缺少字段 {field}：{row}")
    return float(value)


def load_routes(path: Path) -> list[Route]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "架次编号",
        "机型编号",
        "开始时刻_s",
        "访问服务区顺序",
        "返回O01时刻_s",
        "架次能耗_kWh",
        "货箱数量",
        "货箱编号列表",
    }
    if not rows:
        raise ValueError(f"运输架次文件为空：{path}")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"运输架次文件缺少字段：{sorted(missing)}")
    ready_field = (
        "充电完成时刻_s"
        if "充电完成时刻_s" in rows[0]
        else "电池再次可用时刻_s"
    )

    routes: list[Route] = []
    covered: set[str] = set()
    delivered_boxes: list[str] = []
    for row in rows:
        route_id = row["架次编号"]
        kind = row["机型编号"].strip()
        services = tuple(dict.fromkeys(SERVICE_PATTERN.findall(row["访问服务区顺序"])))
        if kind not in KINDS:
            raise ValueError(f"{route_id} 的机型不是 A/B/C：{kind}")
        if not services:
            raise ValueError(f"{route_id} 无法解析服务区：{row['访问服务区顺序']}")
        box_ids = BOX_PATTERN.findall(row["货箱编号列表"])
        declared_boxes = int(float(row["货箱数量"]))
        if len(box_ids) != declared_boxes or len(set(box_ids)) != len(box_ids):
            raise ValueError(
                f"{route_id} 货箱数量与货箱列表不一致："
                f"声明={declared_boxes}，解析={len(box_ids)}"
            )
        delivered_boxes.extend(box_ids)
        start_s = number(row, "开始时刻_s")
        return_s = number(row, "返回O01时刻_s")
        ready_s = number(row, ready_field)
        if not (0 <= start_s <= return_s <= ready_s):
            raise ValueError(
                f"{route_id} 时刻顺序错误：开始={start_s}, 返航={return_s}, "
                f"充电完成={ready_s}"
            )
        covered.update(services)
        routes.append(
            Route(
                route_id=route_id,
                kind=kind,
                services=services,
                start_s=start_s,
                return_s=return_s,
                battery_ready_s=ready_s,
                energy_kwh=number(row, "架次能耗_kWh"),
                boxes=declared_boxes,
            )
        )

    missing_services = sorted(set(SERVICE_AREAS) - covered)
    extra_services = sorted(covered - set(SERVICE_AREAS))
    if missing_services or extra_services:
        raise ValueError(
            f"服务区覆盖异常，缺失={missing_services}，多出={extra_services}"
        )
    if len(delivered_boxes) != 80 or len(set(delivered_boxes)) != 80:
        duplicates = sorted(
            box for box in set(delivered_boxes) if delivered_boxes.count(box) > 1
        )
        raise ValueError(
            f"货箱未精确覆盖 80 箱：总数={len(delivered_boxes)}，"
            f"唯一数={len(set(delivered_boxes))}，重复样例={duplicates[:5]}"
        )
    return routes


def build_atomic_blocks(routes: list[Route]) -> list[tuple[str, ...]]:
    dsu = DisjointSet(SERVICE_AREAS)
    for route in routes:
        for service in route.services[1:]:
            dsu.union(route.services[0], service)
    grouped: dict[str, list[str]] = defaultdict(list)
    for service in SERVICE_AREAS:
        grouped[dsu.find(service)].append(service)
    return sorted(
        (tuple(sorted(group)) for group in grouped.values()),
        key=lambda group: int(group[0][1:]),
    )


def restricted_growth(length: int, groups: int):
    labels = [0] * length

    def visit(index: int, current_max: int):
        if index == length:
            if current_max == groups - 1:
                yield tuple(labels)
            return
        upper = min(current_max + 1, groups - 1)
        for label in range(upper + 1):
            labels[index] = label
            yield from visit(index + 1, max(current_max, label))

    if length == 0:
        return
    yield from visit(1, 0)


def interval_peak(intervals: list[tuple[float, float]]) -> tuple[int, float]:
    if not intervals:
        return 0, 0.0
    times = sorted({time for interval in intervals for time in interval})
    best = (0, times[0])
    for time in times:
        active = sum(start <= time < end for start, end in intervals)
        if active > best[0]:
            best = (active, time)
    return best


def coefficient_of_variation(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return variance**0.5 / mean


def evaluate_partition(
    routes: list[Route],
    blocks: list[tuple[str, ...]],
    labels: tuple[int, ...],
    group_count: int,
) -> dict:
    service_to_group = {
        service: labels[index]
        for index, block in enumerate(blocks)
        for service in block
    }
    group_routes: dict[int, list[Route]] = {group: [] for group in range(group_count)}
    for route in routes:
        route_groups = {service_to_group[service] for service in route.services}
        if len(route_groups) != 1:
            raise ValueError(f"非法分区：架次 {route.route_id} 被拆到多个任务组")
        group_routes[route_groups.pop()].append(route)

    group_demands: list[dict[str, int]] = []
    group_workloads: list[dict[str, float]] = []
    critical: list[dict] = []
    for group in range(group_count):
        resources: dict[str, int] = {}
        workload = {
            "服务区数": sum(len(blocks[i]) for i, label in enumerate(labels) if label == group),
            "运输架次数": len(group_routes[group]),
            "货箱数": sum(route.boxes for route in group_routes[group]),
            "运输能耗_kWh": sum(route.energy_kwh for route in group_routes[group]),
        }
        for kind in KINDS:
            typed = [route for route in group_routes[group] if route.kind == kind]
            drone_intervals = [(route.start_s, route.return_s) for route in typed]
            battery_intervals = [(route.start_s, route.battery_ready_s) for route in typed]
            drone_peak, drone_time = interval_peak(drone_intervals)
            battery_peak, battery_time = interval_peak(battery_intervals)
            resources[f"运输无人机-{kind}"] = drone_peak
            resources[f"共享电池-{kind}"] = battery_peak
            critical.extend(
                [
                    {
                        "任务组": f"G{group + 1}",
                        "资源": f"运输无人机-{kind}",
                        "峰值需求": drone_peak,
                        "峰值时刻_s": drone_time,
                    },
                    {
                        "任务组": f"G{group + 1}",
                        "资源": f"共享电池-{kind}",
                        "峰值需求": battery_peak,
                        "峰值时刻_s": battery_time,
                    },
                ]
            )
        group_demands.append(resources)
        group_workloads.append(workload)

    total = {
        resource: sum(demand[resource] for demand in group_demands)
        for resource in INVENTORY
    }
    baseline = {}
    for resource in INVENTORY:
        intervals = []
        kind = resource.rsplit("-", 1)[1]
        for route in routes:
            if route.kind != kind:
                continue
            if resource.startswith("运输无人机"):
                intervals.append((route.start_s, route.return_s))
            else:
                intervals.append((route.start_s, route.battery_ready_s))
        baseline[resource] = interval_peak(intervals)[0]

    gap = {resource: max(0, total[resource] - INVENTORY[resource]) for resource in INVENTORY}
    redundancy = {resource: total[resource] - baseline[resource] for resource in INVENTORY}
    max_relative_gap = max(
        gap[resource] / INVENTORY[resource] for resource in INVENTORY
    )
    total_relative_gap = sum(gap[resource] / INVENTORY[resource] for resource in INVENTORY)
    total_gap = sum(gap.values())
    workload_imbalance = sum(
        coefficient_of_variation(
            [workload[name] for workload in group_workloads]
        )
        for name in ("运输架次数", "货箱数", "运输能耗_kWh")
    )
    service_imbalance = coefficient_of_variation(
        [workload["服务区数"] for workload in group_workloads]
    )
    return {
        "group_count": group_count,
        "labels": labels,
        "groups": [
            {
                "任务组": f"G{index + 1}",
                "服务区": sorted(
                    service
                    for block_index, block in enumerate(blocks)
                    if labels[block_index] == index
                    for service in block
                ),
                "资源需求": group_demands[index],
                **group_workloads[index],
            }
            for index in range(group_count)
        ],
        "total": total,
        "baseline": baseline,
        "gap": gap,
        "redundancy": redundancy,
        "total_gap": total_gap,
        "max_relative_gap": max_relative_gap,
        "total_relative_gap": total_relative_gap,
        "workload_imbalance": workload_imbalance,
        "service_imbalance": service_imbalance,
        "critical": critical,
    }


def choose(results: list[dict], mode: str) -> dict:
    if mode == "gap_first":
        key = lambda item: (
            item["total_gap"],
            item["max_relative_gap"],
            sum(item["redundancy"].values()),
            item["workload_imbalance"],
            item["service_imbalance"],
        )
    elif mode == "balanced":
        key = lambda item: (
            item["service_imbalance"],
            item["workload_imbalance"],
            item["total_gap"],
            item["max_relative_gap"],
        )
    else:
        raise ValueError(f"未知选择模式：{mode}")
    return min(results, key=key)


def write_outputs(
    output_dir: Path,
    route_file: Path,
    routes: list[Route],
    blocks: list[tuple[str, ...]],
    all_results: list[dict],
    selected: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    block_rows = [
        {"原子块": f"C{index + 1:02d}", "服务区": "、".join(block), "服务区数": len(block)}
        for index, block in enumerate(blocks)
    ]
    with (output_dir / "原子任务块.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=block_rows[0].keys())
        writer.writeheader()
        writer.writerows(block_rows)

    summary_rows = []
    group_rows = []
    for result in selected.values():
        summary_rows.append(
            {
                "分组数": result["group_count"],
                "方案": result["mode"],
                "运输架次": len(routes),
                "总缺口": result["total_gap"],
                "最大相对缺口": result["max_relative_gap"],
                "总相对缺口": result["total_relative_gap"],
                "服务区数量不均衡": result["service_imbalance"],
                "工作量不均衡": result["workload_imbalance"],
                **{f"{key}总需求": value for key, value in result["total"].items()},
            }
        )
        for group in result["groups"]:
            group_rows.append(
                {
                    "分组数": result["group_count"],
                    "方案": result["mode"],
                    "任务组": group["任务组"],
                    "服务区": "、".join(group["服务区"]),
                    "服务区数": group["服务区数"],
                    "运输架次数": group["运输架次数"],
                    "货箱数": group["货箱数"],
                    "运输能耗_kWh": group["运输能耗_kWh"],
                    **group["资源需求"],
                }
            )

    def write_csv(name: str, rows: list[dict]) -> None:
        if not rows:
            return
        with (output_dir / name).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv("方案汇总.csv", summary_rows)
    write_csv("方案分组明细.csv", group_rows)
    write_csv(
        "逐类缺口.csv",
        [
            {
                "分组数": result["group_count"],
                "方案": result["mode"],
                "资源": resource,
                "总需求": result["total"][resource],
                "统一调度峰值": result["baseline"][resource],
                "库存": INVENTORY[resource],
                "缺口": result["gap"][resource],
                "结构性冗余": result["redundancy"][resource],
            }
            for result in selected.values()
            for resource in INVENTORY
        ],
    )
    write_csv("关键峰值.csv", [row for result in selected.values() for row in result["critical"]])
    with (output_dir / "完整结果.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "说明": "本结果独立于问题三，使用问题二固定运输时间轴；中继资源未纳入。",
                "输入运输架次": str(route_file),
                "库存": INVENTORY,
                "候选方案数": len(all_results),
                "selected": selected,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-file", type=Path, default=DEFAULT_ROUTE_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    routes = load_routes(args.route_file)
    blocks = build_atomic_blocks(routes)
    print(f"输入运输架次：{len(routes)}")
    print(f"不可拆分原子任务块：{len(blocks)}")
    for index, block in enumerate(blocks, start=1):
        print(f"  C{index:02d}: {'、'.join(block)}")

    all_results: list[dict] = []
    for group_count in (2, 3):
        for labels in restricted_growth(len(blocks), group_count):
            all_results.append(
                evaluate_partition(routes, blocks, labels, group_count)
            )
    selected: dict[str, dict] = {}
    for group_count in (2, 3):
        subset = [item for item in all_results if item["group_count"] == group_count]
        for mode in ("gap_first", "balanced"):
            result = choose(subset, mode)
            result = dict(result)
            result["mode"] = mode
            selected[f"K{group_count}_{mode}"] = result

    for key, result in selected.items():
        print(f"\n[{key}]")
        print(f"总缺口={result['total_gap']}，最大相对缺口={result['max_relative_gap']:.4f}，"
              f"服务区不均衡={result['service_imbalance']:.4f}，"
              f"工作量不均衡={result['workload_imbalance']:.4f}")
        print("总需求：", result["total"])
        for group in result["groups"]:
            print(
                f"  {group['任务组']} 服务区={'、'.join(group['服务区'])}；"
                f"资源={group['资源需求']}"
            )

    write_outputs(args.output_dir, args.route_file, routes, blocks, all_results, selected)
    print(f"\n候选分区总数：{len(all_results)}")
    print(f"结果目录：{args.output_dir}")
    print("注意：本程序不读取问题三，也不对中继无人机和中继能源组件作正式结论。")


if __name__ == "__main__":
    main()
