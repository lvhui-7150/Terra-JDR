"""问题三高分辨率中继重算：固定运输方案，优化中继覆盖与资源排程。

对运输轨迹按 dt 重建位置，以附录3链路模型计算固定网关直连状态；将每条
轨迹的连续直连缺口合并成服务区间。候选点覆盖一个缺口，要求区间内每个采样
时刻的运输机—中继接入链路及中继—G01回传链路同时可用。CP-SAT 为缺口选点、
分配两架中继机和共享能源组件，并施加飞行及任务-充电区间互斥。

先求有限候选池上的静态集合覆盖位置数，再在硬时限、固定运输组批/机型/航序及
运输资源约束下允许运输架次整体延后，按“加权期望迟到、联合完工时刻、中继能耗、
悬停点数”字典序求解。脚本不重选组批、机型、访问顺序、运输实体机或电池，
因此不是完整的运输-通信全局联合优化；离散采样也不构成连续时间/空间证明。
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
from itertools import combinations
import json
import math
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model


ROOT = Path(__file__).resolve().parents[2]
PHYSICS_PATH = ROOT / "问题三_运输与中继联合调度.py"
DEFAULT_INPUT = ROOT / "_问题三独立模型正式可行结果_修复输出"
DEFAULT_OUTPUT = ROOT / "问题三_高分辨率重算结果"
EPS = 1e-8


def load_physics():
    spec = importlib.util.spec_from_file_location("q3_highres_physics", PHYSICS_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"不能导入物理模型：{PHYSICS_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PHYSICS = load_physics()


@dataclass(frozen=True)
class Gap:
    gap_id: str
    sortie_id: str
    indices: tuple[int, ...]
    start_s: float
    end_s: float


@dataclass(frozen=True)
class Option:
    gap_id: str
    candidate_id: str
    candidate: object
    launch_s: float
    service_start_s: float
    service_end_s: float
    return_s: float
    drone_ready_s: float
    ready_s: float
    energy_kwh: float
    return_soc_pct: float
    covered_gap_ids: tuple[str, ...] = ()


def arguments():
    parser = argparse.ArgumentParser(description="第三问高分辨率通信重算与中继调度")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-step", type=float, default=0.5)
    parser.add_argument("--los-step", type=float, default=30.0)
    parser.add_argument("--candidate-limit", type=int, default=0, help="0为使用全部候选点")
    parser.add_argument("--route-limit", type=int, default=0, help="调试用运输架次上限；0为全部")
    parser.add_argument("--dynamic-anchors", type=int, default=10, help="每个直连缺口沿轨迹生成的局部锚点数")
    parser.add_argument("--max-candidates-per-segment", type=int, default=60)
    parser.add_argument("--relay-drones", type=int, default=2)
    parser.add_argument("--energy-components", type=int, default=6)
    parser.add_argument("--max-route-delay", type=int, default=1800,
                        help="在原起飞时刻基础上允许的最大整体延后秒数")
    parser.add_argument("--solver-seconds", type=float, default=120.0)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def unique_csv(folder: Path, name_fragment: str) -> Path:
    matches = list(folder.glob(f"*{name_fragment}.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"{folder} 下需唯一匹配 *{name_fragment}.csv，找到{len(matches)}个")
    return matches[0]


def candidate_from_row(row):
    return PHYSICS.RelayCandidate(
        candidate_id=str(row["候选点编号"]), longitude=float(row["经度"]), latitude=float(row["纬度"]),
        ground_altitude_m=float(row["地面高程_m"]), agl_m=float(row["离地高度_m"]),
        altitude_m=float(row["绝对海拔_m"]), gateway_link=bool(int(row["回传链路可用"])),
    )


def canonical_candidate(candidate):
    """按经纬度和离地高度归并同一物理悬停点的不同候选标签。"""
    candidate_id = (
        f"SITE-{round(candidate.longitude * 1e6):09d}-"
        f"{round(candidate.latitude * 1e6):09d}-H{round(candidate.agl_m):03d}"
    )
    return PHYSICS.RelayCandidate(
        candidate_id=candidate_id, longitude=candidate.longitude, latitude=candidate.latitude,
        ground_altitude_m=candidate.ground_altitude_m, agl_m=candidate.agl_m,
        altitude_m=candidate.altitude_m, gateway_link=candidate.gateway_link,
    )


def find_gaps(samples: Sequence[object], direct: Sequence[bool], step: float) -> List[Gap]:
    gaps = []
    index = 0
    while index < len(samples):
        if direct[index]:
            index += 1
            continue
        begin = index
        while index + 1 < len(samples) and not direct[index + 1]:
            index += 1
        end = index
        first, last = samples[begin], samples[end]
        gaps.append(Gap(f"{first.sortie_id}-G{len(gaps) + 1:03d}", first.sortie_id,
                        tuple(range(begin, end + 1)), float(first.time_s), float(last.time_s + step)))
        index += 1
    return gaps


def make_option(model, gap: Gap, candidate, samples: Sequence[object]):
    relay_point = model.relay_point(candidate)
    for sample_index in gap.indices:
        if not model.link_available(samples[sample_index].point, relay_point, "transport", "relay"):
            return None
    geometry = model.relay_flight_geometry(candidate)
    relay = model.relay
    launch = gap.start_s - relay["preparation_time_s"] - geometry["out_time_s"] - relay["link_time_s"]
    if launch < -EPS:
        return None
    service_duration = max(0.0, gap.end_s - gap.start_s)
    energy = geometry["flight_energy_kwh"] + (
        relay["hover_power_kw"] + relay["communication_power_kw"]
    ) * service_duration / 3600.0
    if energy > (1.0 - relay["reserve_ratio"]) * relay["usable_energy_kwh"] + EPS:
        return None
    soc = 100.0 * (1.0 - energy / relay["usable_energy_kwh"])
    return_time = gap.end_s + geometry["back_time_s"]
    charge_time = model.charge_time(relay["full_charge_time_s"], soc / 100.0)
    ready_time = max(return_time + relay["turnaround_time_s"], return_time + charge_time)
    return Option(gap.gap_id, candidate.candidate_id, candidate, max(0.0, launch), gap.start_s,
                  gap.end_s, return_time, return_time + relay["turnaround_time_s"], ready_time, energy, soc)


def horizontal_distance(model, point, candidate) -> float:
    return model.horizontal_distance_m(point, model.relay_point(candidate))


def local_candidates(model, gap: Gap, samples: Sequence[object], heights: Sequence[float], anchor_count: int):
    if not gap.indices:
        return []
    count = max(1, min(len(gap.indices), anchor_count))
    indices = np.unique(np.linspace(gap.indices[0], gap.indices[-1], count, dtype=int))
    candidates = []
    for anchor in indices:
        point = samples[int(anchor)].point
        for height in heights:
            candidate = model.make_candidate(
                f"LOCAL-{gap.sortie_id}-{int(anchor):06d}-H{int(height):03d}",
                point.longitude, point.latitude, float(height),
            )
            if candidate is not None:
                candidates.append(canonical_candidate(candidate))
    return candidates


def segment_options(model, gap: Gap, candidates, samples, step: float, max_candidates: int):
    midpoint = samples[gap.indices[len(gap.indices) // 2]].point
    ranked = sorted(candidates, key=lambda item: horizontal_distance(model, midpoint, item))
    if max_candidates > 0:
        ranked = ranked[:max_candidates]
    feasible = [option for candidate in ranked
                if (option := make_option(model, gap, candidate, samples)) is not None]
    feasible.sort(key=lambda item: (item.energy_kwh, item.return_s, item.candidate_id))
    return feasible


def recursively_cover_gap(model, gap: Gap, samples, candidates, step: float, max_candidates: int):
    """若单个悬停点不能覆盖整个直连缺口，则二分时间区间并继续寻找可行覆盖。"""
    options = segment_options(model, gap, candidates, samples, step, max_candidates)
    if options:
        return [(gap, options)]
    if len(gap.indices) <= 1:
        raise RuntimeError(f"单个通信采样时刻也无可行中继覆盖：{gap.gap_id}；需扩充候选空间或重调运输时刻。")
    split = len(gap.indices) // 2
    left_indices = gap.indices[:split]
    right_indices = gap.indices[split:]
    result = []
    for suffix, indices in (("A", left_indices), ("B", right_indices)):
        first, last = samples[indices[0]], samples[indices[-1]]
        child = Gap(
            gap_id=f"{gap.gap_id}-{suffix}", sortie_id=gap.sortie_id, indices=indices,
            start_s=float(first.time_s), end_s=float(last.time_s + step),
        )
        result.extend(recursively_cover_gap(model, child, samples, candidates, step, max_candidates))
    return result


def group_overlapping_demands(model, gaps: Sequence[Gap], options_by_gap: Dict[str, List[Option]]):
    """把同点且时段重叠的缺口合并成一次中继任务，避免一架中继重复飞同一点。

    题面未给中继的并发接入容量，故这里按同一中继可同时保障多个运输机处理；
    如需严格单客户端，应禁用此合并并补充通信容量参数。
    """
    missions = []
    option_by_gap_site = {
        (gap_id, option.candidate_id): option
        for gap_id, options in options_by_gap.items() for option in options
    }
    for gap_id, options in options_by_gap.items():
        missions.extend(options)
    candidate_ids = sorted({option.candidate_id for values in options_by_gap.values() for option in values})
    gap_map = {gap.gap_id: gap for gap in gaps}
    for candidate_id in candidate_ids:
        compatible = [gap for gap in gaps if (gap.gap_id, candidate_id) in option_by_gap_site]
        compatible.sort(key=lambda gap: (gap.start_s, gap.end_s))
        for anchor_index, anchor in enumerate(compatible):
            active = [gap for gap in compatible[anchor_index:]
                      if gap.start_s < anchor.end_s - EPS and gap.end_s > anchor.start_s + EPS]
            active = [gap for gap in active if max(anchor.start_s, gap.start_s) < min(anchor.end_s, gap.end_s) - EPS]
            for group_size in range(2, min(4, len(active)) + 1):
                for selected_gaps in combinations(active, group_size):
                    if selected_gaps[0] is not anchor:
                        continue
                    overlap_start = max(gap.start_s for gap in selected_gaps)
                    overlap_end = min(gap.end_s for gap in selected_gaps)
                    if overlap_start >= overlap_end - EPS:
                        continue
                    start_s = min(gap.start_s for gap in selected_gaps)
                    end_s = max(gap.end_s for gap in selected_gaps)
                    reference = option_by_gap_site[(selected_gaps[0].gap_id, candidate_id)]
                    candidate = reference.candidate
                    geometry = model.relay_flight_geometry(candidate)
                    relay = model.relay
                    launch_s = start_s - relay["preparation_time_s"] - geometry["out_time_s"] - relay["link_time_s"]
                    if launch_s < -EPS:
                        continue
                    energy = geometry["flight_energy_kwh"] + (
                        relay["hover_power_kw"] + relay["communication_power_kw"]
                    ) * (end_s - start_s) / 3600.0
                    energy_limit = (1.0 - relay["reserve_ratio"]) * relay["usable_energy_kwh"]
                    if energy > energy_limit + EPS:
                        continue
                    soc = 100.0 * (1.0 - energy / relay["usable_energy_kwh"])
                    return_s = end_s + geometry["back_time_s"]
                    charge_s = model.charge_time(relay["full_charge_time_s"], soc / 100.0)
                    ready_s = max(return_s + relay["turnaround_time_s"], return_s + charge_s)
                    member_ids = tuple(gap.gap_id for gap in selected_gaps)
                    missions.append(Option(
                        gap_id="MERGED-" + "-".join(member_ids), candidate_id=candidate_id,
                        candidate=candidate, launch_s=max(0.0, launch_s), service_start_s=start_s,
                        service_end_s=end_s, return_s=return_s,
                        drone_ready_s=return_s + relay["turnaround_time_s"], ready_s=ready_s,
                        energy_kwh=energy, return_soc_pct=soc, covered_gap_ids=member_ids,
                    ))
    unique = {}
    for mission in missions:
        covered = mission.covered_gap_ids or (mission.gap_id,)
        key = (mission.candidate_id, tuple(sorted(covered)), round(mission.launch_s, 3), round(mission.service_end_s, 3))
        unique[key] = mission
    return list(unique.values())


def minimum_static_cover(gaps, options_by_gap, limit_s):
    sites = sorted({option.candidate_id for options in options_by_gap.values() for option in options})
    model = cp_model.CpModel()
    used = {site: model.NewBoolVar(f"site_{i}") for i, site in enumerate(sites)}
    site_choices = {site: [] for site in sites}
    for gap_index, gap in enumerate(gaps):
        choices = []
        for option_index, option in enumerate(options_by_gap[gap.gap_id]):
            choose = model.NewBoolVar(f"cover_{gap_index}_{option_index}")
            choices.append(choose)
            site_choices[option.candidate_id].append(choose)
            model.Add(choose <= used[option.candidate_id])
        model.Add(sum(choices) == 1)
    for site in sites:
        model.Add(used[site] <= sum(site_choices[site]))
    count = model.NewIntVar(0, len(sites), "site_count")
    model.Add(count == sum(used.values()))
    model.Minimize(count)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = limit_s
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"集合覆盖模型无解：{solver.StatusName(status)}")
    return solver.Value(count), solver.StatusName(status)


def build_resource_model(gaps, missions, drone_count, component_count, routes, deliveries, max_route_delay):
    model = cp_model.CpModel()
    sites = sorted({mission.candidate_id for mission in missions})
    used = {site: model.NewBoolVar(f"site_{i}") for i, site in enumerate(sites)}
    site_choices = {site: [] for site in sites}
    choices = {}
    drone_intervals = [[] for _ in range(drone_count)]
    component_intervals = [[] for _ in range(component_count)]
    gap_by_id = {gap.gap_id: gap for gap in gaps}
    delay_vars = {}
    delay_bounds = {}
    for _, route in routes.iterrows():
        sortie = str(route["架次编号"])
        route_deliveries = deliveries[deliveries["运输架次编号"].astype(str) == sortie]
        bound = int(max_route_delay)
        for _, delivery in route_deliveries.iterrows():
            box = PHYSICS.Q2.BOX_BY_ID[str(delivery["货箱编号"])]
            actual = float(delivery["实际送达时刻_s"])
            if bool(box.get("is_medical", False)):
                bound = min(bound, max(0, int(math.floor(float(box["desired_time_s"]) - actual))))
            if bool(box.get("is_first_batch", False)):
                bound = min(bound, max(0, int(math.floor(float(box["first_deadline_s"]) - actual))))
        delay_bounds[sortie] = bound
        delay_vars[sortie] = model.NewIntVar(0, bound, f"route_delay_{sortie}")

    tardiness_terms = []
    tardiness_upper = 0
    for _, delivery in deliveries.iterrows():
        sortie = str(delivery["运输架次编号"])
        baseline_late = int(math.ceil(float(delivery["实际送达时刻_s"]) - float(delivery["期望送达时间_s"])))
        upper = max(0, baseline_late + delay_bounds.get(sortie, 0))
        late = model.NewIntVar(0, upper, f"late_{str(delivery['货箱编号']).replace('-', '_')}")
        model.AddMaxEquality(late, [baseline_late + delay_vars[sortie], 0])
        priority = int(round(float(delivery["优先系数"])))
        tardiness_terms.append(priority * late)
        tardiness_upper += priority * upper
    weighted_tardiness = model.NewIntVar(0, max(0, tardiness_upper), "weighted_tardiness")
    model.Add(weighted_tardiness == sum(tardiness_terms))

    transport_drone_intervals = defaultdict(list)
    transport_battery_intervals = defaultdict(list)
    for _, route in routes.iterrows():
        sortie = str(route["架次编号"])
        start = math.floor(float(route["起飞时刻_s"]))
        drone_finish = math.ceil(float(route["返回O01时刻_s"]))
        battery_finish = math.ceil(float(route["电池再次可用时刻_s"]))
        drone_duration = max(1, drone_finish - start)
        battery_duration = max(1, battery_finish - start)
        delay = delay_vars[sortie]
        drone_id, battery_id = str(route["无人机编号"]), str(route["电池编号"])
        transport_drone_intervals[drone_id].append(
            model.NewIntervalVar(start + delay, drone_duration, start + delay + drone_duration,
                                 f"transport_{drone_id}_{sortie}"))
        transport_battery_intervals[battery_id].append(
            model.NewIntervalVar(start + delay, battery_duration, start + delay + battery_duration,
                                 f"transport_battery_{battery_id}_{sortie}"))
    for intervals in list(transport_drone_intervals.values()) + list(transport_battery_intervals.values()):
        model.AddNoOverlap(intervals)

    for mission_index, option in enumerate(missions):
        covered_gaps = option.covered_gap_ids or (option.gap_id,)
        mission_route = gap_by_id[covered_gaps[0]].sortie_id
        delay = delay_vars[mission_route]
        for drone_index in range(drone_count):
            for component_index in range(component_count):
                choose = model.NewBoolVar(f"x_{mission_index}_{drone_index}_{component_index}")
                choices[(mission_index, drone_index, component_index)] = (choose, option)
                site_choices[option.candidate_id].append(choose)
                model.Add(choose <= used[option.candidate_id])
                for covered_gap_id in covered_gaps[1:]:
                    other_route = gap_by_id[covered_gap_id].sortie_id
                    model.Add(delay_vars[other_route] == delay).OnlyEnforceIf(choose)
                start = math.floor(option.launch_s) + delay
                flight_duration = max(1, math.ceil(option.drone_ready_s) - math.floor(option.launch_s))
                energy_duration = max(1, math.ceil(option.ready_s) - math.floor(option.launch_s))
                flight = model.NewOptionalIntervalVar(start, flight_duration, start + flight_duration, choose,
                                                      f"flight_{mission_index}_{drone_index}_{component_index}")
                battery = model.NewOptionalIntervalVar(start, energy_duration, start + energy_duration, choose,
                                                       f"energy_{mission_index}_{drone_index}_{component_index}")
                drone_intervals[drone_index].append(flight)
                component_intervals[component_index].append(battery)
    for gap in gaps:
        covering_choices = [
            choose for (mission_index, _, _), (choose, mission) in choices.items()
            if gap.gap_id in (mission.covered_gap_ids or (mission.gap_id,))
        ]
        model.Add(sum(covering_choices) == 1)
    for site in sites:
        model.Add(used[site] <= sum(site_choices[site]))
    for intervals in drone_intervals + component_intervals:
        model.AddNoOverlap(intervals)

    max_return = max([math.ceil(float(route["返回O01时刻_s"])) + max_route_delay for _, route in routes.iterrows()]
                     + [math.ceil(o.return_s) + max_route_delay for o in missions] + [0])
    completion = model.NewIntVar(0, max_return, "joint_completion")
    for _, route in routes.iterrows():
        sortie = str(route["架次编号"])
        model.Add(completion >= math.ceil(float(route["返回O01时刻_s"])) + delay_vars[sortie])
    for choose, option in choices.values():
        mission_route = gap_by_id[option.covered_gap_ids[0] if option.covered_gap_ids else option.gap_id].sortie_id
        model.Add(completion >= math.ceil(option.return_s) + delay_vars[mission_route]).OnlyEnforceIf(choose)
    site_count = model.NewIntVar(0, len(sites), "used_site_count")
    model.Add(site_count == sum(used.values()))
    scale = 1_000_000
    max_energy = sum(o.energy_kwh for o in missions)
    energy = model.NewIntVar(0, int(math.ceil(max_energy * scale)), "total_relay_energy")
    model.Add(energy == sum(int(round(option.energy_kwh * scale)) * choose for choose, option in choices.values()))
    relay_sorties = model.NewIntVar(0, len(gaps), "relay_sortie_count")
    model.Add(relay_sorties == sum(choose for choose, _ in choices.values()))
    return model, choices, site_count, completion, energy, weighted_tardiness, relay_sorties, delay_vars, delay_bounds


def lexicographic_solve(model, objectives, seconds, workers):
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    solver.parameters.num_search_workers = workers
    solver.parameters.random_seed = 20260924
    records = []
    for label, objective in objectives:
        model.Minimize(objective)
        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            status_name = solver.StatusName(status)
            if status == cp_model.INFEASIBLE:
                conclusion = "已证明当前离散候选与资源约束下不可行"
            elif status == cp_model.UNKNOWN:
                conclusion = "求解时限内未找到可行解，也未证明不可行"
            else:
                conclusion = "求解器未返回可用解"
            raise RuntimeError(f"{conclusion}；阶段={label}，状态={status_name}")
        value = solver.Value(objective)
        records.append({"阶段": label, "状态": solver.StatusName(status), "目标值": value})
        if status != cp_model.OPTIMAL:
            break
        model.Add(objective == value)
    return solver, records


def run():
    args = arguments()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    route_file = unique_csv(input_dir, "运输架次")
    candidate_file = unique_csv(input_dir, "全局中继候选点")
    seed_relay_file = unique_csv(input_dir, "中继架次")
    delivery_file = unique_csv(input_dir, "逐箱交付")
    routes = pd.read_csv(route_file, encoding="utf-8-sig")
    candidate_data = pd.read_csv(candidate_file, encoding="utf-8-sig")
    seed_relay_data = pd.read_csv(seed_relay_file, encoding="utf-8-sig")
    deliveries = pd.read_csv(delivery_file, encoding="utf-8-sig")
    if args.route_limit > 0:
        routes = routes.head(args.route_limit).copy()
        selected_sorties = set(routes["架次编号"].astype(str))
        deliveries = deliveries[deliveries["运输架次编号"].astype(str).isin(selected_sorties)].copy()
    if args.candidate_limit > 0:
        candidate_data = candidate_data.head(args.candidate_limit)
    model = PHYSICS.CommunicationModel(args.sample_step, args.los_step)

    samples_by_sortie, direct_by_sortie, gaps = {}, {}, []
    for _, route in routes.iterrows():
        sortie = str(route["架次编号"])
        samples = model.build_transport_samples(route)
        direct = [model.link_available(sample.point, model.gateway_point(), "transport", "gateway") for sample in samples]
        samples_by_sortie[sortie] = samples
        direct_by_sortie[sortie] = direct
        gaps.extend(find_gaps(samples, direct, args.sample_step))
        print(f"直连扫描 {sortie}: {len(samples)}点，直连缺口{sum(not v for v in direct)}点")
    total_points = sum(len(v) for v in samples_by_sortie.values())
    direct_points = sum(sum(v) for v in direct_by_sortie.values())
    print(f"累计采样{total_points}点，直连率{direct_points / max(total_points, 1):.2%}，缺口区间{len(gaps)}个。")

    candidates = [canonical_candidate(candidate_from_row(row)) for _, row in candidate_data.iterrows()]
    candidates = [
        c for c in candidates
        if c.agl_m <= model.relay["max_hover_agl_m"] + EPS
        and model.link_available(model.relay_point(c), model.gateway_point(), "relay", "gateway")
    ]
    for _, row in seed_relay_data.iterrows():
        seed = PHYSICS.RelayCandidate(
            candidate_id=str(row["悬停点编号"]), longitude=float(row["经度"]), latitude=float(row["纬度"]),
            ground_altitude_m=float(row["地面高程_m"]), agl_m=float(row["悬停离地高度_m"]),
            altitude_m=float(row["悬停绝对海拔_m"]), gateway_link=True,
        )
        if (seed.agl_m <= model.relay["max_hover_agl_m"] + EPS
                and model.link_available(model.relay_point(seed), model.gateway_point(), "relay", "gateway")):
            candidates.append(canonical_candidate(seed))
    covered_gaps: List[Gap] = []
    options_by_gap: Dict[str, List[Option]] = {}
    for gap_index, gap in enumerate(gaps, 1):
        samples = samples_by_sortie[gap.sortie_id]
        dynamic = local_candidates(model, gap, samples, (100.0, 200.0, 300.0), args.dynamic_anchors)
        by_key = {
            (round(c.longitude * 1e6), round(c.latitude * 1e6), round(c.agl_m)): c
            for c in candidates + dynamic
        }
        gap_candidates = list(by_key.values())
        segments = recursively_cover_gap(
            model, gap, samples, gap_candidates, args.sample_step, args.max_candidates_per_segment
        )
        for segment, options in segments:
            covered_gaps.append(segment)
            options_by_gap[segment.gap_id] = options
        if gap_index % 10 == 0 or gap_index == len(gaps):
            print(f"缺口分段与候选覆盖：{gap_index}/{len(gaps)}个原始缺口，生成{len(covered_gaps)}个中继服务段。")
    gaps = covered_gaps

    min_sites, cover_status = minimum_static_cover(gaps, options_by_gap, args.solver_seconds)
    missions = group_overlapping_demands(model, gaps, options_by_gap)
    option_counts = [len(options) for options in options_by_gap.values()]
    diagnostics = {
        "transport_sorties": len(routes),
        "communication_sample_points": total_points,
        "gateway_direct_ratio": direct_points / max(total_points, 1),
        "relay_service_segments": len(gaps),
        "candidate_sites_after_filter": len(candidates),
        "static_set_cover_min_sites": min_sites,
        "static_set_cover_status": cover_status,
        "relay_mission_options": len(missions),
        "min_options_per_segment": min(option_counts, default=0),
        "median_options_per_segment": float(np.median(option_counts)) if option_counts else 0,
        "max_options_per_segment": max(option_counts, default=0),
        "relay_drones": args.relay_drones,
        "energy_components": args.energy_components,
        "max_route_delay_s": args.max_route_delay,
        "sample_step_s": args.sample_step,
        "los_step_m": args.los_step,
    }
    (output_dir / "问题三_求解前诊断.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"静态集合覆盖下界={min_sites}个位置（{cover_status}）；联合候选任务={len(missions)}。")
    print("运输架次允许在硬截止余量内整体延后，以缓解中继资源冲突。")
    resource_model, choices, site_count, relay_completion, total_energy, weighted_tardiness, relay_sorties, delay_vars, delay_bounds = build_resource_model(
        gaps, missions, args.relay_drones, args.energy_components, routes, deliveries, args.max_route_delay
    )
    try:
        solver, solve_records = lexicographic_solve(
            resource_model,
            [("加权期望迟到", weighted_tardiness), ("联合完工时刻", relay_completion),
             ("中继能耗", total_energy), ("中继架次数", relay_sorties), ("悬停点数", site_count)],
            args.solver_seconds, args.workers,
        )
    except RuntimeError as error:
        diagnostics["resource_solve_status"] = str(error)
        diagnostics["resource_solve_conclusion"] = (
            "infeasible" if "已证明" in str(error) else "not_solved_within_limit"
        )
        (output_dir / "问题三_求解前诊断.json").write_text(
            json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
        raise
    route_delays = {sortie: solver.Value(variable) for sortie, variable in delay_vars.items()}
    chosen = []
    for (mission_index, drone_index, component_index), (var, option) in choices.items():
        if solver.BooleanValue(var):
            chosen.append((option, drone_index, component_index))
    chosen_by_gap = {}
    for option, drone_index, component_index in chosen:
        for gap_id in (option.covered_gap_ids or (option.gap_id,)):
            chosen_by_gap[gap_id] = (option, drone_index, component_index)
    gap_by_id = {gap.gap_id: gap for gap in gaps}
    gap_at_sample = {(gap.sortie_id, index): gap for gap in gaps for index in gap.indices}

    relay_rows, communication_rows = [], []
    for option, drone_index, component_index in chosen:
        covered_gaps = option.covered_gap_ids or (option.gap_id,)
        covered_sorties = sorted({gap_by_id[gap_id].sortie_id for gap_id in covered_gaps})
        shift = route_delays[covered_sorties[0]]
        relay_rows.append({
            "通信缺口编号": "、".join(covered_gaps), "运输架次编号": covered_sorties[0],
            "覆盖运输架次": "、".join(covered_sorties),
            "中继无人机编号": f"R{drone_index + 1:02d}", "能源组件编号": f"R-ENERGY-{component_index + 1:02d}",
            "候选点编号": option.candidate_id, "经度": option.candidate.longitude, "纬度": option.candidate.latitude,
            "地面高程_m": option.candidate.ground_altitude_m, "悬停离地高度_m": option.candidate.agl_m,
            "悬停绝对海拔_m": option.candidate.altitude_m, "中继起飞时刻_s": option.launch_s + shift,
            "开始服务时刻_s": option.service_start_s + shift, "结束服务时刻_s": option.service_end_s + shift,
            "返回O01时刻_s": option.return_s + shift, "无人机再次可用时刻_s": option.drone_ready_s + shift,
            "能源组件再次可用时刻_s": option.ready_s + shift,
            "中继能耗_kWh": option.energy_kwh, "返航SOC_pct": option.return_soc_pct,
        })

    for sortie, samples in samples_by_sortie.items():
        for index, sample in enumerate(samples):
            direct = direct_by_sortie[sortie][index]
            row = {
                "运输架次编号": sortie, "时刻_s": sample.time_s + route_delays[sortie], "阶段": sample.phase,
                "经度": sample.point.longitude, "纬度": sample.point.latitude,
                "绝对高度_m": sample.point.altitude_m, "固定网关直连可用": int(direct),
                "通信方式": "直连" if direct else "通信中断", "中继候选点": "", "中继无人机": "",
                "中继接入链路可用": "", "中继回传链路可用": "", "通信可用": int(direct),
            }
            gap = gap_at_sample.get((sortie, index))
            if gap is not None:
                option, drone_index, _ = chosen_by_gap[gap.gap_id]
                relay_point = model.relay_point(option.candidate)
                access_available = model.link_available(sample.point, relay_point, "transport", "relay")
                backhaul_available = model.link_available(relay_point, model.gateway_point(), "relay", "gateway")
                available = access_available and backhaul_available
                row.update({"通信方式": "中继" if available else "通信中断",
                            "中继候选点": option.candidate_id, "中继无人机": f"R{drone_index + 1:02d}",
                            "中继接入链路可用": int(access_available),
                            "中继回传链路可用": int(backhaul_available),
                            "通信可用": int(available)})
            communication_rows.append(row)

    communication = pd.DataFrame(communication_rows)
    relays = pd.DataFrame(relay_rows)
    adjusted_routes = routes.copy()
    adjusted_routes["整体延后_s"] = adjusted_routes["架次编号"].astype(str).map(route_delays).fillna(0).astype(int)
    for column in ("开始时刻_s", "起飞时刻_s", "返回O01时刻_s", "电池再次可用时刻_s"):
        if column in adjusted_routes.columns:
            adjusted_routes[column] = adjusted_routes[column] + adjusted_routes["整体延后_s"]
    adjusted_deliveries = deliveries.copy()
    adjusted_deliveries["整体延后_s"] = adjusted_deliveries["运输架次编号"].astype(str).map(route_delays).fillna(0).astype(int)
    adjusted_deliveries["调整后实际送达时刻_s"] = (
        adjusted_deliveries["实际送达时刻_s"] + adjusted_deliveries["整体延后_s"]
    )
    adjusted_deliveries["调整后期望迟到_s"] = np.maximum(
        0.0, adjusted_deliveries["调整后实际送达时刻_s"] - adjusted_deliveries["期望送达时间_s"]
    )
    adjusted_deliveries["硬时限是否满足"] = adjusted_deliveries.apply(
        lambda row: "是" if (
            pd.isna(PHYSICS.Q2.BOX_BY_ID[str(row["货箱编号"])].get("hard_deadline_s"))
            or row["调整后实际送达时刻_s"] <=
            float(PHYSICS.Q2.BOX_BY_ID[str(row["货箱编号"])]["hard_deadline_s"]) + EPS
        ) else "否", axis=1
    )
    outage_points = int((communication["通信可用"] == 0).sum())
    transport_end = float(adjusted_routes["返回O01时刻_s"].max()) if not adjusted_routes.empty else 0.0
    relay_end = float(relays["返回O01时刻_s"].max()) if not relays.empty else 0.0
    used_sites = int(relays["候选点编号"].nunique()) if not relays.empty else 0
    metrics = [{
        "时间采样步长_s": args.sample_step, "视线DEM采样步长_m": args.los_step,
        "运输架次数": len(routes), "通信采样点数": total_points, "直连可用点数": direct_points,
        "直连比例": direct_points / max(total_points, 1), "直连缺口区间数": len(gaps),
        "静态集合覆盖最少位置数": min_sites, "集合覆盖状态": cover_status,
        "资源可行方案使用位置数": used_sites, "中继架次": len(relays),
        "中继无人机数": int(relays["中继无人机编号"].nunique()) if not relays.empty else 0,
        "联合完工时间_s": max(transport_end, relay_end),
        "加权期望迟到_优先系数秒": float((adjusted_deliveries["优先系数"] * adjusted_deliveries["调整后期望迟到_s"]).sum()),
        "有延后运输架次数": sum(value > 0 for value in route_delays.values()),
        "最大运输架次延后_s": max(route_delays.values(), default=0),
        "中继总能耗_kWh": float(relays["中继能耗_kWh"].sum()) if not relays.empty else 0.0,
        "通信中断采样点数": outage_points,
        "字典序求解状态": json.dumps(solve_records, ensure_ascii=False),
        "最优性口径": "固定运输计划、离散时空采样和有限候选点集；请结合每阶段状态判断最优性",
    }]
    checks = [("所有通信采样点均可用", outage_points == 0),
              ("运输货箱硬时限满足", bool((adjusted_deliveries["硬时限是否满足"] == "是").all())),
              ("返航SOC不低于20%", relays.empty or bool((relays["返航SOC_pct"] >= 20 - EPS).all())),
              ("悬停离地高度不超限", relays.empty or bool((relays["悬停离地高度_m"] <= model.relay["max_hover_agl_m"] + EPS).all()))]
    for resource, end_column, label in (
        ("中继无人机编号", "无人机再次可用时刻_s", "中继实体机任务互斥"),
        ("能源组件编号", "能源组件再次可用时刻_s", "能源组件任务与充电互斥"),
    ):
        feasible = True
        if not relays.empty:
            for _, group in relays.sort_values("中继起飞时刻_s").groupby(resource):
                starts = group["中继起飞时刻_s"].to_numpy()
                ends = group[end_column].to_numpy()
                feasible &= len(starts) < 2 or bool(np.all(starts[1:] >= ends[:-1] - EPS))
        checks.append((label, feasible))

    communication.to_csv(output_dir / "问题三_逐时刻通信校验.csv", index=False, encoding="utf-8-sig")
    relays.to_csv(output_dir / "问题三_重算中继架次.csv", index=False, encoding="utf-8-sig")
    adjusted_routes.to_csv(output_dir / "问题三_调整后运输架次.csv", index=False, encoding="utf-8-sig")
    adjusted_deliveries.to_csv(output_dir / "问题三_调整后逐箱交付.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(metrics).to_csv(output_dir / "问题三_高分辨率指标汇总.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"检查项目": name, "是否通过": "是" if passed else "否"} for name, passed in checks]).to_csv(
        output_dir / "问题三_高分辨率约束检查.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(solve_records).to_csv(output_dir / "问题三_中继求解阶段.csv", index=False, encoding="utf-8-sig")
    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    config = {
        "input_dir": str(input_dir), "output_dir": str(output_dir), "sample_step_s": args.sample_step,
        "los_step_m": args.los_step, "candidate_pool_size": len(candidates),
        "relay_drones": args.relay_drones, "energy_components": args.energy_components,
        "max_route_delay_s": args.max_route_delay,
        "route_limit": args.route_limit,
        "dynamic_anchors": args.dynamic_anchors,
        "max_candidates_per_segment": args.max_candidates_per_segment,
        "solver_seconds_per_stage": args.solver_seconds, "workers": args.workers,
        "route_file": str(route_file), "candidate_file": str(candidate_file),
        "delivery_file": str(delivery_file), "seed_relay_file": str(seed_relay_file),
        "random_seed": 20260924,
        "input_sha256": {str(path): sha256(path) for path in
                         (route_file, candidate_file, delivery_file, seed_relay_file)},
        "limitations": ["固定输入既有运输组批、机型、航序、运输实体机和电池；仅允许运输架次整体延后。",
                        "最少悬停点数针对有限候选池和离散采样，不是连续空间下界。",
                        f"{args.sample_step:g}秒时间采样和{args.los_step:g}米DEM视线离散检查不构成连续时间严格证明。"],
    }
    (output_dir / "问题三_高分辨率运行配置.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完成：直连率{direct_points / max(total_points, 1):.2%}，静态覆盖位置下界{min_sites}，资源可行位置{used_sites}。")
    print(f"中继架次{len(relays)}，通信中断采样点{outage_points}，联合完工{max(transport_end, relay_end):.2f}s。")
    print(f"输出目录：{output_dir}")


if __name__ == "__main__":
    run()
