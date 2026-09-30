"""问题三独立求解：通信感知 ALNS + CP-SAT 联合优化。

本程序不读取问题二的任何运输方案或结果文件，而是直接从原始附件重新决定：
1. 货箱组批、服务区访问顺序和运输机型；
2. 实体运输无人机、共享电池和架次开始时刻；
3. 中继悬停位置、高度、服务时段、中继无人机和共享能源组件；
4. 配送及时性、联合完工时间、联合能耗和两类架次的字典序目标。

算法由两部分组成：
- C-ALNS：从原始货箱出发，使用自适应合并、移动、交换和拆分算子生成候选组批；
- CP-SAT：在通信可行候选架次上，同时安排运输资源和中继资源。

CP-SAT 的 OPTIMAL 仅表示当前有限候选池内最优，不代表 80 箱原问题全局最优。
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
from time import perf_counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model


ROOT = Path(__file__).resolve().parent
Q2_PLANNER_FILE = ROOT / "问题二_第一问_CP-SAT强化精确求解.py"
Q3_PHYSICS_FILE = ROOT / "问题三_运输与中继联合调度.py"
OUTPUT_DIR = ROOT / "问题三独立联合优化结果"
DEFAULT_SEED_ROUTE_FILE = ROOT / "问题二_21架次零迟到最短时间" / "问题二第一问_CP-SAT强化方案运输架次.csv"
TOLERANCE = 1e-8
ENERGY_SCALE = 1_000_000
EARTH_RADIUS_M = 6_371_000.0


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PLANNER = load_module("q3_independent_candidate_planner", Q2_PLANNER_FILE)
PHYSICS = load_module("q3_independent_communication_physics", Q3_PHYSICS_FILE)


@dataclass(frozen=True)
class RelayOption:
    option_id: str
    candidate_id: str
    longitude: float
    latitude: float
    ground_altitude_m: float
    agl_m: float
    altitude_m: float
    launch_offset_s: float
    service_start_offset_s: float
    service_end_offset_s: float
    return_offset_s: float
    ready_offset_s: float
    energy_kwh: float
    return_soc_pct: float


@dataclass(frozen=True)
class GapProfile:
    gap_id: str
    start_index: int
    end_index: int
    start_offset_s: float
    end_offset_s: float
    options: Tuple[RelayOption, ...]


@dataclass(frozen=True)
class JointCandidate:
    candidate_id: str
    transport: object
    samples: Tuple[object, ...]
    direct_flags: Tuple[bool, ...]
    gaps: Tuple[GapProfile, ...]


@dataclass
class JointSolution:
    solver: object
    variables: dict
    stage_log: List[dict]
    objective_values: Tuple[int, int, int, int]


def initialize_physics(args: argparse.Namespace):
    """初始化统一物理模型，并把候选生成器绑定到同一份原始数据。"""
    model = PHYSICS.CommunicationModel(args.sample_step, args.los_step)
    PLANNER.BASE = PHYSICS.Q2
    PLANNER.BASE.feasible_route_options.cache_clear()
    PLANNER.candidate_route_options.cache_clear()
    PLANNER.ENABLE_ROUTE_ORDER_VARIANTS = True
    PLANNER.MAX_ROUTE_ORDERS_PER_DRONE = args.max_orders_per_drone
    PLANNER.MAX_ROUTE_GROUPS = args.max_route_groups
    return model, PHYSICS.Q2


def singleton_partition(base) -> Tuple[Tuple[str, ...], ...]:
    return base.normalize_partition((str(box_id),) for box_id in base.BOXES["box_id"])


def route_proxy(route: Tuple[str, ...]) -> float:
    options = PLANNER.candidate_route_options(tuple(sorted(route)))
    if not options:
        return 1e30
    best = min(
        PLANNER.route_option_key_at_start(option, route, 0.0)
        for option in options
    )
    hard_count, hard_lateness, tardiness, completion, energy = best
    return hard_count * 1e14 + hard_lateness * 1e9 + tardiness * 1e4 + completion * 10.0 + energy * 1000.0


@lru_cache(maxsize=100_000)
def partition_proxy(partition: Tuple[Tuple[str, ...], ...]) -> float:
    return sum(route_proxy(route) for route in partition) + 300.0 * len(partition)


def valid_partition(base, routes: Sequence[Sequence[str]]) -> Optional[Tuple[Tuple[str, ...], ...]]:
    partition = base.normalize_partition(route for route in routes if route)
    flattened = [box_id for route in partition for box_id in route]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(base.BOX_BY_ID):
        return None
    if all(PLANNER.candidate_route_options(route) for route in partition):
        return partition
    return None


def load_seed_partition(base, route_file: str) -> Optional[Tuple[Tuple[str, ...], ...]]:
    """读取一组已验证的高质量运输组批，仅作为候选种子，不固定其时刻和资源。"""
    if not route_file:
        return None
    path = Path(route_file).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    if not path.exists():
        print(f"提示：候选种子文件不存在，继续独立生成：{path}")
        return None
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if "货箱编号列表" not in frame.columns:
        raise ValueError(f"候选种子文件缺少货箱编号列表列：{path}")
    routes = [str(value).split("、") for value in frame["货箱编号列表"]]
    partition = valid_partition(base, routes)
    if partition is None:
        raise ValueError(f"候选种子文件不是覆盖全部货箱的可行组批：{path}")
    return partition


def apply_alns_operator(base, partition, operator: str, rng: random.Random):
    routes = [list(route) for route in partition]
    if operator == "merge" and len(routes) >= 2:
        first, second = rng.sample(range(len(routes)), 2)
        merged = routes[first] + routes[second]
        candidate = [route for index, route in enumerate(routes) if index not in {first, second}]
        candidate.append(merged)
        return valid_partition(base, candidate)
    if operator == "relocate" and routes:
        source = rng.randrange(len(routes))
        box_id = rng.choice(routes[source])
        destination = rng.choice(list(range(len(routes))) + [None])
        if destination == source:
            return None
        routes[source].remove(box_id)
        if destination is None:
            routes.append([box_id])
        else:
            routes[destination].append(box_id)
        return valid_partition(base, routes)
    if operator == "swap" and len(routes) >= 2:
        first, second = rng.sample(range(len(routes)), 2)
        first_box = rng.choice(routes[first])
        second_box = rng.choice(routes[second])
        routes[first].remove(first_box)
        routes[second].remove(second_box)
        routes[first].append(second_box)
        routes[second].append(first_box)
        return valid_partition(base, routes)
    if operator == "split":
        choices = [index for index, route in enumerate(routes) if len(route) >= 2]
        if not choices:
            return None
        index = rng.choice(choices)
        route = routes.pop(index)
        rng.shuffle(route)
        cut = rng.randrange(1, len(route))
        routes.extend([route[:cut], route[cut:]])
        return valid_partition(base, routes)
    return None


def generate_route_groups(base, args: argparse.Namespace):
    """不读取问题二结果，从原始货箱生成独立候选组批池。"""
    pool = PLANNER.RouteGroupPool()
    singletons = singleton_partition(base)
    pool.add_partition(singletons, mandatory=True)
    seed_partition = load_seed_partition(base, args.seed_route_file)
    seeds = [singletons, base.greedy_partition("deadline"), base.greedy_partition("mass"), base.greedy_partition("service")]
    if seed_partition is not None:
        seeds.insert(0, seed_partition)
        print(f"已加入高质量运输组批种子：{len(seed_partition)} 架次。")
    seeds = list(dict.fromkeys(seeds))
    for partition in seeds:
        pool.add_partition(partition, mandatory=True)

    boxes_by_service: Dict[str, List[str]] = defaultdict(list)
    for box_id, record in base.BOX_BY_ID.items():
        boxes_by_service[str(record["service_id"])].append(box_id)
    pair_count = 0
    for box_ids in boxes_by_service.values():
        for first_index in range(len(box_ids)):
            for second_index in range(first_index + 1, len(box_ids)):
                if pair_count >= args.pair_limit:
                    break
                pair_count += int(pool.add_group((box_ids[first_index], box_ids[second_index])))
            if pair_count >= args.pair_limit:
                break
        if pair_count >= args.pair_limit:
            break

    rng = random.Random(args.seed)
    operators = ("merge", "relocate", "swap", "split")
    weights = {operator: 1.0 for operator in operators}
    scores = {operator: 0.0 for operator in operators}
    uses = {operator: 0 for operator in operators}
    current = min(seeds, key=partition_proxy)
    current_score = partition_proxy(current)
    best, best_score = current, current_score
    temperature = max(1.0, abs(current_score) * 0.02)
    history: List[dict] = []

    for step in range(1, args.alns_steps + 1):
        operator = rng.choices(operators, weights=[weights[name] for name in operators], k=1)[0]
        uses[operator] += 1
        candidate = apply_alns_operator(base, current, operator, rng)
        if candidate is None:
            scores[operator] += 0.1
            continue
        candidate_score = partition_proxy(candidate)
        delta = candidate_score - current_score
        accepted = delta <= 0 or rng.random() < math.exp(-delta / max(temperature, 1e-9))
        added = pool.add_partition(candidate)
        reward = 0.5 + 0.05 * added
        if candidate_score < best_score - TOLERANCE:
            best, best_score = candidate, candidate_score
            reward = 8.0 + 0.1 * added
        elif accepted:
            reward = 2.0 + 0.05 * added
        scores[operator] += reward
        if accepted:
            current, current_score = candidate, candidate_score
        if step % args.alns_segment == 0:
            for name in operators:
                performance = scores[name] / max(1, uses[name])
                weights[name] = 0.75 * weights[name] + 0.25 * max(0.1, performance)
                scores[name] = 0.0
                uses[name] = 0
            history.append({"迭代": step, "候选组批数": len(pool.groups), "当前架次数": len(current), "最好架次数": len(best), "最好代理目标": best_score, **{f"权重_{name}": weights[name] for name in operators}})
        temperature *= args.cooling
        if step % max(10, args.alns_segment * 2) == 0:
            restart = rng.choice(seeds + [best])
            current, current_score = restart, partition_proxy(restart)

    pool.add_partition(best, mandatory=True)
    pool.prune()
    return pool, history, seeds


def candidate_rank(candidate) -> Tuple[float, ...]:
    key = PLANNER.route_option_key_at_start(candidate.option, candidate.box_ids, 0.0)
    return key[:3] + (-float(len(candidate.box_ids)), key[3], key[4])


def select_transport_candidates(pool, args: argparse.Namespace) -> List[object]:
    all_candidates = [
        candidate
        for candidate in pool.candidates()
        if PLANNER.route_option_key_at_start(candidate.option, candidate.box_ids, 0.0)[0] == 0
    ]
    by_group: Dict[Tuple[str, ...], List[object]] = defaultdict(list)
    for candidate in all_candidates:
        by_group[candidate.box_ids].append(candidate)
    mandatory_candidates: List[object] = []
    for group in sorted(pool.mandatory):
        mandatory_candidates.extend(sorted(by_group[group], key=candidate_rank)[: args.mandatory_options_per_group])

    singleton_by_box: Dict[str, List[object]] = defaultdict(list)
    for candidate in all_candidates:
        if len(candidate.box_ids) == 1:
            singleton_by_box[candidate.box_ids[0]].append(candidate)
    singleton_candidates: List[object] = []
    for box_id in sorted(singleton_by_box):
        options = sorted(singleton_by_box[box_id], key=candidate_rank)
        singleton_candidates.extend(options[: args.singleton_options_per_box])
    mandatory_ids = {candidate.candidate_id for candidate in mandatory_candidates}
    singleton_ids = {candidate.candidate_id for candidate in singleton_candidates}
    other_candidates = [candidate for candidate in all_candidates if candidate.candidate_id not in mandatory_ids | singleton_ids]
    singleton_candidates.sort(key=candidate_rank)
    other_candidates.sort(key=candidate_rank)
    required = list({candidate.candidate_id: candidate for candidate in mandatory_candidates + singleton_candidates}.values())
    required.sort(key=candidate_rank)
    limit = max(args.max_route_candidates, len(required))
    return required + other_candidates[: max(0, limit - len(required))]


def xy_to_lonlat(x_m: float, y_m: float, center_lon: float, center_lat: float) -> Tuple[float, float]:
    latitude = center_lat + math.degrees(y_m / EARTH_RADIUS_M)
    longitude = center_lon + math.degrees(x_m / (EARTH_RADIUS_M * math.cos(math.radians(center_lat))))
    return longitude, latitude


def build_global_relay_candidates(model, args: argparse.Namespace) -> List[object]:
    """生成与任何既有运输方案无关的全局悬停点集合。"""
    center_lon = float(model.center["longitude"])
    center_lat = float(model.center["latitude"])
    base_points: Dict[Tuple[int, int], Tuple[float, float]] = {}

    def add_point(longitude: float, latitude: float) -> None:
        if not (model.longitude.min() <= longitude <= model.longitude.max()):
            return
        if not (model.latitude.min() <= latitude <= model.latitude.max()):
            return
        base_points.setdefault((round(longitude * 1e6), round(latitude * 1e6)), (longitude, latitude))

    for node_id, row in model.node_map.items():
        if node_id != "O01":
            add_point(float(row["longitude"]), float(row["latitude"]))
            for ratio in (0.25, 0.50, 0.75):
                add_point(
                    center_lon + ratio * (float(row["longitude"]) - center_lon),
                    center_lat + ratio * (float(row["latitude"]) - center_lat),
                )

    lon_bounds = np.array([model.longitude.min(), model.longitude.max()])
    lat_bounds = np.array([model.latitude.min(), model.latitude.max()])
    x_bounds, y_bounds = model.Q1.local_xy(lon_bounds, lat_bounds, center_lon, center_lat) if hasattr(model, "Q1") else PHYSICS.Q1.local_xy(lon_bounds, lat_bounds, center_lon, center_lat)
    x_values = np.arange(math.floor(x_bounds.min() / args.relay_grid) * args.relay_grid, x_bounds.max() + args.relay_grid, args.relay_grid)
    y_values = np.arange(math.floor(y_bounds.min() / args.relay_grid) * args.relay_grid, y_bounds.max() + args.relay_grid, args.relay_grid)
    for x_m in x_values:
        for y_m in y_values:
            longitude, latitude = xy_to_lonlat(float(x_m), float(y_m), center_lon, center_lat)
            add_point(longitude, latitude)

    ordered_points = list(base_points.values())[: args.max_relay_base_points]
    candidates: List[object] = []
    for point_index, (longitude, latitude) in enumerate(ordered_points, 1):
        for height in args.heights:
            candidate = model.make_candidate(
                f"GLOBAL-{point_index:03d}-H{int(round(height)):03d}",
                longitude,
                latitude,
                float(height),
            )
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def build_relative_samples(model, candidate) -> Tuple[object, ...]:
    option = candidate.option
    row = pd.Series(
        {
            "架次编号": candidate.candidate_id,
            "机型编号": candidate.drone_type,
            "起飞时刻_s": option.takeoff_offset_s,
            "访问服务区顺序": "→".join(candidate.service_order),
            "货箱编号列表": "、".join(candidate.box_ids),
            "返回O01时刻_s": option.duration_s,
        }
    )
    return tuple(model.build_transport_samples(row))


def direct_flags_and_gaps(model, samples: Sequence[object]):
    gateway = model.gateway_point()
    flags = tuple(
        bool(model.link_available(sample.point, gateway, "transport", "gateway"))
        for sample in samples
    )
    gaps: List[Tuple[int, int, float, float]] = []
    index = 0
    while index < len(samples):
        if flags[index]:
            index += 1
            continue
        begin = index
        while index + 1 < len(samples) and not flags[index + 1]:
            index += 1
        end = index
        gaps.append((begin, end, float(samples[begin].time_s), float(samples[end].time_s + model.sample_step_s)))
        index += 1
    return flags, gaps


def make_dynamic_candidates(model, samples: Sequence[object], begin: int, end: int, args: argparse.Namespace) -> List[object]:
    count = max(1, min(args.dynamic_points_per_gap, end - begin + 1))
    indices = np.linspace(begin, end, count, dtype=int)
    candidates: List[object] = []
    seen: set[Tuple[int, int, int]] = set()
    for sample_index in indices:
        sample = samples[int(sample_index)]
        for height in args.heights:
            key = (round(sample.point.longitude * 1e6), round(sample.point.latitude * 1e6), int(round(height)))
            if key in seen:
                continue
            seen.add(key)
            candidate = model.make_candidate(
                f"DYN-{key[0]}-{key[1]}-H{key[2]:03d}",
                sample.point.longitude,
                sample.point.latitude,
                float(height),
            )
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def relay_option_from_candidate(model, gap_id: str, start_s: float, end_s: float, candidate) -> Optional[RelayOption]:
    geometry = model.relay_flight_geometry(candidate)
    relay = model.relay
    preflight = relay["preparation_time_s"] + geometry["out_time_s"] + relay["link_time_s"]
    launch_offset = start_s - preflight
    service_duration = max(0.0, end_s - start_s)
    energy = geometry["flight_energy_kwh"] + (relay["hover_power_kw"] + relay["communication_power_kw"]) * service_duration / 3600.0
    energy_limit = (1.0 - relay["reserve_ratio"]) * relay["usable_energy_kwh"]
    if energy > energy_limit + TOLERANCE:
        return None
    return_soc = 100.0 * (1.0 - energy / relay["usable_energy_kwh"])
    return_offset = end_s + geometry["back_time_s"]
    charge_time = model.charge_time(relay["full_charge_time_s"], return_soc / 100.0)
    ready_offset = max(return_offset + relay["turnaround_time_s"], return_offset + charge_time)
    return RelayOption(
        option_id=f"{gap_id}-{candidate.candidate_id}",
        candidate_id=candidate.candidate_id,
        longitude=candidate.longitude,
        latitude=candidate.latitude,
        ground_altitude_m=candidate.ground_altitude_m,
        agl_m=candidate.agl_m,
        altitude_m=candidate.altitude_m,
        launch_offset_s=launch_offset,
        service_start_offset_s=start_s,
        service_end_offset_s=end_s,
        return_offset_s=return_offset,
        ready_offset_s=ready_offset,
        energy_kwh=energy,
        return_soc_pct=return_soc,
    )


def horizontal_distance_m(model, point, candidate) -> float:
    x, y = PHYSICS.Q1.local_xy(
        np.array([point.longitude, candidate.longitude]),
        np.array([point.latitude, candidate.latitude]),
        float(model.center["longitude"]),
        float(model.center["latitude"]),
    )
    return float(math.hypot(x[1] - x[0], y[1] - y[0]))


def relay_options_for_gap(
    model,
    joint_id: str,
    samples: Sequence[object],
    begin: int,
    end: int,
    start_s: float,
    end_s: float,
    global_candidates: Sequence[object],
    args: argparse.Namespace,
) -> Tuple[RelayOption, ...]:
    dynamic = make_dynamic_candidates(model, samples, begin, end, args)
    midpoint = samples[(begin + end) // 2].point
    merged: Dict[Tuple[int, int, int], object] = {}
    for candidate in list(dynamic) + list(global_candidates):
        key = (round(candidate.longitude * 1e6), round(candidate.latitude * 1e6), int(round(candidate.agl_m)))
        merged.setdefault(key, candidate)
    ranked = sorted(merged.values(), key=lambda item: (horizontal_distance_m(model, midpoint, item), -item.agl_m))
    options: List[RelayOption] = []
    relay_link_cache: Dict[Tuple[int, str], bool] = {}
    for candidate in ranked[: args.relay_test_limit]:
        relay_point = model.relay_point(candidate)
        covered = True
        for sample_index in range(begin, end + 1):
            cache_key = (sample_index, candidate.candidate_id)
            available = relay_link_cache.get(cache_key)
            if available is None:
                available = bool(model.link_available(samples[sample_index].point, relay_point, "transport", "relay"))
                relay_link_cache[cache_key] = available
            if not available:
                covered = False
                break
        if not covered:
            continue
        option = relay_option_from_candidate(model, joint_id, start_s, end_s, candidate)
        if option is not None:
            options.append(option)
    options.sort(key=lambda item: (item.energy_kwh, item.ready_offset_s, item.agl_m))
    return tuple(options[: args.max_relay_options])


def merged_relay_option(model, gap_id: str, start_s: float, end_s: float, option: RelayOption) -> Optional[RelayOption]:
    candidate = PHYSICS.RelayCandidate(
        candidate_id=option.candidate_id,
        longitude=option.longitude,
        latitude=option.latitude,
        ground_altitude_m=option.ground_altitude_m,
        agl_m=option.agl_m,
        altitude_m=option.altitude_m,
        gateway_link=True,
    )
    return relay_option_from_candidate(model, gap_id, start_s, end_s, candidate)


def merge_gap_profiles(model, joint_id: str, gaps: Sequence[GapProfile], args: argparse.Namespace) -> Tuple[GapProfile, ...]:
    """把同一悬停点可连续覆盖的多个缺口合并为一个物理中继架次。"""
    if len(gaps) <= 1:
        return tuple(gaps)
    block_options: Dict[Tuple[int, int], Tuple[RelayOption, ...]] = {}
    for begin in range(len(gaps)):
        common: Dict[str, RelayOption] = {option.candidate_id: option for option in gaps[begin].options}
        for end in range(begin, len(gaps)):
            if end > begin:
                available = {option.candidate_id: option for option in gaps[end].options}
                common = {candidate_id: option for candidate_id, option in common.items() if candidate_id in available}
            if not common:
                break
            gap_id = f"{joint_id}-M{begin + 1:02d}-{end + 1:02d}"
            merged_options = [
                merged_relay_option(model, gap_id, gaps[begin].start_offset_s, gaps[end].end_offset_s, option)
                for option in common.values()
            ]
            feasible_options = [option for option in merged_options if option is not None]
            feasible_options.sort(key=lambda option: (option.energy_kwh, option.ready_offset_s, option.agl_m))
            if feasible_options:
                block_options[(begin, end)] = tuple(feasible_options[: args.max_relay_options])

    best: List[Optional[Tuple[int, float, List[Tuple[int, int]]]]] = [None] * (len(gaps) + 1)
    best[len(gaps)] = (0, 0.0, [])
    for begin in range(len(gaps) - 1, -1, -1):
        for end in range(begin, len(gaps)):
            options = block_options.get((begin, end))
            suffix = best[end + 1]
            if not options or suffix is None:
                continue
            proposal = (1 + suffix[0], min(option.energy_kwh for option in options) + suffix[1], [(begin, end)] + suffix[2])
            current = best[begin]
            if current is None or proposal[:2] < current[:2]:
                best[begin] = proposal
    if best[0] is None:
        return tuple(gaps)

    merged: List[GapProfile] = []
    for mission_number, (begin, end) in enumerate(best[0][2], 1):
        mission_id = f"{joint_id}-M{mission_number:02d}"
        options = tuple(
            RelayOption(
                option_id=f"{mission_id}-{option.candidate_id}",
                candidate_id=option.candidate_id,
                longitude=option.longitude,
                latitude=option.latitude,
                ground_altitude_m=option.ground_altitude_m,
                agl_m=option.agl_m,
                altitude_m=option.altitude_m,
                launch_offset_s=option.launch_offset_s,
                service_start_offset_s=option.service_start_offset_s,
                service_end_offset_s=option.service_end_offset_s,
                return_offset_s=option.return_offset_s,
                ready_offset_s=option.ready_offset_s,
                energy_kwh=option.energy_kwh,
                return_soc_pct=option.return_soc_pct,
            )
            for option in block_options[(begin, end)]
        )
        merged.append(
            GapProfile(
                mission_id,
                gaps[begin].start_index,
                gaps[end].end_index,
                gaps[begin].start_offset_s,
                gaps[end].end_offset_s,
                options,
            )
        )
    return tuple(merged)


def profile_joint_candidates(model, candidates: Sequence[object], global_candidates: Sequence[object], args: argparse.Namespace):
    joint_candidates: List[JointCandidate] = []
    profile_rows: List[dict] = []
    for candidate_index, candidate in enumerate(candidates, 1):
        samples = build_relative_samples(model, candidate)
        flags, detected_gaps = direct_flags_and_gaps(model, samples)
        raw_gaps: List[Tuple[int, int, float, float]] = []
        max_block_samples = max(1, int(math.floor(args.max_relay_block / model.sample_step_s)))
        for begin, end, _, _ in detected_gaps:
            block_begin = begin
            while block_begin <= end:
                block_end = min(end, block_begin + max_block_samples - 1)
                raw_gaps.append(
                    (
                        block_begin,
                        block_end,
                        float(samples[block_begin].time_s),
                        float(samples[block_end].time_s + model.sample_step_s),
                    )
                )
                block_begin = block_end + 1
        gap_profiles: List[GapProfile] = []
        feasible = True
        for gap_number, (begin, end, start_s, end_s) in enumerate(raw_gaps, 1):
            gap_id = f"{candidate.candidate_id}-G{gap_number:02d}"
            options = relay_options_for_gap(model, gap_id, samples, begin, end, start_s, end_s, global_candidates, args)
            if not options:
                feasible = False
                break
            gap_profiles.append(GapProfile(gap_id, begin, end, start_s, end_s, options))
        merged_gap_profiles = merge_gap_profiles(model, candidate.candidate_id, gap_profiles, args) if feasible else tuple()
        profile_rows.append(
            {
                "候选架次": candidate.candidate_id,
                "货箱数量": len(candidate.box_ids),
                "机型": candidate.drone_type,
                "访问顺序": "→".join(candidate.service_order),
                "通信采样点": len(samples),
                "直连中断采样点": sum(not flag for flag in flags),
                "原始通信缺口数": len(raw_gaps),
                "合并后中继任务数": len(merged_gap_profiles),
                "联合通信可行": "是" if feasible else "否",
            }
        )
        if feasible:
            joint_candidates.append(JointCandidate(candidate.candidate_id, candidate, samples, flags, merged_gap_profiles))
        if candidate_index % 50 == 0:
            print(f"  已计算 {candidate_index}/{len(candidates)} 个运输候选的通信剖面，可行 {len(joint_candidates)} 个。")

    covered_boxes = {box_id for candidate in joint_candidates for box_id in candidate.transport.box_ids}
    missing_boxes = sorted(set(PHYSICS.Q2.BOX_BY_ID) - covered_boxes)
    if missing_boxes:
        raise RuntimeError(f"通信可行候选池不能覆盖以下货箱：{missing_boxes}")
    return joint_candidates, profile_rows


def ceil_int(value: float) -> int:
    return int(math.ceil(value - 1e-9))


def floor_int(value: float) -> int:
    return int(math.floor(value + 1e-9))


def preferred_seed_relay_option(gap: GapProfile) -> RelayOption:
    """为热启动选择占用时间短、充电恢复快且能耗低的中继方案。"""
    return min(
        gap.options,
        key=lambda option: (
            ceil_int(option.return_offset_s) - floor_int(option.launch_offset_s),
            ceil_int(option.ready_offset_s) - floor_int(option.launch_offset_s),
            option.energy_kwh,
            option.option_id,
        ),
    )


def build_joint_model(base, candidates: Sequence[JointCandidate], args: argparse.Namespace):
    model = cp_model.CpModel()
    horizon = args.horizon
    max_time = horizon + 20_000
    selected: List[object] = []
    starts: List[object] = []
    task_ends: List[object] = []
    battery_ends: List[object] = []
    transport_intervals: Dict[str, List[object]] = defaultdict(list)
    battery_intervals: Dict[str, List[object]] = defaultdict(list)

    for index, joint in enumerate(candidates):
        candidate = joint.transport
        chosen = model.NewBoolVar(f"x_{index}")
        start = model.NewIntVar(0, horizon, f"start_{index}")
        duration = candidate.duration_s_int
        task_end = model.NewIntVar(0, max_time, f"task_end_{index}")
        model.Add(task_end == start + duration)
        task_interval = model.NewOptionalIntervalVar(start, duration, task_end, chosen, f"task_interval_{index}")
        occupied = candidate.occupied_battery_s_int
        battery_end = model.NewIntVar(0, max_time, f"battery_end_{index}")
        model.Add(battery_end == start + occupied)
        battery_interval = model.NewOptionalIntervalVar(start, occupied, battery_end, chosen, f"battery_interval_{index}")
        selected.append(chosen)
        starts.append(start)
        task_ends.append(task_end)
        battery_ends.append(battery_end)
        transport_intervals[candidate.drone_type].append(task_interval)
        battery_intervals[candidate.drone_type].append(battery_interval)

    for drone_type, intervals in transport_intervals.items():
        model.AddCumulative(intervals, [1] * len(intervals), len(base.DRONE_RESOURCES[drone_type]))
    for drone_type, intervals in battery_intervals.items():
        model.AddCumulative(intervals, [1] * len(intervals), len(base.BATTERY_RESOURCES[drone_type]))

    candidates_by_box: Dict[str, List[int]] = {box_id: [] for box_id in base.BOX_BY_ID}
    delivery_offsets: Dict[Tuple[int, str], int] = {}
    for index, joint in enumerate(candidates):
        for box_id, offset in joint.transport.delivery_offset_int:
            candidates_by_box[box_id].append(index)
            delivery_offsets[(index, box_id)] = offset

    delivery_vars: Dict[str, object] = {}
    tardiness_vars: Dict[str, object] = {}
    weighted_terms: List[object] = []
    for box_id, candidate_indices in candidates_by_box.items():
        if not candidate_indices:
            raise RuntimeError(f"货箱 {box_id} 没有通信可行候选架次。")
        model.Add(sum(selected[index] for index in candidate_indices) == 1)
        delivery = model.NewIntVar(0, max_time, f"delivery_{box_id}")
        for index in candidate_indices:
            model.Add(delivery == starts[index] + delivery_offsets[(index, box_id)]).OnlyEnforceIf(selected[index])
        record = base.BOX_BY_ID[box_id]
        hard_deadline = base.hard_deadline(record)
        if hard_deadline is not None:
            model.Add(delivery <= floor_int(hard_deadline))
        desired = floor_int(float(record["desired_time_s"]))
        tardiness = model.NewIntVar(0, max_time, f"tardiness_{box_id}")
        model.AddMaxEquality(tardiness, [0, delivery - desired])
        priority_scaled = max(1, int(round(float(record["priority"]) * 1000.0)))
        weighted_terms.append(priority_scaled * tardiness)
        delivery_vars[box_id] = delivery
        tardiness_vars[box_id] = tardiness

    relay_choice: Dict[Tuple[int, int, int], object] = {}
    relay_location_jobs: Dict[str, List[Tuple[object, object, object, object]]] = defaultdict(list)
    relay_return_vars: Dict[Tuple[int, int, int], object] = {}
    relay_ready_vars: Dict[Tuple[int, int, int], object] = {}
    relay_energy_terms: List[object] = []

    for candidate_index, joint in enumerate(candidates):
        for gap_index, gap in enumerate(joint.gaps):
            option_bools: List[object] = []
            for option_index, option in enumerate(gap.options):
                key = (candidate_index, gap_index, option_index)
                chosen = model.NewBoolVar(f"relay_choice_{candidate_index}_{gap_index}_{option_index}")
                option_bools.append(chosen)
                relay_choice[key] = chosen

                launch_offset = floor_int(option.launch_offset_s)
                return_offset = ceil_int(option.return_offset_s)
                ready_offset = ceil_int(option.ready_offset_s)
                flight_occupancy = max(1, return_offset - launch_offset)
                energy_occupancy = max(1, ready_offset - launch_offset)

                relay_start = model.NewIntVar(0, max_time, f"relay_start_{candidate_index}_{gap_index}_{option_index}")
                relay_return = model.NewIntVar(0, max_time, f"relay_return_{candidate_index}_{gap_index}_{option_index}")
                relay_ready = model.NewIntVar(0, max_time, f"relay_ready_{candidate_index}_{gap_index}_{option_index}")
                model.Add(relay_start == starts[candidate_index] + launch_offset).OnlyEnforceIf(chosen)
                model.Add(relay_return == starts[candidate_index] + return_offset).OnlyEnforceIf(chosen)
                model.Add(relay_ready == starts[candidate_index] + ready_offset).OnlyEnforceIf(chosen)
                model.Add(relay_start == max_time).OnlyEnforceIf(chosen.Not())
                model.Add(relay_return == 0).OnlyEnforceIf(chosen.Not())
                model.Add(relay_ready == 0).OnlyEnforceIf(chosen.Not())
                relay_location_jobs[option.candidate_id].append((chosen, relay_start, relay_return, relay_ready))
                relay_return_vars[key] = relay_return
                relay_ready_vars[key] = relay_ready
                relay_energy_terms.append(int(round(option.energy_kwh * ENERGY_SCALE)) * chosen)
            if option_bools:
                model.Add(sum(option_bools) == selected[candidate_index])
            else:
                model.Add(selected[candidate_index] >= 0)

    # 通信缺口是硬约束：只要一个运输候选被选中，其全部直连缺口
    # 都必须恰好绑定一个中继选项。额外的最少中继任务约束用于
    # 防止在存在通信缺口的场景下退化为“中继数量为 0”的伪方案。
    minimum_relay_tasks = int(getattr(args, "min_relay_tasks", 0))
    if minimum_relay_tasks > 0 and relay_choice:
        model.Add(sum(relay_choice.values()) >= min(minimum_relay_tasks, len(relay_choice)))

    relay_location_flight_intervals: List[object] = []
    relay_location_energy_intervals: List[object] = []
    relay_location_used: Dict[str, object] = {}
    for location_id, jobs in relay_location_jobs.items():
        used = model.NewBoolVar(f"relay_location_used_{location_id}")
        relay_location_used[location_id] = used
        starts_at_location = [job[1] for job in jobs]
        returns_at_location = [job[2] for job in jobs]
        ready_at_location = [job[3] for job in jobs]
        model.AddMaxEquality(used, [job[0] for job in jobs])
        location_start = model.NewIntVar(0, max_time, f"relay_location_start_{location_id}")
        location_return = model.NewIntVar(0, max_time, f"relay_location_return_{location_id}")
        location_ready = model.NewIntVar(0, max_time, f"relay_location_ready_{location_id}")
        model.AddMinEquality(location_start, starts_at_location)
        model.AddMaxEquality(location_return, returns_at_location)
        model.AddMaxEquality(location_ready, ready_at_location)
        flight_size = model.NewIntVar(0, max_time, f"relay_location_flight_size_{location_id}")
        energy_size = model.NewIntVar(0, max_time, f"relay_location_energy_size_{location_id}")
        model.Add(flight_size == location_return - location_start).OnlyEnforceIf(used)
        model.Add(energy_size == location_ready - location_start).OnlyEnforceIf(used)
        model.Add(flight_size == 0).OnlyEnforceIf(used.Not())
        model.Add(energy_size == 0).OnlyEnforceIf(used.Not())
        relay_location_flight_intervals.append(
            model.NewOptionalIntervalVar(location_start, flight_size, location_return, used, f"relay_location_flight_{location_id}")
        )
        relay_location_energy_intervals.append(
            model.NewOptionalIntervalVar(location_start, energy_size, location_ready, used, f"relay_location_energy_{location_id}")
        )
    if relay_location_flight_intervals:
        model.AddCumulative(relay_location_flight_intervals, [1] * len(relay_location_flight_intervals), args.relay_drone_count)
    if relay_location_energy_intervals:
        model.AddCumulative(relay_location_energy_intervals, [1] * len(relay_location_energy_intervals), args.relay_component_count)

    joint_makespan = model.NewIntVar(0, max_time, "joint_makespan")
    for index in range(len(candidates)):
        model.Add(joint_makespan >= task_ends[index]).OnlyEnforceIf(selected[index])
    for key, chosen in relay_choice.items():
        model.Add(joint_makespan >= relay_return_vars[key]).OnlyEnforceIf(chosen)

    weighted_tardiness = model.NewIntVar(0, 10**12, "weighted_tardiness")
    model.Add(weighted_tardiness == sum(weighted_terms))
    transport_energy_terms = [joint.transport.energy_int * selected[index] for index, joint in enumerate(candidates)]
    total_energy = model.NewIntVar(0, 10**10, "total_energy")
    model.Add(total_energy == sum(transport_energy_terms) + sum(relay_energy_terms))
    relay_location_count = model.NewIntVar(0, len(relay_location_used), "relay_location_count")
    model.Add(relay_location_count == sum(relay_location_used.values()))
    total_sorties = model.NewIntVar(0, 1000, "total_sorties")
    model.Add(total_sorties == sum(selected) + relay_location_count)

    variables = {
        "selected": selected,
        "starts": starts,
        "task_ends": task_ends,
        "battery_ends": battery_ends,
        "deliveries": delivery_vars,
        "tardiness": tardiness_vars,
        "relay_choice": relay_choice,
        "relay_returns": relay_return_vars,
        "relay_ready": relay_ready_vars,
        "relay_location_used": relay_location_used,
        "relay_location_count": relay_location_count,
        "objectives": {
            "加权迟到": weighted_tardiness,
            "联合任务完成时间": joint_makespan,
            "联合总能耗": total_energy,
            "运输与中继总架次": total_sorties,
        },
    }
    return model, variables


def objective_sequence(args: argparse.Namespace) -> List[str]:
    sequences = {
        "timeliness": ["加权迟到", "联合任务完成时间", "联合总能耗", "运输与中继总架次"],
        "relayaware": ["加权迟到", "运输与中继总架次", "联合任务完成时间", "联合总能耗"],
        "makespan": ["联合任务完成时间", "加权迟到", "联合总能耗", "运输与中继总架次"],
        "energy": ["联合总能耗", "加权迟到", "联合任务完成时间", "运输与中继总架次"],
        "sortie": ["运输与中继总架次", "加权迟到", "联合任务完成时间", "联合总能耗"],
    }
    return sequences[args.objective]


def solve_joint_model(model, variables: dict, args: argparse.Namespace) -> JointSolution:
    def refresh_solution_hint(target_model, target_solver) -> None:
        target_model.ClearHints()
        hint_variables = (
            list(variables["selected"])
            + list(variables["starts"])
            + list(variables["task_ends"])
            + list(variables["battery_ends"])
            + list(variables["deliveries"].values())
            + list(variables["tardiness"].values())
            + list(variables["relay_choice"].values())
            + list(variables["relay_returns"].values())
            + list(variables["relay_ready"].values())
            + list(variables["objectives"].values())
        )
        seen = set()
        for variable in hint_variables:
            index = variable.Index()
            if index in seen:
                continue
            seen.add(index)
            target_model.AddHint(variable, target_solver.Value(variable))

    stage_log: List[dict] = []
    final_solver = None
    start_clock = perf_counter()
    model.Minimize(0)
    feasibility_solver = cp_model.CpSolver()
    feasibility_solver.parameters.max_time_in_seconds = args.feasibility_time
    feasibility_solver.parameters.num_search_workers = args.workers
    feasibility_solver.parameters.random_seed = args.seed
    feasibility_status = feasibility_solver.Solve(model)
    if feasibility_status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        final_solver = feasibility_solver
        stage_log.append(
            {
                "阶段": 0,
                "目标": "寻找首个联合可行解",
                "状态": feasibility_solver.StatusName(feasibility_status),
                "目标值": 0,
                "最好界": 0,
                "相对gap": 0,
                "已证明最优": "否",
                "累计求解时间_s": perf_counter() - start_clock,
            }
        )
        refresh_solution_hint(model, feasibility_solver)
        print("  已找到联合可行热启动解，开始字典序优化。")
    else:
        print(f"  可行热启动阶段状态={feasibility_solver.StatusName(feasibility_status)}，继续直接优化。")

    for stage_number, objective_name in enumerate(objective_sequence(args)[: args.max_stages], 1):
        objective = variables["objectives"][objective_name]
        incumbent_value = None if final_solver is None else int(final_solver.Value(objective))
        if incumbent_value is not None and incumbent_value > 0:
            search_model = model.Clone()
            search_model.ClearHints()
            search_model.ClearObjective()
            search_model.Add(objective <= incumbent_value - 1)
            search_model.Minimize(objective)
            strict_improvement = True
        else:
            search_model = model
            search_model.Minimize(objective)
            strict_improvement = False
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = args.time_limit
        solver.parameters.num_search_workers = args.workers
        solver.parameters.random_seed = args.seed + stage_number
        solver.parameters.log_search_progress = args.solver_log
        solver.parameters.cp_model_presolve = True
        solver.parameters.linearization_level = 2
        solver.parameters.use_lns = True
        status = solver.Solve(search_model)
        status_name = solver.StatusName(status)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            value = int(solver.Value(objective))
            if incumbent_value is not None and value >= incumbent_value:
                value = incumbent_value
                bound = float(value)
                gap = 0.0
                proven = "否"
                print(f"  阶段 {stage_number}/4 {objective_name}：未找到严格改进，保留当前值={value}")
            else:
                bound = float(solver.BestObjectiveBound())
                gap = 0.0 if value == 0 else max(0.0, (value - bound) / abs(value))
                proven = "是" if status == cp_model.OPTIMAL else "否"
                final_solver = solver
                refresh_solution_hint(model, final_solver)
                print(f"  阶段 {stage_number}/4 {objective_name}：{status_name}，改进值={value}，gap={gap:.6f}")
            model.Add(objective == value)
            stage_log.append(
                {
                    "阶段": stage_number,
                    "目标": objective_name,
                    "状态": status_name,
                    "目标值": value,
                    "最好界": bound,
                    "相对gap": gap,
                    "已证明最优": proven,
                    "累计求解时间_s": perf_counter() - start_clock,
                }
            )
            continue

        if final_solver is None:
            raise RuntimeError(f"联合模型在阶段 {objective_name} 未找到可行解，状态={status_name}。")

        value = incumbent_value
        model.Add(objective == value)
        if strict_improvement and status == cp_model.INFEASIBLE:
            stage_status = "OPTIMAL"
            proven = "是"
            bound = float(value)
            gap = 0.0
            print(f"  阶段 {stage_number}/4 {objective_name}：已证明不存在更优解，最优值={value}")
        else:
            stage_status = status_name
            proven = "否"
            bound = float(solver.BestObjectiveBound())
            gap = None
            print(f"  阶段 {stage_number}/4 {objective_name}：{status_name}，保留当前可行值={value}，未证明最优")
        stage_log.append(
            {
                "阶段": stage_number,
                "目标": objective_name,
                "状态": stage_status,
                "目标值": value,
                "最好界": bound,
                "相对gap": gap,
                "已证明最优": proven,
                "累计求解时间_s": perf_counter() - start_clock,
            }
        )

    if final_solver is None:
        raise RuntimeError("CP-SAT 未返回可行解。")
    objective_values = tuple(int(final_solver.Value(variables["objectives"][name])) for name in ["加权迟到", "联合任务完成时间", "联合总能耗", "运输与中继总架次"])
    return JointSolution(final_solver, variables, stage_log, objective_values)


def seed_candidate_subset(candidates: Sequence[JointCandidate], partition) -> List[JointCandidate]:
    route_keys = set(partition)
    by_route: Dict[Tuple[str, ...], List[JointCandidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.transport.box_ids in route_keys:
            by_route[candidate.transport.box_ids].append(candidate)
    subset: List[JointCandidate] = []
    for route_key in route_keys:
        options = by_route.get(route_key, [])
        if not options:
            return []
        subset.append(min(options, key=lambda item: (len(item.gaps), candidate_rank(item.transport))))
    available = {candidate.transport.box_ids for candidate in subset}
    if available != route_keys:
        return []
    return subset


def find_seed_joint_solution(base, candidates: Sequence[JointCandidate], seeds, args: argparse.Namespace):
    best = None
    for seed_number, partition in enumerate(seeds, 1):
        subset = seed_candidate_subset(candidates, partition)
        if not subset:
            print(f"  种子分区 {seed_number} 缺少通信可行航序，跳过。")
            continue
        local_args = argparse.Namespace(**vars(args))
        local_args.feasibility_time = args.seed_feasibility_time
        local_args.time_limit = args.seed_time_limit
        local_args.max_stages = 1
        local_model, local_variables = build_joint_model(base, subset, local_args)
        try:
            local_solution = solve_joint_model(local_model, local_variables, local_args)
        except RuntimeError as error:
            print(f"  种子分区 {seed_number} 未找到可行解：{error}")
            continue
        comparison = local_solution.objective_values
        print(f"  种子分区 {seed_number} 可行：候选={len(subset)}，目标={comparison}")
        if best is None or comparison < best[0]:
            best = (comparison, subset, local_solution)
    if best is None:
        raise RuntimeError("三个独立构造种子均未找到联合可行解，请扩大候选池或检查资源假设。")
    return best[1], best[2]


def find_cover_seed(base, candidates: Sequence[JointCandidate], args: argparse.Namespace) -> List[JointCandidate]:
    """先忽略资源时间轴，选择一套覆盖全部货箱且通信缺口尽量少的组批。"""
    model = cp_model.CpModel()
    selected = [model.NewBoolVar(f"cover_{index}") for index in range(len(candidates))]
    by_box: Dict[str, List[int]] = {box_id: [] for box_id in base.BOX_BY_ID}
    for index, candidate in enumerate(candidates):
        for box_id in candidate.transport.box_ids:
            by_box[box_id].append(index)
    for box_id, indices in by_box.items():
        if not indices:
            raise RuntimeError(f"通信可行候选池不能覆盖货箱 {box_id}。")
        model.Add(sum(selected[index] for index in indices) == 1)
    gap_cost = sum(len(candidate.gaps) * 100_000 * selected[index] for index, candidate in enumerate(candidates))
    route_cost = sum(1_000 * selected[index] for index in range(len(candidates)))
    energy_cost = sum(int(round(candidate.transport.option.energy_kwh * 10.0)) * selected[index] for index, candidate in enumerate(candidates))
    model.Minimize(gap_cost + route_cost + energy_cost)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = args.cover_time_limit
    solver.parameters.num_search_workers = args.workers
    solver.parameters.random_seed = args.seed + 97
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"通信感知集合划分未找到覆盖解：{solver.StatusName(status)}")
    chosen = [candidate for index, candidate in enumerate(candidates) if solver.Value(selected[index])]
    print(f"  通信感知集合划分：状态={solver.StatusName(status)}，选中运输候选={len(chosen)}，通信缺口块数={sum(len(item.gaps) for item in chosen)}")
    return chosen


def find_transport_resource_seed(base, candidates: Sequence[JointCandidate], args: argparse.Namespace):
    """建立同时满足运输资源和中继容量的保守联合热启动。"""
    model = cp_model.CpModel()
    max_time = args.horizon + 20_000
    selected: List[object] = []
    starts: List[object] = []
    ends: List[object] = []
    drone_intervals: Dict[str, List[object]] = defaultdict(list)
    battery_intervals: Dict[str, List[object]] = defaultdict(list)
    relay_location_jobs: Dict[str, List[Tuple[object, object, object, object]]] = defaultdict(list)
    relay_returns: List[Tuple[object, object]] = []
    preferred_options: Dict[Tuple[int, int], RelayOption] = {}
    by_box: Dict[str, List[int]] = {box_id: [] for box_id in base.BOX_BY_ID}
    offset_by_box: Dict[Tuple[int, str], int] = {}

    for index, joint in enumerate(candidates):
        candidate = joint.transport
        chosen = model.NewBoolVar(f"transport_seed_x_{index}")
        start = model.NewIntVar(0, args.horizon, f"transport_seed_start_{index}")
        end = model.NewIntVar(0, max_time, f"transport_seed_end_{index}")
        model.Add(end == start + candidate.duration_s_int)
        drone_interval = model.NewOptionalIntervalVar(start, candidate.duration_s_int, end, chosen, f"transport_seed_drone_{index}")
        battery_end = model.NewIntVar(0, max_time, f"transport_seed_battery_end_{index}")
        model.Add(battery_end == start + candidate.occupied_battery_s_int)
        battery_interval = model.NewOptionalIntervalVar(start, candidate.occupied_battery_s_int, battery_end, chosen, f"transport_seed_battery_{index}")
        selected.append(chosen)
        starts.append(start)
        ends.append(end)
        drone_intervals[candidate.drone_type].append(drone_interval)
        battery_intervals[candidate.drone_type].append(battery_interval)
        for box_id, offset in candidate.delivery_offset_int:
            by_box[box_id].append(index)
            offset_by_box[(index, box_id)] = offset
        for gap_index, gap in enumerate(joint.gaps):
            option = preferred_seed_relay_option(gap)
            preferred_options[(index, gap_index)] = option
            launch_offset = floor_int(option.launch_offset_s)
            return_offset = ceil_int(option.return_offset_s)
            ready_offset = ceil_int(option.ready_offset_s)
            relay_start = model.NewIntVar(0, max_time, f"transport_seed_relay_start_{index}_{gap_index}")
            relay_return = model.NewIntVar(0, max_time, f"transport_seed_relay_return_{index}_{gap_index}")
            relay_ready = model.NewIntVar(0, max_time, f"transport_seed_relay_ready_{index}_{gap_index}")
            model.Add(relay_start == start + launch_offset).OnlyEnforceIf(chosen)
            model.Add(relay_return == start + return_offset).OnlyEnforceIf(chosen)
            model.Add(relay_ready == start + ready_offset).OnlyEnforceIf(chosen)
            model.Add(relay_start == max_time).OnlyEnforceIf(chosen.Not())
            model.Add(relay_return == 0).OnlyEnforceIf(chosen.Not())
            model.Add(relay_ready == 0).OnlyEnforceIf(chosen.Not())
            relay_location_jobs[option.candidate_id].append((chosen, relay_start, relay_return, relay_ready))
            relay_returns.append((chosen, relay_return))

    for drone_type, intervals in drone_intervals.items():
        model.AddCumulative(intervals, [1] * len(intervals), len(base.DRONE_RESOURCES[drone_type]))
    for drone_type, intervals in battery_intervals.items():
        model.AddCumulative(intervals, [1] * len(intervals), len(base.BATTERY_RESOURCES[drone_type]))
    relay_location_flight_intervals: List[object] = []
    relay_location_energy_intervals: List[object] = []
    for location_id, jobs in relay_location_jobs.items():
        used = model.NewBoolVar(f"transport_seed_location_used_{location_id}")
        model.AddMaxEquality(used, [job[0] for job in jobs])
        location_start = model.NewIntVar(0, max_time, f"transport_seed_location_start_{location_id}")
        location_return = model.NewIntVar(0, max_time, f"transport_seed_location_return_{location_id}")
        location_ready = model.NewIntVar(0, max_time, f"transport_seed_location_ready_{location_id}")
        model.AddMinEquality(location_start, [job[1] for job in jobs])
        model.AddMaxEquality(location_return, [job[2] for job in jobs])
        model.AddMaxEquality(location_ready, [job[3] for job in jobs])
        flight_size = model.NewIntVar(0, max_time, f"transport_seed_location_flight_size_{location_id}")
        energy_size = model.NewIntVar(0, max_time, f"transport_seed_location_energy_size_{location_id}")
        model.Add(flight_size == location_return - location_start).OnlyEnforceIf(used)
        model.Add(energy_size == location_ready - location_start).OnlyEnforceIf(used)
        model.Add(flight_size == 0).OnlyEnforceIf(used.Not())
        model.Add(energy_size == 0).OnlyEnforceIf(used.Not())
        relay_location_flight_intervals.append(model.NewOptionalIntervalVar(location_start, flight_size, location_return, used, f"transport_seed_location_flight_{location_id}"))
        relay_location_energy_intervals.append(model.NewOptionalIntervalVar(location_start, energy_size, location_ready, used, f"transport_seed_location_energy_{location_id}"))
    if relay_location_flight_intervals:
        model.AddCumulative(relay_location_flight_intervals, [1] * len(relay_location_flight_intervals), args.relay_drone_count)
    if relay_location_energy_intervals:
        model.AddCumulative(relay_location_energy_intervals, [1] * len(relay_location_energy_intervals), args.relay_component_count)

    weighted_terms: List[object] = []
    for box_id, indices in by_box.items():
        model.Add(sum(selected[index] for index in indices) == 1)
        delivery = model.NewIntVar(0, max_time, f"transport_seed_delivery_{box_id}")
        for index in indices:
            model.Add(delivery == starts[index] + offset_by_box[(index, box_id)]).OnlyEnforceIf(selected[index])
        record = base.BOX_BY_ID[box_id]
        hard_deadline = base.hard_deadline(record)
        if hard_deadline is not None:
            model.Add(delivery <= floor_int(hard_deadline))
        tardiness = model.NewIntVar(0, max_time, f"transport_seed_tardiness_{box_id}")
        model.AddMaxEquality(tardiness, [0, delivery - floor_int(float(record["desired_time_s"]))])
        weighted_terms.append(max(1, int(round(float(record["priority"]) * 1000.0))) * tardiness)

    weighted_tardiness = model.NewIntVar(0, 10**12, "transport_seed_weighted_tardiness")
    model.Add(weighted_tardiness == sum(weighted_terms))
    gap_burden = model.NewIntVar(0, 10**8, "transport_seed_gap_burden")
    gap_coefficients = [
        sum(ceil_int(gap.end_offset_s - gap.start_offset_s) for gap in joint.gaps)
        + 300 * len(joint.gaps)
        for joint in candidates
    ]
    model.Add(gap_burden == sum(gap_coefficients[index] * selected[index] for index in range(len(candidates))))
    makespan = model.NewIntVar(0, max_time, "transport_seed_makespan")
    for index in range(len(candidates)):
        model.Add(makespan >= ends[index]).OnlyEnforceIf(selected[index])
    for chosen, relay_return in relay_returns:
        model.Add(makespan >= relay_return).OnlyEnforceIf(chosen)
    sortie_count = model.NewIntVar(0, 200, "transport_seed_sorties")
    model.Add(sortie_count == sum(selected))

    stages = [
        ("通信缺口负担", gap_burden),
        ("加权迟到", weighted_tardiness),
        ("运输完工时间", makespan),
        ("运输架次数", sortie_count),
    ]
    final_solver = None
    log: List[dict] = []
    for stage_number, (name, objective) in enumerate(stages[: args.transport_seed_stages], 1):
        model.Minimize(objective)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = args.transport_seed_time_limit
        solver.parameters.num_search_workers = args.workers
        solver.parameters.random_seed = args.seed + 200 + stage_number
        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            if final_solver is None:
                raise RuntimeError(f"运输资源主问题无可行解：{solver.StatusName(status)}")
            break
        value = int(solver.Value(objective))
        log.append({"阶段": stage_number, "目标": name, "状态": solver.StatusName(status), "目标值": value})
        print(f"  运输种子阶段 {stage_number} {name}：{solver.StatusName(status)}，目标={value}")
        final_solver = solver
        model.Add(objective == value)
    if final_solver is None:
        raise RuntimeError("运输资源主问题未执行。")
    chosen_indices = [index for index, variable in enumerate(selected) if final_solver.Value(variable)]
    chosen = [candidates[index] for index in chosen_indices]
    start_hint = {candidates[index].candidate_id: int(final_solver.Value(starts[index])) for index in chosen_indices}
    relay_option_hint = {
        preferred_options[(index, gap_index)].option_id
        for index in chosen_indices
        for gap_index in range(len(candidates[index].gaps))
    }
    print(f"  运输-中继资源可行种子：运输架次={len(chosen)}，通信缺口块数={sum(len(item.gaps) for item in chosen)}。")
    return chosen, start_hint, relay_option_hint, log


def build_transport_master(base, candidates: Sequence[JointCandidate], args: argparse.Namespace):
    """构造不含中继变量的运输主问题，供分解式热启动反复求解。"""
    model = cp_model.CpModel()
    max_time = args.horizon + 20_000
    selected: List[object] = []
    starts: List[object] = []
    ends: List[object] = []
    drone_intervals: Dict[str, List[object]] = defaultdict(list)
    battery_intervals: Dict[str, List[object]] = defaultdict(list)
    by_box: Dict[str, List[int]] = {box_id: [] for box_id in base.BOX_BY_ID}
    offset_by_box: Dict[Tuple[int, str], int] = {}

    for index, joint in enumerate(candidates):
        candidate = joint.transport
        chosen = model.NewBoolVar(f"master_x_{index}")
        start = model.NewIntVar(0, args.horizon, f"master_start_{index}")
        end = model.NewIntVar(0, max_time, f"master_end_{index}")
        model.Add(end == start + candidate.duration_s_int)
        battery_end = model.NewIntVar(0, max_time, f"master_battery_end_{index}")
        model.Add(battery_end == start + candidate.occupied_battery_s_int)
        selected.append(chosen)
        starts.append(start)
        ends.append(end)
        drone_intervals[candidate.drone_type].append(
            model.NewOptionalIntervalVar(start, candidate.duration_s_int, end, chosen, f"master_drone_{index}")
        )
        battery_intervals[candidate.drone_type].append(
            model.NewOptionalIntervalVar(start, candidate.occupied_battery_s_int, battery_end, chosen, f"master_battery_{index}")
        )
        for box_id, offset in candidate.delivery_offset_int:
            by_box[box_id].append(index)
            offset_by_box[(index, box_id)] = offset

    for drone_type, intervals in drone_intervals.items():
        model.AddCumulative(intervals, [1] * len(intervals), len(base.DRONE_RESOURCES[drone_type]))
    for drone_type, intervals in battery_intervals.items():
        model.AddCumulative(intervals, [1] * len(intervals), len(base.BATTERY_RESOURCES[drone_type]))

    weighted_terms: List[object] = []
    for box_id, indices in by_box.items():
        model.Add(sum(selected[index] for index in indices) == 1)
        delivery = model.NewIntVar(0, max_time, f"master_delivery_{box_id}")
        for index in indices:
            model.Add(delivery == starts[index] + offset_by_box[(index, box_id)]).OnlyEnforceIf(selected[index])
        record = base.BOX_BY_ID[box_id]
        hard_deadline = base.hard_deadline(record)
        if hard_deadline is not None:
            model.Add(delivery <= floor_int(hard_deadline))
        tardiness = model.NewIntVar(0, max_time, f"master_tardiness_{box_id}")
        model.AddMaxEquality(tardiness, [0, delivery - floor_int(float(record["desired_time_s"]))])
        weighted_terms.append(max(1, int(round(float(record["priority"]) * 1000.0))) * tardiness)

    gap_burden = model.NewIntVar(0, 10**9, "master_gap_burden")
    gap_coefficients = [
        sum(ceil_int(gap.end_offset_s - gap.start_offset_s) for gap in joint.gaps) + 300 * len(joint.gaps)
        for joint in candidates
    ]
    model.Add(gap_burden == sum(gap_coefficients[index] * selected[index] for index in range(len(candidates))))
    weighted_tardiness = model.NewIntVar(0, 10**12, "master_weighted_tardiness")
    model.Add(weighted_tardiness == sum(weighted_terms))
    makespan = model.NewIntVar(0, max_time, "master_makespan")
    for index in range(len(candidates)):
        model.Add(makespan >= ends[index]).OnlyEnforceIf(selected[index])
    sortie_count = model.NewIntVar(0, 1000, "master_sorties")
    model.Add(sortie_count == sum(selected))
    variables = {
        "selected": selected,
        "starts": starts,
        "objectives": {
            "通信缺口负担": gap_burden,
            "加权迟到": weighted_tardiness,
            "运输完工时间": makespan,
            "运输架次数": sortie_count,
        },
    }
    return model, variables


def solve_fixed_joint_seed(base, candidates: Sequence[JointCandidate], start_hint: Dict[str, int], args: argparse.Namespace):
    """固定一套运输组批，仅求中继、开始时刻和资源安排的可行性。"""
    model, variables = build_joint_model(base, candidates, args)
    for index, candidate in enumerate(candidates):
        model.Add(variables["selected"][index] == 1)
        if candidate.candidate_id in start_hint:
            model.AddHint(variables["starts"][index], start_hint[candidate.candidate_id])
    model.Minimize(0)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = args.seed_feasibility_time
    solver.parameters.num_search_workers = args.workers
    solver.parameters.random_seed = args.seed + 701
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    values = tuple(int(solver.Value(variables["objectives"][name])) for name in ["加权迟到", "联合任务完成时间", "联合总能耗", "运输与中继总架次"])
    log = [{"阶段": 0, "目标": "分解式联合可行解", "状态": solver.StatusName(status), "目标值": 0, "最好界": 0, "相对gap": 0, "已证明最优": "否", "累计求解时间_s": solver.WallTime()}]
    return JointSolution(solver, variables, log, values)


def find_decomposition_seed(base, candidates: Sequence[JointCandidate], args: argparse.Namespace):
    """运输主问题 + 中继可行性检验的 Benders 风格热启动。"""
    model, variables = build_transport_master(base, candidates, args)
    for iteration in range(1, args.seed_iterations + 1):
        model.Minimize(
            variables["objectives"]["通信缺口负担"] * 1_000_000
            + variables["objectives"]["加权迟到"] * 10
            + variables["objectives"]["运输完工时间"]
            + variables["objectives"]["运输架次数"]
        )
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = args.master_time_limit
        solver.parameters.num_search_workers = args.workers
        solver.parameters.random_seed = args.seed + 800 + iteration
        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            print(f"  分解热启动第{iteration}轮运输主问题：{solver.StatusName(status)}")
            break
        chosen_indices = [index for index, variable in enumerate(variables["selected"]) if solver.Value(variable)]
        chosen = [candidates[index] for index in chosen_indices]
        start_hint = {candidates[index].candidate_id: int(solver.Value(variables["starts"][index])) for index in chosen_indices}
        print(f"  分解热启动第{iteration}轮：运输架次={len(chosen)}，通信任务={sum(len(c.gaps) for c in chosen)}")
        joint_solution = solve_fixed_joint_seed(base, chosen, start_hint, args)
        if joint_solution is not None:
            print(f"  分解热启动找到联合可行解：第{iteration}轮。")
            return chosen, joint_solution
        model.Add(sum(variables["selected"][index] for index in chosen_indices) <= len(chosen_indices) - 1)
    return None


def add_solution_hint(model, variables: dict, full_candidates: Sequence[JointCandidate], seed_candidates: Sequence[JointCandidate], seed_solution: JointSolution) -> None:
    seed_solver = seed_solution.solver
    seed_variables = seed_solution.variables
    selected_start: Dict[str, int] = {}
    selected_relay: set[str] = set()
    for index, candidate in enumerate(seed_candidates):
        if seed_solver.Value(seed_variables["selected"][index]):
            selected_start[candidate.candidate_id] = int(seed_solver.Value(seed_variables["starts"][index]))
    for (candidate_index, gap_index, option_index), variable in seed_variables["relay_choice"].items():
        if seed_solver.Value(variable):
            selected_relay.add(seed_candidates[candidate_index].gaps[gap_index].options[option_index].option_id)

    for index, candidate in enumerate(full_candidates):
        is_selected = candidate.candidate_id in selected_start
        model.AddHint(variables["selected"][index], int(is_selected))
        model.AddHint(variables["starts"][index], selected_start.get(candidate.candidate_id, 0))
    for (candidate_index, gap_index, option_index), variable in variables["relay_choice"].items():
        option_id = full_candidates[candidate_index].gaps[gap_index].options[option_index].option_id
        model.AddHint(variable, int(option_id in selected_relay))


def add_transport_seed_hint(
    model,
    variables: dict,
    candidates: Sequence[JointCandidate],
    start_hint: Dict[str, int],
    relay_option_hint: Iterable[str] = (),
) -> None:
    relay_option_ids = set(relay_option_hint)
    for index, candidate in enumerate(candidates):
        selected_value = int(candidate.candidate_id in start_hint)
        model.AddHint(variables["selected"][index], selected_value)
        model.AddHint(variables["starts"][index], start_hint.get(candidate.candidate_id, 0))
    for (candidate_index, gap_index, option_index), variable in variables["relay_choice"].items():
        option_id = candidates[candidate_index].gaps[gap_index].options[option_index].option_id
        model.AddHint(variable, int(option_id in relay_option_ids))


def fix_transport_seed_decisions(
    model,
    variables: dict,
    candidates: Sequence[JointCandidate],
    start_hint: Dict[str, int],
    relay_option_hint: Iterable[str],
) -> None:
    """固定已由保守容量模型验证的种子决策，快速恢复联合可行解。"""
    relay_option_ids = set(relay_option_hint)
    for index, candidate in enumerate(candidates):
        selected_value = int(candidate.candidate_id in start_hint)
        model.Add(variables["selected"][index] == selected_value)
        if selected_value:
            model.Add(variables["starts"][index] == start_hint[candidate.candidate_id])
    for (candidate_index, gap_index, option_index), variable in variables["relay_choice"].items():
        option_id = candidates[candidate_index].gaps[gap_index].options[option_index].option_id
        model.Add(variable == int(option_id in relay_option_ids))


def color_intervals(records: Sequence[dict], resource_ids: Sequence[str], start_key: str, end_key: str, identity_key: str) -> Dict[str, str]:
    ready = {resource_id: 0.0 for resource_id in resource_ids}
    assignments: Dict[str, str] = {}
    for record in sorted(records, key=lambda item: (item[start_key], item[end_key], item[identity_key])):
        feasible = [resource_id for resource_id in resource_ids if ready[resource_id] <= record[start_key] + TOLERANCE]
        if not feasible:
            raise RuntimeError(f"资源恢复失败：{record[identity_key]} 在 {record[start_key]} 没有可用资源。")
        chosen = min(feasible, key=lambda resource_id: (ready[resource_id], resource_id))
        assignments[str(record[identity_key])] = chosen
        ready[chosen] = float(record[end_key])
    return assignments


def extract_solution(base, candidates: Sequence[JointCandidate], solution: JointSolution, args: argparse.Namespace):
    solver = solution.solver
    variables = solution.variables
    selected_indices = [index for index, chosen in enumerate(variables["selected"]) if solver.Value(chosen)]
    transport_rows: List[dict] = []
    transport_internal: List[dict] = []
    for sequence, index in enumerate(sorted(selected_indices, key=lambda idx: (solver.Value(variables["starts"][idx]), candidates[idx].candidate_id)), 1):
        joint = candidates[index]
        candidate = joint.transport
        option = candidate.option
        start = int(solver.Value(variables["starts"][index]))
        record = {
            "架次编号": f"Q3-T-{sequence:03d}",
            "候选编号": joint.candidate_id,
            "机型编号": candidate.drone_type,
            "开始时刻_s": start,
            "起飞时刻_s": start + option.takeoff_offset_s,
            "访问服务区顺序": "→".join(candidate.service_order),
            "完整路线": "O01→" + "→".join(candidate.service_order) + "→O01",
            "货箱编号列表": "、".join(candidate.box_ids),
            "货箱数量": len(candidate.box_ids),
            "总质量_kg": option.total_mass_kg,
            "总体积_m3": option.total_volume_m3,
            "返回O01时刻_s": start + option.duration_s,
            "电池再次可用时刻_s": start + candidate.occupied_battery_s_int,
            "架次能耗_kWh": option.energy_kwh,
            "返航SOC_pct": option.return_soc_pct,
            "通信缺口块数": len(joint.gaps),
            "candidate_index": index,
        }
        transport_rows.append(record)
        transport_internal.append(record)

    drone_assignments: Dict[str, str] = {}
    battery_assignments: Dict[str, str] = {}
    for drone_type in sorted(base.DRONE_RESOURCES):
        typed = [row for row in transport_internal if row["机型编号"] == drone_type]
        drone_assignments.update(color_intervals(typed, base.DRONE_RESOURCES[drone_type], "开始时刻_s", "返回O01时刻_s", "架次编号"))
        battery_assignments.update(color_intervals(typed, base.BATTERY_RESOURCES[drone_type], "开始时刻_s", "电池再次可用时刻_s", "架次编号"))
    for row in transport_rows:
        row["无人机编号"] = drone_assignments[row["架次编号"]]
        row["电池编号"] = battery_assignments[row["架次编号"]]

    selected_relay: List[dict] = []
    relay_by_gap: Dict[Tuple[int, int], dict] = {}
    transport_id_by_index = {row["candidate_index"]: row["架次编号"] for row in transport_rows}
    for key, chosen in variables["relay_choice"].items():
        if not solver.Value(chosen):
            continue
        candidate_index, gap_index, option_index = key
        joint = candidates[candidate_index]
        gap = joint.gaps[gap_index]
        option = gap.options[option_index]
        transport_start = int(solver.Value(variables["starts"][candidate_index]))
        record = {
            "中继任务键": option.option_id,
            "运输架次编号": transport_id_by_index[candidate_index],
            "运输候选编号": joint.candidate_id,
            "通信缺口编号": gap.gap_id,
            "悬停点编号": option.candidate_id,
            "经度": option.longitude,
            "纬度": option.latitude,
            "地面高程_m": option.ground_altitude_m,
            "悬停离地高度_m": option.agl_m,
            "悬停绝对海拔_m": option.altitude_m,
            "中继起飞时刻_s": transport_start + option.launch_offset_s,
            "开始服务时刻_s": transport_start + option.service_start_offset_s,
            "结束服务时刻_s": transport_start + option.service_end_offset_s,
            "返回O01时刻_s": transport_start + option.return_offset_s,
            "能源组件再次可用时刻_s": transport_start + option.ready_offset_s,
            "中继能耗_kWh": option.energy_kwh,
            "返航SOC_pct": option.return_soc_pct,
            "candidate_index": candidate_index,
            "gap_index": gap_index,
            "option_index": option_index,
        }
        selected_relay.append(record)
        relay_by_gap[(candidate_index, gap_index)] = record

    location_records: Dict[str, dict] = {}
    for row in selected_relay:
        location_id = str(row["悬停点编号"])
        current = location_records.get(location_id)
        if current is None:
            location_records[location_id] = {
                "悬停点编号": location_id,
                "中继起飞时刻_s": float(row["中继起飞时刻_s"]),
                "返回O01时刻_s": float(row["返回O01时刻_s"]),
                "能源组件再次可用时刻_s": float(row["能源组件再次可用时刻_s"]),
            }
        else:
            current["中继起飞时刻_s"] = min(current["中继起飞时刻_s"], float(row["中继起飞时刻_s"]))
            current["返回O01时刻_s"] = max(current["返回O01时刻_s"], float(row["返回O01时刻_s"]))
            current["能源组件再次可用时刻_s"] = max(current["能源组件再次可用时刻_s"], float(row["能源组件再次可用时刻_s"]))
    location_list = list(location_records.values())
    relay_drone_by_location = color_intervals(
        location_list,
        tuple(f"R{index:02d}" for index in range(1, args.relay_drone_count + 1)),
        "中继起飞时刻_s",
        "返回O01时刻_s",
        "悬停点编号",
    )
    relay_component_by_location = color_intervals(
        location_list,
        tuple(f"R-ENERGY-{index:02d}" for index in range(1, args.relay_component_count + 1)),
        "中继起飞时刻_s",
        "能源组件再次可用时刻_s",
        "悬停点编号",
    )
    for sequence, row in enumerate(sorted(selected_relay, key=lambda item: (item["中继起飞时刻_s"], item["中继任务键"])), 1):
        row["中继架次编号"] = f"Q3-R-{sequence:03d}"
        location_id = str(row["悬停点编号"])
        row["中继无人机编号"] = relay_drone_by_location[location_id]
        row["能源组件编号"] = relay_component_by_location[location_id]

    delivery_rows: List[dict] = []
    transport_row_by_index = {row["candidate_index"]: row for row in transport_rows}
    for box_id, record in base.BOX_BY_ID.items():
        selected_index = next(index for index in selected_indices if box_id in candidates[index].transport.box_ids)
        delivery = int(solver.Value(variables["deliveries"][box_id]))
        desired = float(record["desired_time_s"])
        hard_deadline = base.hard_deadline(record)
        delivery_rows.append(
            {
                "货箱编号": box_id,
                "运输架次编号": transport_row_by_index[selected_index]["架次编号"],
                "服务区": record["service_id"],
                "物资类型": record["material_type"],
                "优先系数": record["priority"],
                "期望送达时间_s": desired,
                "硬截止时间_s": "" if hard_deadline is None else hard_deadline,
                "实际送达时刻_s": delivery,
                "期望时间迟到_s": max(0.0, delivery - desired),
                "硬时限是否满足": "是" if hard_deadline is None or delivery <= hard_deadline + TOLERANCE else "否",
            }
        )
    return transport_rows, selected_relay, delivery_rows, relay_by_gap


def communication_records(candidates: Sequence[JointCandidate], transport_rows: Sequence[dict], relay_by_gap: Dict[Tuple[int, int], dict]):
    records: List[dict] = []
    for transport in transport_rows:
        candidate_index = int(transport["candidate_index"])
        joint = candidates[candidate_index]
        start = float(transport["开始时刻_s"])
        sample_gap: Dict[int, int] = {}
        for gap_index, gap in enumerate(joint.gaps):
            for sample_index in range(gap.start_index, gap.end_index + 1):
                sample_gap[sample_index] = gap_index
        for sample_index, sample in enumerate(joint.samples):
            direct = bool(joint.direct_flags[sample_index])
            relay = relay_by_gap.get((candidate_index, sample_gap.get(sample_index, -1)))
            method = "直连" if direct else ("中继" if relay is not None else "通信中断")
            records.append(
                {
                    "运输架次编号": transport["架次编号"],
                    "时刻_s": start + float(sample.time_s),
                    "阶段": sample.phase,
                    "经度": sample.point.longitude,
                    "纬度": sample.point.latitude,
                    "绝对高度_m": sample.point.altitude_m,
                    "固定网关直连可用": int(direct),
                    "通信方式": method,
                    "中继无人机编号": "" if relay is None else relay["中继无人机编号"],
                    "中继悬停点": "" if relay is None else relay["悬停点编号"],
                }
            )
    return records


def aggregate_relay_missions(model, global_relay_candidates: Sequence[object], relay_rows: Sequence[dict]):
    """按悬停点聚合中继任务，得到物理中继架次和合并后的能耗。"""
    candidate_by_id = {str(item.candidate_id): item for item in global_relay_candidates}
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in relay_rows:
        grouped[str(row["悬停点编号"])].append(row)

    physical_rows: List[dict] = []
    total_energy = 0.0
    for location_id, rows in grouped.items():
        candidate = candidate_by_id.get(location_id)
        if candidate is None:
            raise RuntimeError(f"找不到中继悬停点物理参数：{location_id}")
        first_row = min(rows, key=lambda row: float(row["中继起飞时刻_s"]))
        geometry = model.relay_flight_geometry(candidate)
        intervals = sorted((float(row["开始服务时刻_s"]), float(row["结束服务时刻_s"])) for row in rows)
        union_service_s = 0.0
        current_start = None
        current_end = None
        for start, end in intervals:
            if current_start is None:
                current_start, current_end = start, end
            elif start <= current_end + TOLERANCE:
                current_end = max(current_end, end)
            else:
                union_service_s += current_end - current_start
                current_start, current_end = start, end
        if current_start is not None:
            union_service_s += current_end - current_start

        relay_energy = geometry["flight_energy_kwh"] + (
            model.relay["hover_power_kw"] + model.relay["communication_power_kw"]
        ) * union_service_s / 3600.0
        return_soc = 100.0 * (1.0 - relay_energy / model.relay["usable_energy_kwh"])
        total_energy += relay_energy
        physical_rows.append(
            {
                "悬停点编号": location_id,
                "中继无人机编号": first_row["中继无人机编号"],
                "能源组件编号": first_row["能源组件编号"],
                "经度": candidate.longitude,
                "纬度": candidate.latitude,
                "地面高程_m": candidate.ground_altitude_m,
                "悬停离地高度_m": candidate.agl_m,
                "悬停绝对海拔_m": candidate.altitude_m,
                "中继任务数": len(rows),
                "首次起飞时刻_s": min(float(row["中继起飞时刻_s"]) for row in rows),
                "最后返航时刻_s": max(float(row["返回O01时刻_s"]) for row in rows),
                "能源组件最后可用时刻_s": max(float(row["能源组件再次可用时刻_s"]) for row in rows),
                "服务时间并集_s": union_service_s,
                "中继物理能耗_kWh": relay_energy,
                "物理返航SOC_pct": return_soc,
                "服务运输架次": "、".join(sorted({str(row["运输架次编号"]) for row in rows})),
            }
        )
    physical_rows.sort(key=lambda row: (float(row["首次起飞时刻_s"]), str(row["悬停点编号"])))
    for sequence, row in enumerate(physical_rows, 1):
        row["中继架次编号"] = f"Q3-R-{sequence:03d}"
    return physical_rows, total_energy


def write_outputs(
    output_dir: Path,
    base,
    model,
    args: argparse.Namespace,
    joint_candidates: Sequence[JointCandidate],
    alns_history: Sequence[dict],
    profile_rows: Sequence[dict],
    global_relay_candidates: Sequence[object],
    solution: JointSolution,
    transport_rows: Sequence[dict],
    relay_rows: Sequence[dict],
    delivery_rows: Sequence[dict],
    communication_rows: Sequence[dict],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    transport_output = [{key: value for key, value in row.items() if key != "candidate_index"} for row in transport_rows]
    relay_output = [{key: value for key, value in row.items() if key not in {"candidate_index", "gap_index", "option_index", "中继任务键"}} for row in relay_rows]
    physical_relay_rows, relay_energy = aggregate_relay_missions(model, global_relay_candidates, relay_rows)
    pd.DataFrame(transport_output).to_csv(output_dir / "问题三独立模型_运输架次.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(relay_output).to_csv(output_dir / "问题三独立模型_中继通信任务.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(physical_relay_rows).to_csv(output_dir / "问题三独立模型_中继架次.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(delivery_rows).to_csv(output_dir / "问题三独立模型_逐箱交付.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(communication_rows).to_csv(output_dir / "问题三独立模型_逐时刻通信.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(alns_history).to_csv(output_dir / "问题三独立模型_ALNS迭代记录.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(profile_rows).to_csv(output_dir / "问题三独立模型_候选通信剖面.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(solution.stage_log).to_csv(output_dir / "问题三独立模型_字典序求解记录.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "候选点编号": candidate.candidate_id,
                "经度": candidate.longitude,
                "纬度": candidate.latitude,
                "地面高程_m": candidate.ground_altitude_m,
                "离地高度_m": candidate.agl_m,
                "绝对海拔_m": candidate.altitude_m,
                "回传链路可用": int(candidate.gateway_link),
            }
            for candidate in global_relay_candidates
        ]
    ).to_csv(output_dir / "问题三独立模型_全局中继候选点.csv", index=False, encoding="utf-8-sig")

    transport_energy = float(sum(row["架次能耗_kWh"] for row in transport_rows))
    relay_task_energy = float(sum(row["中继能耗_kWh"] for row in relay_rows))
    weighted_tardiness = float(sum(float(row["优先系数"]) * float(row["期望时间迟到_s"]) for row in delivery_rows))
    desired_on_time = sum(float(row["期望时间迟到_s"]) <= TOLERANCE for row in delivery_rows)
    first_box_ids = {box_id for box_id, record in base.BOX_BY_ID.items() if bool(record["is_first_batch"])}
    medical_box_ids = {box_id for box_id, record in base.BOX_BY_ID.items() if bool(record["is_medical"])}
    delivery_by_box = {row["货箱编号"]: row for row in delivery_rows}
    first_on_time = sum(delivery_by_box[box_id]["硬时限是否满足"] == "是" for box_id in first_box_ids)
    medical_on_time = sum(delivery_by_box[box_id]["硬时限是否满足"] == "是" for box_id in medical_box_ids)
    direct_samples = sum(int(row["固定网关直连可用"]) for row in communication_rows)
    relay_samples = sum(row["通信方式"] == "中继" for row in communication_rows)
    interrupted_samples = sum(row["通信方式"] == "通信中断" for row in communication_rows)
    joint_makespan = max(
        [float(row["返回O01时刻_s"]) for row in transport_rows]
        + [float(row["最后返航时刻_s"]) for row in physical_relay_rows]
        + [0.0]
    )

    metrics = {
        "模型": "不依赖问题二结果的通信感知C-ALNS+CP-SAT联合模型",
        "结果来源": getattr(args, "solution_source", "完整候选池联合优化"),
        "加权迟到_优先系数乘秒": weighted_tardiness,
        "联合任务完成时间_s": joint_makespan,
        "运输能耗_kWh": transport_energy,
        "中继能耗_kWh": relay_energy,
        "中继任务逐项能耗之和_仅供诊断_kWh": relay_task_energy,
        "联合总能耗_kWh": transport_energy + relay_energy,
        "运输架次数": len(transport_rows),
        "中继架次数": len(physical_relay_rows),
        "中继通信任务数": len(relay_rows),
        "两类无人机总架次数": len(transport_rows) + len(physical_relay_rows),
        "期望时间内送达箱数": desired_on_time,
        "首批按时箱数": first_on_time,
        "医疗物资按时箱数": medical_on_time,
        "通信采样点数": len(communication_rows),
        "固定网关直连采样点数": direct_samples,
        "中继通信采样点数": relay_samples,
        "通信中断采样点数": interrupted_samples,
        "通信可行运输候选数": len(joint_candidates),
        "当前候选池已完成字典序阶段数": len(solution.stage_log),
        "当前候选池已证明最优阶段数": sum(row["已证明最优"] == "是" for row in solution.stage_log),
        "最优性口径": "仅对实际进入最终求解的有限通信可行候选池成立；若求解记录未显示OPTIMAL，则为当前时限内最好可行解",
    }
    pd.DataFrame([metrics]).to_csv(output_dir / "问题三独立模型_指标汇总.csv", index=False, encoding="utf-8-sig")

    coverage_count: Dict[str, int] = defaultdict(int)
    for row in transport_rows:
        for box_id in str(row["货箱编号列表"]).split("、"):
            coverage_count[box_id] += 1
    checks = [
        ("未读取问题二结果文件", True),
        ("80个货箱精确覆盖一次", len(coverage_count) == 80 and all(value == 1 for value in coverage_count.values())),
        ("全部硬时限满足", all(row["硬时限是否满足"] == "是" for row in delivery_rows)),
        ("全部通信采样点连续可用", interrupted_samples == 0),
        ("运输架次返航SOC不低于20%", all(float(row["返航SOC_pct"]) >= 20.0 - TOLERANCE for row in transport_rows)),
        ("中继物理架次返航SOC不低于20%", all(float(row["物理返航SOC_pct"]) >= 20.0 - TOLERANCE for row in physical_relay_rows)),
        ("中继悬停离地高度不超过300m", all(float(row["悬停离地高度_m"]) <= model.relay["max_hover_agl_m"] + TOLERANCE for row in relay_rows)),
        ("中继均在运输通信缺口前到位", all(float(row["中继起飞时刻_s"]) >= -TOLERANCE and float(row["中继起飞时刻_s"]) <= float(row["开始服务时刻_s"]) + TOLERANCE for row in relay_rows)),
    ]
    pd.DataFrame([{"检查项目": name, "是否通过": "是" if passed else "否"} for name, passed in checks]).to_csv(output_dir / "问题三独立模型_约束检查.csv", index=False, encoding="utf-8-sig")

    config = vars(args).copy()
    config.update(
        {
            "generated_at": "2026-09-24",
            "reads_q2_result": False,
            "raw_data_root": str(ROOT / "数据" / "无人机应急物资运输基础数据"),
            "output_dir": str(output_dir),
            "transport_candidate_count": len(joint_candidates),
            "global_relay_candidate_count": len(global_relay_candidates),
        }
    )
    with (output_dir / "问题三独立模型_运行配置.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)

    print("\n独立联合优化结果：")
    print(f"加权迟到={weighted_tardiness:.3f}，联合完工时间={joint_makespan:.2f}s")
    print(f"运输能耗={transport_energy:.6f}kWh，中继能耗={relay_energy:.6f}kWh，联合能耗={transport_energy + relay_energy:.6f}kWh")
    print(f"运输架次={len(transport_rows)}，中继物理架次={len(physical_relay_rows)}，中继通信任务={len(relay_rows)}，通信中断采样点={interrupted_samples}")
    print(f"输出目录：{output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="问题三：不依赖问题二结果的C-ALNS+CP-SAT运输与中继联合优化")
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--alns-steps", type=int, default=2500)
    parser.add_argument("--alns-segment", type=int, default=100)
    parser.add_argument("--cooling", type=float, default=0.998)
    parser.add_argument("--pair-limit", type=int, default=500)
    parser.add_argument("--max-route-groups", type=int, default=400)
    parser.add_argument("--max-route-candidates", type=int, default=300)
    parser.add_argument("--min-relay-tasks", type=int, default=1)
    parser.add_argument("--singleton-options-per-box", type=int, default=3)
    parser.add_argument("--mandatory-options-per-group", type=int, default=2)
    parser.add_argument("--max-orders-per-drone", type=int, default=8)
    parser.add_argument("--seed-route-file", default=str(DEFAULT_SEED_ROUTE_FILE))
    parser.add_argument("--sample-step", type=float, default=10.0)
    parser.add_argument("--los-step", type=float, default=30.0)
    parser.add_argument("--heights", type=float, nargs="+", default=[100.0, 200.0, 300.0])
    parser.add_argument("--relay-grid", type=float, default=750.0)
    parser.add_argument("--max-relay-base-points", type=int, default=100)
    parser.add_argument("--dynamic-points-per-gap", type=int, default=6)
    parser.add_argument("--relay-test-limit", type=int, default=60)
    parser.add_argument("--max-relay-options", type=int, default=4)
    parser.add_argument("--max-relay-block", type=float, default=3600.0)
    parser.add_argument("--relay-drone-count", type=int, default=2)
    parser.add_argument("--relay-component-count", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=40_000)
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--feasibility-time", type=float, default=120.0)
    parser.add_argument("--seed-time-limit", type=float, default=60.0)
    parser.add_argument("--seed-feasibility-time", type=float, default=60.0)
    parser.add_argument("--cover-time-limit", type=float, default=60.0)
    parser.add_argument("--transport-seed-time-limit", type=float, default=90.0)
    parser.add_argument("--transport-seed-stages", type=int, choices=[1, 2, 3, 4], default=3)
    parser.add_argument("--seed-iterations", type=int, default=20)
    parser.add_argument("--master-time-limit", type=float, default=30.0)
    parser.add_argument("--max-stages", type=int, choices=[1, 2, 3, 4], default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--objective", choices=["timeliness", "relayaware", "makespan", "energy", "sortie"], default="relayaware")
    parser.add_argument("--solver-log", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="使用小候选池进行快速链路和模型冒烟测试")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()
    if args.smoke:
        args.alns_steps = min(args.alns_steps, 80)
        args.alns_segment = min(args.alns_segment, 20)
        args.pair_limit = min(args.pair_limit, 80)
        args.max_route_groups = min(args.max_route_groups, 120)
        args.max_route_candidates = min(args.max_route_candidates, 260)
        args.singleton_options_per_box = 1
        args.mandatory_options_per_group = 1
        args.max_relay_block = max(args.max_relay_block, 3600.0)
        args.max_orders_per_drone = min(args.max_orders_per_drone, 2)
        args.sample_step = max(args.sample_step, 30.0)
        args.los_step = max(args.los_step, 90.0)
        args.relay_grid = max(args.relay_grid, 1500.0)
        args.max_relay_base_points = min(args.max_relay_base_points, 40)
        args.dynamic_points_per_gap = min(args.dynamic_points_per_gap, 3)
        args.relay_test_limit = min(args.relay_test_limit, 20)
        args.max_relay_options = min(args.max_relay_options, 2)
        args.time_limit = min(args.time_limit, 30.0)
        args.feasibility_time = min(args.feasibility_time, 30.0)
        args.seed_time_limit = min(args.seed_time_limit, 15.0)
        args.seed_feasibility_time = min(args.seed_feasibility_time, 15.0)
        args.cover_time_limit = min(args.cover_time_limit, 15.0)
        args.transport_seed_time_limit = min(args.transport_seed_time_limit, 20.0)
        args.transport_seed_stages = min(args.transport_seed_stages, 2)
        args.seed_iterations = min(args.seed_iterations, 5)
        args.master_time_limit = min(args.master_time_limit, 10.0)
        args.max_stages = 1
    return args


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    print("初始化原始80箱数据、DEM、运输资源、通信参数和中继资源……")
    print("确认：本程序不读取问题二运输方案或问题二结果目录。")
    print(f"通信缺口覆盖模式：每个选中候选的缺口必须绑定中继；全局最少中继任务={args.min_relay_tasks}。")
    model, base = initialize_physics(args)

    print("\n[1/5] C-ALNS从原始货箱生成候选组批……")
    pool, alns_history, seeds = generate_route_groups(base, args)
    transport_candidates = select_transport_candidates(pool, args)
    print(f"候选组批={len(pool.groups)}，组批-机型-航序候选={len(transport_candidates)}。")

    print("\n[2/5] 生成与运输方案无关的全局中继悬停候选……")
    global_relay_candidates = build_global_relay_candidates(model, args)
    print(f"全局中继候选点={len(global_relay_candidates)}。")

    print("\n[3/5] 计算每个运输候选的直连缺口和中继选项……")
    joint_candidates, profile_rows = profile_joint_candidates(model, transport_candidates, global_relay_candidates, args)
    print(f"通信可行运输候选={len(joint_candidates)}。")

    print("\n[4/6] 先求运输与中继资源同时可行的通信感知热启动……")
    seed_candidates: List[JointCandidate] = []
    seed_solution: Optional[JointSolution] = None
    try:
        decomposition_seed = find_decomposition_seed(base, joint_candidates, args)
        if decomposition_seed is None:
            raise RuntimeError("分解式热启动在规定轮数内未找到联合可行解")
        seed_candidates, seed_solution = decomposition_seed
    except RuntimeError as seed_error:
        print(f"  分解式运输-中继热启动失败：{seed_error}")
        print("  回退到保守容量热启动。")
        try:
            seed_candidates, transport_start_hint, relay_option_hint, _ = find_transport_resource_seed(base, joint_candidates, args)
            seed_model, seed_variables = build_joint_model(base, seed_candidates, args)
            add_transport_seed_hint(seed_model, seed_variables, seed_candidates, transport_start_hint, relay_option_hint)
            fix_transport_seed_decisions(seed_model, seed_variables, seed_candidates, transport_start_hint, relay_option_hint)
            seed_solution = solve_joint_model(seed_model, seed_variables, argparse.Namespace(**{**vars(args), "max_stages": 1, "time_limit": args.seed_time_limit, "feasibility_time": args.seed_feasibility_time}))
        except RuntimeError as fallback_error:
            print(f"  保守容量热启动同样失败：{fallback_error}")
            print("  不再异常退出，继续直接求完整候选池联合模型。")

    print("\n[5/6] 建立并求解运输-通信-中继联合CP-SAT模型……")
    cp_model_instance, variables = build_joint_model(base, joint_candidates, args)
    if seed_solution is not None:
        add_solution_hint(cp_model_instance, variables, joint_candidates, seed_candidates, seed_solution)
    solution_candidates = joint_candidates
    try:
        solution = solve_joint_model(cp_model_instance, variables, args)
        args.solution_source = "完整候选池联合优化"
    except RuntimeError as full_error:
        if seed_solution is None:
            raise RuntimeError(f"完整联合模型未找到可行解，且没有可用热启动：{full_error}") from full_error
        print(f"  完整候选池在当前时限内未返回可行解：{full_error}")
        print("  输出已通过全部约束的联合热启动方案；该方案是可行上界，不宣称完整候选池最优。")
        solution_candidates = seed_candidates
        solution = seed_solution
        args.solution_source = "运输-中继资源可行热启动（完整候选池求解未返回可行解）"

    print("\n[6/6] 恢复具体资源编号并执行独立检查……")
    transport_rows, relay_rows, delivery_rows, relay_by_gap = extract_solution(base, solution_candidates, solution, args)
    communication_rows = communication_records(solution_candidates, transport_rows, relay_by_gap)
    write_outputs(output_dir, base, model, args, solution_candidates, alns_history, profile_rows, global_relay_candidates, solution, transport_rows, relay_rows, delivery_rows, communication_rows)


if __name__ == "__main__":
    main()
