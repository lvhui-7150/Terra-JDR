"""问题二第一小问：异构无人机多点多架次运输调度。

求解框架：
1. 按题设水平直线航段和 DEM 净空规则预计算任意节点间航段；
2. 将80个货箱组合成可访问多个服务区的候选架次；
3. 每个架次逐段使用当前剩余载荷计算能耗和送达时刻；
4. 显式安排具体实体无人机、共享电池、开始时刻和充电周转；
5. 用确定性多起点局部搜索比较及时性、完工时间、能耗和架次数优先方案。

本程序给出经过完整约束检查的启发式可行解，不宣称全局最优。增加
SEARCH_ITERATIONS 或 MAX_STOPS_PER_SORTIE 可以扩大搜索，但会增加运行时间。
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, permutations
import importlib.util
import math
from pathlib import Path
import random
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ==============================
# 1. 文件位置与求解参数
# ==============================

ROOT = Path(__file__).resolve().parent
BASE_DATA = ROOT / "数据" / "无人机应急物资运输基础数据"
OUTPUT_DIR = ROOT / "问题二第一问结果"
Q1_PLAN_FILE = ROOT / "问题一第二问结果" / "问题一第二问_字典序主方案.csv"

RANDOM_SEED = 20260923
MAX_STOPS_PER_SORTIE = 5
SEARCH_ITERATIONS = 1200
DEM_SAMPLE_STEP_M = 1.0
TOLERANCE = 1e-8

SCENARIOS = {
    "及时性优先": "硬时限 → 加权迟到 → 全部任务完成时间 → 能耗 → 架次数",
    "完工时间优先": "硬时限 → 全部任务完成时间 → 加权迟到 → 能耗 → 架次数",
    "能耗优先": "硬时限 → 能耗 → 加权迟到 → 全部任务完成时间 → 架次数",
    "架次优先": "硬时限 → 架次数 → 加权迟到 → 全部任务完成时间 → 能耗",
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


Q1 = load_module("q1_physics_for_q2", ROOT / "问题一_第一问.py")


@dataclass(frozen=True)
class Segment:
    start_node: str
    end_node: str
    distance_m: float
    max_terrain_m: float
    cruise_altitude_m: float
    climb_m: float
    descent_m: float


@dataclass(frozen=True)
class RouteOption:
    drone_type: str
    service_order: Tuple[str, ...]
    total_mass_kg: float
    total_volume_m3: float
    energy_kwh: float
    energy_limit_kwh: float
    return_soc_pct: float
    duration_s: float
    takeoff_offset_s: float
    charge_time_s: float
    delivery_offsets: Tuple[Tuple[str, float], ...]


@dataclass
class ScheduleResult:
    scenario: str
    priority: str
    partition: Tuple[Tuple[str, ...], ...]
    sorties: List[dict]
    deliveries: List[dict]
    hard_violation_count: int
    hard_lateness_s: float
    weighted_tardiness: float
    makespan_s: float
    total_energy_kwh: float
    sortie_count: int
    desired_on_time_count: int
    first_batch_on_time_count: int
    medical_on_time_count: int


BOXES = pd.DataFrame()
BOX_BY_ID: Dict[str, dict] = {}
DRONES: Dict[str, object] = {}
SEGMENTS: Dict[Tuple[str, str], Segment] = {}
DRONE_RESOURCES: Dict[str, Tuple[str, ...]] = {}
BATTERY_RESOURCES: Dict[str, Tuple[str, ...]] = {}
FULL_CHARGE_TIME: Dict[str, float] = {}


# ==============================
# 2. 数据读取
# ==============================


def read_boxes() -> pd.DataFrame:
    file = BASE_DATA / "物资需求与配送时限.xlsx"
    boxes = pd.read_excel(file, sheet_name="逐箱货箱清单", header=0)
    boxes = boxes.rename(
        columns={
            "货箱编号": "box_id",
            "服务区编号": "service_id",
            "物资类型": "material_type",
            "单箱质量（kg）": "mass_kg",
            "单箱体积（m³）": "volume_m3",
            "是否首批保障": "is_first_batch",
            "首批截止时间（s）": "first_deadline_s",
            "期望送达时间（s）": "desired_time_s",
            "应急优先系数": "priority",
        }
    )
    required = [
        "box_id",
        "service_id",
        "material_type",
        "mass_kg",
        "volume_m3",
        "is_first_batch",
        "first_deadline_s",
        "desired_time_s",
        "priority",
    ]
    boxes = boxes[required].copy()
    boxes["box_id"] = boxes["box_id"].astype(str)
    boxes["service_id"] = boxes["service_id"].astype(str)
    boxes["is_first_batch"] = boxes["is_first_batch"].eq("是")
    for column in [
        "mass_kg",
        "volume_m3",
        "first_deadline_s",
        "desired_time_s",
        "priority",
    ]:
        boxes[column] = pd.to_numeric(boxes[column], errors="coerce")

    boxes["is_medical"] = boxes["material_type"].eq("医疗物资")
    medical_deadline = boxes["desired_time_s"].where(boxes["is_medical"])
    first_batch_deadline = boxes["first_deadline_s"].where(
        boxes["is_first_batch"]
    )
    boxes["hard_deadline_s"] = pd.concat(
        [medical_deadline, first_batch_deadline], axis=1
    ).min(axis=1, skipna=True)

    if len(boxes) != 80 or boxes["box_id"].duplicated().any():
        raise ValueError("逐箱货箱清单必须包含80个不重复货箱。")
    if boxes[["mass_kg", "volume_m3", "desired_time_s", "priority"]].isna().any().any():
        raise ValueError("货箱质量、体积、期望时间或优先系数存在缺失。")
    if boxes.loc[boxes["is_first_batch"], "first_deadline_s"].isna().any():
        raise ValueError("首批保障货箱缺少首批截止时间。")
    return boxes


def read_transport_resources() -> Tuple[
    Dict[str, Tuple[str, ...]], Dict[str, Tuple[str, ...]], Dict[str, float]
]:
    file = BASE_DATA / "运输无人机数据.xlsx"
    raw = pd.read_excel(file, sheet_name="数据", header=None)

    drone_rows = raw.iloc[8:16, [0, 1]].copy()
    drone_rows.columns = ["drone_id", "drone_type"]
    drone_rows = drone_rows.dropna()
    drone_resources = {
        drone_type: tuple(
            sorted(drone_rows.loc[drone_rows["drone_type"].eq(drone_type), "drone_id"].astype(str))
        )
        for drone_type in ["A", "B", "C"]
    }

    battery_rows = raw.iloc[19:22, [0, 1, 2]].copy()
    battery_rows.columns = ["drone_type", "battery_count", "full_charge_time_s"]
    battery_rows = battery_rows.dropna()
    battery_resources: Dict[str, Tuple[str, ...]] = {}
    full_charge_time: Dict[str, float] = {}
    for _, row in battery_rows.iterrows():
        drone_type = str(row["drone_type"])
        count = int(row["battery_count"])
        battery_resources[drone_type] = tuple(
            f"{drone_type}-BAT-{index:02d}" for index in range(1, count + 1)
        )
        full_charge_time[drone_type] = float(row["full_charge_time_s"])

    expected_drones = {"A": 4, "B": 2, "C": 2}
    expected_batteries = {"A": 6, "B": 4, "C": 4}
    if {key: len(value) for key, value in drone_resources.items()} != expected_drones:
        raise ValueError("实体运输无人机数量与附件不一致。")
    if {key: len(value) for key, value in battery_resources.items()} != expected_batteries:
        raise ValueError("共享电池库存与附件不一致。")
    return drone_resources, battery_resources, full_charge_time


def read_nodes() -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    center, services = Q1.read_center_and_services()
    center_row = pd.DataFrame(
        [
            {
                "node_id": center["id"],
                "longitude": center["longitude"],
                "latitude": center["latitude"],
                "ground_altitude_m": center["ground_altitude_m"],
                "work_altitude_m": center["ground_altitude_m"],
            }
        ]
    )
    service_rows = services.rename(columns={"service_id": "node_id"}).copy()
    service_rows["work_altitude_m"] = service_rows["ground_altitude_m"] + 30.0
    nodes = pd.concat([center_row, service_rows], ignore_index=True)
    return center, services, nodes


# ==============================
# 3. 任意任务节点间 DEM 航段
# ==============================


def build_all_segments(center: dict, nodes: pd.DataFrame) -> Dict[Tuple[str, str], Segment]:
    longitude, latitude, elevation = Q1.load_dem()
    node_map = nodes.set_index("node_id").to_dict("index")
    node_ids = list(nodes["node_id"])
    segment_map: Dict[Tuple[str, str], Segment] = {}

    for node_a, node_b in combinations(node_ids, 2):
        start = node_map[node_a]
        end = node_map[node_b]
        xy_x, xy_y = Q1.local_xy(
            np.array([start["longitude"], end["longitude"]]),
            np.array([start["latitude"], end["latitude"]]),
            center["longitude"],
            center["latitude"],
        )
        distance_m = float(np.hypot(xy_x[1] - xy_x[0], xy_y[1] - xy_y[0]))
        sample_count = max(2, int(np.ceil(distance_m / DEM_SAMPLE_STEP_M)) + 1)
        ratio = np.linspace(0.0, 1.0, sample_count)
        profile_lon = start["longitude"] + ratio * (end["longitude"] - start["longitude"])
        profile_lat = start["latitude"] + ratio * (end["latitude"] - start["latitude"])
        profile = Q1.sample_dem_nearest(
            profile_lon, profile_lat, longitude, latitude, elevation
        )
        max_terrain_m = float(np.max(profile))
        cruise_altitude_m = max_terrain_m + 50.0

        for from_id, to_id in [(node_a, node_b), (node_b, node_a)]:
            from_node = node_map[from_id]
            to_node = node_map[to_id]
            climb_m = cruise_altitude_m - float(from_node["work_altitude_m"])
            descent_m = cruise_altitude_m - float(to_node["work_altitude_m"])
            if climb_m < -TOLERANCE or descent_m < -TOLERANCE:
                raise ValueError(f"航段 {from_id}->{to_id} 的巡航海拔低于节点作业高度。")
            segment_map[(from_id, to_id)] = Segment(
                start_node=from_id,
                end_node=to_id,
                distance_m=distance_m,
                max_terrain_m=max_terrain_m,
                cruise_altitude_m=cruise_altitude_m,
                climb_m=max(0.0, climb_m),
                descent_m=max(0.0, descent_m),
            )
    return segment_map


def segment_flight_time(drone, segment: Segment) -> float:
    return (
        segment.climb_m / drone.climb_speed_mps
        + segment.distance_m / drone.cruise_speed_mps
        + segment.descent_m / drone.descent_speed_mps
    )


def battery_charge_time(drone_type: str, return_soc: float) -> float:
    full_time = FULL_CHARGE_TIME[drone_type]
    soc = float(np.clip(return_soc, 0.0, 1.0))
    if soc < 0.90:
        return full_time * (0.65 * (0.90 - soc) / 0.90 + 0.35)
    return full_time * 0.35 * (1.0 - soc) / 0.10


def segment_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "起点": segment.start_node,
                "终点": segment.end_node,
                "水平距离_m": segment.distance_m,
                "沿线最高地面高程_m": segment.max_terrain_m,
                "巡航海拔_m": segment.cruise_altitude_m,
                "爬升高度_m": segment.climb_m,
                "下降高度_m": segment.descent_m,
            }
            for segment in SEGMENTS.values()
        ]
    ).sort_values(["起点", "终点"])


# ==============================
# 4. 多点架次物理计算
# ==============================


def route_box_records(box_ids: Sequence[str]) -> List[dict]:
    return [BOX_BY_ID[box_id] for box_id in box_ids]


def route_basic_totals(box_ids: Sequence[str]) -> Tuple[float, float, Tuple[str, ...]]:
    records = route_box_records(box_ids)
    mass = float(sum(record["mass_kg"] for record in records))
    volume = float(sum(record["volume_m3"] for record in records))
    services = tuple(sorted({record["service_id"] for record in records}))
    return mass, volume, services


def hard_deadline(record: dict) -> Optional[float]:
    value = record["hard_deadline_s"]
    return None if pd.isna(value) else float(value)


@lru_cache(maxsize=100_000)
def feasible_route_options(box_ids_key: Tuple[str, ...]) -> Tuple[RouteOption, ...]:
    box_ids = tuple(sorted(box_ids_key))
    total_mass, total_volume, services = route_basic_totals(box_ids)
    if not box_ids or len(services) > MAX_STOPS_PER_SORTIE:
        return tuple()

    boxes_by_service: Dict[str, List[dict]] = defaultdict(list)
    for record in route_box_records(box_ids):
        boxes_by_service[record["service_id"]].append(record)

    options: List[RouteOption] = []
    for drone_type, drone in sorted(DRONES.items()):
        if total_mass > drone.max_payload_kg + TOLERANCE:
            continue
        if total_volume > drone.volume_m3 + TOLERANCE:
            continue

        best_option: Optional[RouteOption] = None
        best_order_key: Optional[Tuple[float, ...]] = None
        for service_order in permutations(services):
            current_node = "O01"
            remaining_mass = total_mass
            current_time = (
                drone.preparation_time_s
                + len(box_ids) * drone.loading_time_per_box_s
            )
            takeoff_offset = current_time
            total_energy = 0.0
            delivery_offsets: Dict[str, float] = {}

            for service_id in service_order:
                segment = SEGMENTS[(current_node, service_id)]
                current_time += segment_flight_time(drone, segment)
                total_energy += Q1.segment_energy(
                    drone, segment.distance_m, segment.climb_m, remaining_mass
                )

                service_boxes = boxes_by_service[service_id]
                current_time += (
                    drone.handover_base_time_s
                    + len(service_boxes) * drone.handover_per_box_time_s
                )
                for record in service_boxes:
                    delivery_offsets[record["box_id"]] = current_time
                remaining_mass -= sum(record["mass_kg"] for record in service_boxes)
                current_node = service_id

            return_segment = SEGMENTS[(current_node, "O01")]
            current_time += segment_flight_time(drone, return_segment)
            total_energy += Q1.segment_energy(
                drone, return_segment.distance_m, return_segment.climb_m, 0.0
            )

            energy_limit = Q1.task_energy_limit(drone)
            if total_energy > energy_limit + TOLERANCE:
                continue

            hard_count = 0
            hard_seconds = 0.0
            weighted_completion = 0.0
            for record in route_box_records(box_ids):
                offset = delivery_offsets[record["box_id"]]
                deadline = hard_deadline(record)
                if deadline is not None and offset > deadline + TOLERANCE:
                    hard_count += 1
                    hard_seconds += offset - deadline
                weighted_completion += float(record["priority"]) * offset

            order_key = (
                float(hard_count),
                hard_seconds,
                weighted_completion,
                current_time,
                total_energy,
            )
            return_soc = Q1.return_soc_ratio(drone, total_energy)
            option = RouteOption(
                drone_type=drone_type,
                service_order=tuple(service_order),
                total_mass_kg=total_mass,
                total_volume_m3=total_volume,
                energy_kwh=total_energy,
                energy_limit_kwh=energy_limit,
                return_soc_pct=100.0 * return_soc,
                duration_s=current_time,
                takeoff_offset_s=takeoff_offset,
                charge_time_s=battery_charge_time(drone_type, return_soc),
                delivery_offsets=tuple(sorted(delivery_offsets.items())),
            )
            if best_order_key is None or order_key < best_order_key:
                best_order_key = order_key
                best_option = option

        if best_option is not None:
            options.append(best_option)
    return tuple(options)


def normalize_partition(routes: Iterable[Iterable[str]]) -> Tuple[Tuple[str, ...], ...]:
    normalized = [tuple(sorted(route)) for route in routes if route]
    return tuple(sorted(normalized))


def partition_is_physically_feasible(partition: Tuple[Tuple[str, ...], ...]) -> bool:
    flattened = [box_id for route in partition for box_id in route]
    if len(flattened) != len(BOX_BY_ID) or set(flattened) != set(BOX_BY_ID):
        return False
    if len(flattened) != len(set(flattened)):
        return False
    return all(feasible_route_options(route) for route in partition)


# ==============================
# 5. 初始货箱组批
# ==============================


def q1_initial_partition() -> Tuple[Tuple[str, ...], ...]:
    if not Q1_PLAN_FILE.exists():
        raise FileNotFoundError(f"缺少问题一主方案：{Q1_PLAN_FILE}")
    plan = pd.read_csv(Q1_PLAN_FILE, encoding="utf-8-sig")
    routes = [
        tuple(str(value).split("、"))
        for value in plan["货箱编号列表"].astype(str)
    ]
    partition = normalize_partition(routes)
    if not partition_is_physically_feasible(partition):
        raise AssertionError("问题一主方案无法作为问题二初始可行分组。")
    return partition


def ordered_box_ids(mode: str) -> List[str]:
    rows = list(BOX_BY_ID.values())
    if mode == "deadline":
        rows.sort(
            key=lambda record: (
                hard_deadline(record) if hard_deadline(record) is not None else math.inf,
                float(record["desired_time_s"]),
                -float(record["priority"]),
                -float(record["mass_kg"]),
                record["box_id"],
            )
        )
    elif mode == "mass":
        rows.sort(
            key=lambda record: (
                -float(record["mass_kg"]),
                hard_deadline(record) if hard_deadline(record) is not None else math.inf,
                record["box_id"],
            )
        )
    elif mode == "service":
        rows.sort(
            key=lambda record: (
                record["service_id"],
                hard_deadline(record) if hard_deadline(record) is not None else math.inf,
                -float(record["priority"]),
                record["box_id"],
            )
        )
    else:
        raise ValueError(f"未知组批顺序：{mode}")
    return [record["box_id"] for record in rows]


def insertion_score(route: Tuple[str, ...], is_new: bool, mode: str) -> Tuple[float, ...]:
    options = feasible_route_options(tuple(sorted(route)))
    best_option = min(options, key=lambda option: (option.energy_kwh, option.duration_s))
    records = route_box_records(route)
    deadlines = [
        hard_deadline(record)
        for record in records
        if hard_deadline(record) is not None
    ]
    deadline_spread = 0.0 if len(deadlines) < 2 else max(deadlines) - min(deadlines)
    drone = DRONES[best_option.drone_type]
    residual = (
        (drone.max_payload_kg - best_option.total_mass_kg) / drone.max_payload_kg
        + (drone.volume_m3 - best_option.total_volume_m3) / drone.volume_m3
    )
    services = len(best_option.service_order)

    if mode == "deadline":
        return (float(is_new), deadline_spread, residual, services, best_option.energy_kwh)
    if mode == "mass":
        return (float(is_new), residual, services, best_option.energy_kwh, deadline_spread)
    return (float(is_new), services, deadline_spread, residual, best_option.energy_kwh)


def greedy_partition(mode: str) -> Tuple[Tuple[str, ...], ...]:
    routes: List[List[str]] = []
    for box_id in ordered_box_ids(mode):
        choices: List[Tuple[Tuple[float, ...], Optional[int], Tuple[str, ...]]] = []
        for route_index, route in enumerate(routes):
            candidate = tuple(sorted(route + [box_id]))
            if feasible_route_options(candidate):
                choices.append((insertion_score(candidate, False, mode), route_index, candidate))

        new_route = (box_id,)
        if feasible_route_options(new_route):
            choices.append((insertion_score(new_route, True, mode), None, new_route))
        if not choices:
            raise RuntimeError(f"货箱 {box_id} 无法放入任何可行架次。")

        _, route_index, selected = min(choices, key=lambda item: item[0])
        if route_index is None:
            routes.append(list(selected))
        else:
            routes[route_index] = list(selected)

    partition = normalize_partition(routes)
    if not partition_is_physically_feasible(partition):
        raise AssertionError(f"{mode} 初始组批未通过物理可行性检查。")
    return partition


# ==============================
# 6. 实体无人机与共享电池调度
# ==============================


def route_assignment_metrics(
    route: Tuple[str, ...], option: RouteOption, start_time: float
) -> dict:
    delivery_times = {
        box_id: start_time + offset for box_id, offset in option.delivery_offsets
    }
    hard_count = 0
    hard_seconds = 0.0
    weighted_tardiness = 0.0
    earliest_hard = math.inf
    earliest_desired = math.inf
    minimum_hard_slack = math.inf
    for box_id in route:
        record = BOX_BY_ID[box_id]
        delivered = delivery_times[box_id]
        deadline = hard_deadline(record)
        if deadline is not None:
            earliest_hard = min(earliest_hard, deadline)
            minimum_hard_slack = min(minimum_hard_slack, deadline - delivered)
            if delivered > deadline + TOLERANCE:
                hard_count += 1
                hard_seconds += delivered - deadline
        desired = float(record["desired_time_s"])
        earliest_desired = min(earliest_desired, desired)
        weighted_tardiness += float(record["priority"]) * max(0.0, delivered - desired)
    return {
        "delivery_times": delivery_times,
        "hard_count": hard_count,
        "hard_seconds": hard_seconds,
        "weighted_tardiness": weighted_tardiness,
        "earliest_hard": earliest_hard,
        "earliest_desired": earliest_desired,
        "minimum_hard_slack": minimum_hard_slack,
        "return_time": start_time + option.duration_s,
    }


def assignment_key(scenario: str, metrics: dict, option: RouteOption) -> Tuple[float, ...]:
    common = (
        float(metrics["hard_count"]),
        metrics["hard_seconds"],
        metrics["minimum_hard_slack"],
        metrics["earliest_hard"],
    )
    desired_key = (metrics["earliest_desired"],)
    if scenario == "及时性优先":
        return common + (
            metrics["weighted_tardiness"],
            *desired_key,
            metrics["return_time"],
            option.energy_kwh,
        )
    if scenario == "完工时间优先":
        return common + (
            metrics["return_time"],
            *desired_key,
            metrics["weighted_tardiness"],
            option.energy_kwh,
        )
    if scenario == "能耗优先":
        return common + (
            option.energy_kwh,
            *desired_key,
            metrics["weighted_tardiness"],
            metrics["return_time"],
        )
    return common + (
        metrics["weighted_tardiness"],
        *desired_key,
        metrics["return_time"],
        option.energy_kwh,
    )


def schedule_partition(
    partition: Tuple[Tuple[str, ...], ...], scenario: str
) -> ScheduleResult:
    if scenario not in SCENARIOS:
        raise ValueError(f"未知优化情景：{scenario}")
    if not partition_is_physically_feasible(partition):
        raise ValueError("输入组批不是完整物理可行分区。")

    drone_ready = {
        drone_id: 0.0
        for drone_ids in DRONE_RESOURCES.values()
        for drone_id in drone_ids
    }
    battery_ready = {
        battery_id: 0.0
        for battery_ids in BATTERY_RESOURCES.values()
        for battery_id in battery_ids
    }
    unscheduled = list(partition)
    sorties: List[dict] = []
    deliveries: List[dict] = []

    while unscheduled:
        candidates: List[Tuple[Tuple[float, ...], tuple]] = []
        for route_index, route in enumerate(unscheduled):
            for option in feasible_route_options(route):
                drone_id = min(
                    DRONE_RESOURCES[option.drone_type],
                    key=lambda resource_id: (drone_ready[resource_id], resource_id),
                )
                battery_id = min(
                    BATTERY_RESOURCES[option.drone_type],
                    key=lambda resource_id: (battery_ready[resource_id], resource_id),
                )
                start_time = max(drone_ready[drone_id], battery_ready[battery_id])
                metrics = route_assignment_metrics(route, option, start_time)
                key = assignment_key(scenario, metrics, option)
                candidates.append(
                    (
                        key,
                        (
                            route_index,
                            route,
                            option,
                            drone_id,
                            battery_id,
                            start_time,
                            metrics,
                        ),
                    )
                )

        _, selected = min(candidates, key=lambda item: item[0])
        (
            route_index,
            route,
            option,
            drone_id,
            battery_id,
            start_time,
            metrics,
        ) = selected
        unscheduled.pop(route_index)

        sortie_id = f"Q2-{len(sorties) + 1:03d}"
        return_time = metrics["return_time"]
        charge_complete = return_time + option.charge_time_s
        drone_ready[drone_id] = return_time
        battery_ready[battery_id] = charge_complete

        route_string = "O01→" + "→".join(option.service_order) + "→O01"
        sorties.append(
            {
                "架次编号": sortie_id,
                "无人机编号": drone_id,
                "机型编号": option.drone_type,
                "电池编号": battery_id,
                "开始时刻_s": start_time,
                "起飞时刻_s": start_time + option.takeoff_offset_s,
                "访问服务区顺序": "→".join(option.service_order),
                "完整路线": route_string,
                "货箱编号列表": "、".join(route),
                "货箱数量": len(route),
                "总质量_kg": option.total_mass_kg,
                "总体积_m3": option.total_volume_m3,
                "返回O01时刻_s": return_time,
                "架次持续时间_s": option.duration_s,
                "架次能耗_kWh": option.energy_kwh,
                "任务能量上限_kWh": option.energy_limit_kwh,
                "返航SOC_pct": option.return_soc_pct,
                "充电完成时刻_s": charge_complete,
            }
        )

        for box_id in route:
            record = BOX_BY_ID[box_id]
            delivered = metrics["delivery_times"][box_id]
            deadline = hard_deadline(record)
            deliveries.append(
                {
                    "货箱编号": box_id,
                    "架次编号": sortie_id,
                    "服务区编号": record["service_id"],
                    "物资类型": record["material_type"],
                    "是否首批保障": "是" if record["is_first_batch"] else "否",
                    "硬截止时间_s": np.nan if deadline is None else deadline,
                    "期望送达时间_s": float(record["desired_time_s"]),
                    "应急优先系数": float(record["priority"]),
                    "交付完成时刻_s": delivered,
                    "硬时限是否满足": "是" if deadline is None or delivered <= deadline + TOLERANCE else "否",
                    "期望时间迟到_s": max(0.0, delivered - float(record["desired_time_s"])),
                }
            )

    delivery_frame = pd.DataFrame(deliveries)
    hard_rows = delivery_frame[delivery_frame["硬截止时间_s"].notna()]
    hard_lateness = np.maximum(
        0.0,
        hard_rows["交付完成时刻_s"].to_numpy(dtype=float)
        - hard_rows["硬截止时间_s"].to_numpy(dtype=float),
    )
    weighted_tardiness = float(
        (
            delivery_frame["应急优先系数"]
            * delivery_frame["期望时间迟到_s"]
        ).sum()
    )
    makespan = max(row["返回O01时刻_s"] for row in sorties)
    total_energy = sum(row["架次能耗_kWh"] for row in sorties)
    desired_on_time = int((delivery_frame["期望时间迟到_s"] <= TOLERANCE).sum())
    first_rows = delivery_frame[delivery_frame["是否首批保障"].eq("是")]
    medical_rows = delivery_frame[delivery_frame["物资类型"].eq("医疗物资")]

    return ScheduleResult(
        scenario=scenario,
        priority=SCENARIOS[scenario],
        partition=partition,
        sorties=sorties,
        deliveries=deliveries,
        hard_violation_count=int((hard_lateness > TOLERANCE).sum()),
        hard_lateness_s=float(hard_lateness.sum()),
        weighted_tardiness=weighted_tardiness,
        makespan_s=float(makespan),
        total_energy_kwh=float(total_energy),
        sortie_count=len(sorties),
        desired_on_time_count=desired_on_time,
        first_batch_on_time_count=int(first_rows["硬时限是否满足"].eq("是").sum()),
        medical_on_time_count=int(medical_rows["硬时限是否满足"].eq("是").sum()),
    )


def objective_key(result: ScheduleResult) -> Tuple[float, ...]:
    common = (float(result.hard_violation_count), result.hard_lateness_s)
    if result.scenario == "及时性优先":
        return common + (
            result.weighted_tardiness,
            result.makespan_s,
            result.total_energy_kwh,
            float(result.sortie_count),
        )
    if result.scenario == "完工时间优先":
        return common + (
            result.makespan_s,
            result.weighted_tardiness,
            result.total_energy_kwh,
            float(result.sortie_count),
        )
    if result.scenario == "能耗优先":
        return common + (
            result.total_energy_kwh,
            result.weighted_tardiness,
            result.makespan_s,
            float(result.sortie_count),
        )
    return common + (
        float(result.sortie_count),
        result.weighted_tardiness,
        result.makespan_s,
        result.total_energy_kwh,
    )


def objective_scalar(result: ScheduleResult) -> float:
    hard_penalty = 1e12 * result.hard_violation_count + 1e7 * result.hard_lateness_s
    tardiness_scale = result.weighted_tardiness / 1000.0
    makespan_scale = result.makespan_s / 1000.0
    if result.scenario == "及时性优先":
        return hard_penalty + 1e5 * tardiness_scale + 1e3 * makespan_scale + 10.0 * result.total_energy_kwh + result.sortie_count
    if result.scenario == "完工时间优先":
        return hard_penalty + 1e5 * makespan_scale + 100.0 * tardiness_scale + 10.0 * result.total_energy_kwh + result.sortie_count
    if result.scenario == "能耗优先":
        return hard_penalty + 1e5 * result.total_energy_kwh + 100.0 * tardiness_scale + 10.0 * makespan_scale + result.sortie_count
    return hard_penalty + 1e7 * result.sortie_count + 100.0 * tardiness_scale + 10.0 * makespan_scale + result.total_energy_kwh


# ==============================
# 7. 多起点局部搜索
# ==============================


def feasible_partition_after_change(routes: List[List[str]]) -> Optional[Tuple[Tuple[str, ...], ...]]:
    partition = normalize_partition(routes)
    if all(feasible_route_options(route) for route in partition):
        return partition
    return None


def mutate_partition(
    partition: Tuple[Tuple[str, ...], ...], rng: random.Random
) -> Optional[Tuple[Tuple[str, ...], ...]]:
    if not partition:
        return None
    for _ in range(40):
        routes = [list(route) for route in partition]
        move = rng.choices(
            ["merge", "move", "swap", "split"],
            weights=[0.22, 0.42, 0.24, 0.12],
            k=1,
        )[0]

        if move == "merge" and len(routes) >= 2:
            first, second = rng.sample(range(len(routes)), 2)
            merged = routes[first] + routes[second]
            if feasible_route_options(tuple(sorted(merged))):
                new_routes = [
                    route
                    for index, route in enumerate(routes)
                    if index not in {first, second}
                ]
                new_routes.append(merged)
                return normalize_partition(new_routes)

        if move == "move" and len(routes) >= 1:
            source = rng.randrange(len(routes))
            box_id = rng.choice(routes[source])
            destination_choices = list(range(len(routes))) + [None]
            destination = rng.choice(destination_choices)
            if destination == source:
                continue
            new_routes = [route.copy() for route in routes]
            new_routes[source].remove(box_id)
            if destination is None:
                new_routes.append([box_id])
            else:
                new_routes[destination].append(box_id)
            new_routes = [route for route in new_routes if route]
            candidate = feasible_partition_after_change(new_routes)
            if candidate is not None:
                return candidate

        if move == "swap" and len(routes) >= 2:
            first, second = rng.sample(range(len(routes)), 2)
            box_first = rng.choice(routes[first])
            box_second = rng.choice(routes[second])
            new_routes = [route.copy() for route in routes]
            new_routes[first].remove(box_first)
            new_routes[second].remove(box_second)
            new_routes[first].append(box_second)
            new_routes[second].append(box_first)
            candidate = feasible_partition_after_change(new_routes)
            if candidate is not None:
                return candidate

        if move == "split":
            splittable = [index for index, route in enumerate(routes) if len(route) >= 2]
            if not splittable:
                continue
            source = rng.choice(splittable)
            shuffled = routes[source].copy()
            rng.shuffle(shuffled)
            cut = rng.randint(1, len(shuffled) - 1)
            first_part = shuffled[:cut]
            second_part = shuffled[cut:]
            new_routes = [
                route for index, route in enumerate(routes) if index != source
            ] + [first_part, second_part]
            candidate = feasible_partition_after_change(new_routes)
            if candidate is not None:
                return candidate
    return None


def optimize_scenario(
    scenario: str,
    initial_partitions: Sequence[Tuple[Tuple[str, ...], ...]],
    seed: int,
) -> ScheduleResult:
    rng = random.Random(seed)
    evaluated: Dict[Tuple[Tuple[str, ...], ...], ScheduleResult] = {}

    def evaluate(partition: Tuple[Tuple[str, ...], ...]) -> ScheduleResult:
        if partition not in evaluated:
            evaluated[partition] = schedule_partition(partition, scenario)
        return evaluated[partition]

    initial_results = [evaluate(partition) for partition in initial_partitions]
    best = min(initial_results, key=objective_key)
    steps_per_start = max(1, SEARCH_ITERATIONS // len(initial_partitions))

    for start_number, initial in enumerate(initial_partitions, start=1):
        current_partition = initial
        current = evaluate(current_partition)
        start_temperature = max(1.0, objective_scalar(current) * 0.002)

        for step in range(steps_per_start):
            candidate_partition = mutate_partition(current_partition, rng)
            if candidate_partition is None or candidate_partition == current_partition:
                continue
            candidate = evaluate(candidate_partition)
            delta = objective_scalar(candidate) - objective_scalar(current)
            temperature = max(
                1e-6,
                start_temperature * (1.0 - step / max(1, steps_per_start)) ** 2,
            )
            if delta <= 0.0 or rng.random() < math.exp(-min(delta / temperature, 700.0)):
                current_partition = candidate_partition
                current = candidate
            if objective_key(candidate) < objective_key(best):
                best = candidate

        print(
            f"  起点 {start_number}/{len(initial_partitions)} 完成："
            f"当前全局最好 {best.sortie_count} 架，"
            f"完工 {best.makespan_s:.2f} s，"
            f"能耗 {best.total_energy_kwh:.6f} kWh，"
            f"硬时限违约 {best.hard_violation_count} 箱。"
        )
    return best


# ==============================
# 8. 结果校验与输出
# ==============================


def intervals_overlap(intervals: List[Tuple[float, float]]) -> bool:
    ordered = sorted(intervals)
    return any(
        ordered[index][0] < ordered[index - 1][1] - TOLERANCE
        for index in range(1, len(ordered))
    )


def validate_result(result: ScheduleResult) -> pd.DataFrame:
    deliveries = pd.DataFrame(result.deliveries)
    sorties = pd.DataFrame(result.sorties)
    issues: List[str] = []

    counts = deliveries["货箱编号"].value_counts()
    if len(deliveries) != 80 or set(counts.index) != set(BOX_BY_ID) or not counts.eq(1).all():
        issues.append("80个货箱未恰好覆盖一次")
    if result.hard_violation_count != 0:
        issues.append(f"存在{result.hard_violation_count}个硬时限违约货箱")
    if (sorties["架次能耗_kWh"] > sorties["任务能量上限_kWh"] + TOLERANCE).any():
        issues.append("存在超出返航安全能量上限的架次")
    if (sorties["返航SOC_pct"] < 20.0 - TOLERANCE).any():
        issues.append("存在返航SOC低于20%的架次")

    for drone_id, group in sorties.groupby("无人机编号"):
        intervals = list(zip(group["开始时刻_s"], group["返回O01时刻_s"]))
        if intervals_overlap(intervals):
            issues.append(f"实体无人机{drone_id}任务时段重叠")
    for battery_id, group in sorties.groupby("电池编号"):
        intervals = list(zip(group["开始时刻_s"], group["充电完成时刻_s"]))
        if intervals_overlap(intervals):
            issues.append(f"共享电池{battery_id}任务或充电时段重叠")

    checks = [
        ("货箱精确覆盖", len(deliveries) == 80 and counts.eq(1).all()),
        ("医疗物资期望时间", result.medical_on_time_count == int(BOXES["is_medical"].sum())),
        ("首批保障截止时间", result.first_batch_on_time_count == int(BOXES["is_first_batch"].sum())),
        ("返航能量与SOC", not any("返航" in issue for issue in issues)),
        ("实体无人机时段无冲突", not any("实体无人机" in issue for issue in issues)),
        ("共享电池任务与充电无冲突", not any("共享电池" in issue for issue in issues)),
    ]
    check_frame = pd.DataFrame(
        {
            "检查项目": [item[0] for item in checks],
            "是否通过": ["是" if item[1] else "否" for item in checks],
        }
    )
    if issues:
        raise AssertionError("；".join(issues))
    return check_frame


def result_summary(result: ScheduleResult) -> dict:
    return {
        "优化情景": result.scenario,
        "目标优先级": result.priority,
        "硬时限违约箱数": result.hard_violation_count,
        "硬时限总迟到_s": result.hard_lateness_s,
        "加权迟到_优先系数乘秒": result.weighted_tardiness,
        "全部任务完成时间_s": result.makespan_s,
        "总运输能耗_kWh": result.total_energy_kwh,
        "运输架次数": result.sortie_count,
        "期望时间内送达箱数": result.desired_on_time_count,
        "首批按时箱数": result.first_batch_on_time_count,
        "医疗物资按时箱数": result.medical_on_time_count,
        "算法说明": "多起点组批+局部搜索+实体机/共享电池贪心时间轴调度",
    }


def write_results(results: Dict[str, ScheduleResult]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = pd.DataFrame([result_summary(result) for result in results.values()])
    summaries.to_csv(
        OUTPUT_DIR / "问题二第一问_多目标指标比较.csv",
        index=False,
        encoding="utf-8-sig",
    )

    all_sorties = []
    all_deliveries = []
    for scenario, result in results.items():
        sortie_frame = pd.DataFrame(result.sorties)
        sortie_frame.insert(0, "优化情景", scenario)
        delivery_frame = pd.DataFrame(result.deliveries)
        delivery_frame.insert(0, "优化情景", scenario)
        all_sorties.append(sortie_frame)
        all_deliveries.append(delivery_frame)

    pd.concat(all_sorties, ignore_index=True).to_csv(
        OUTPUT_DIR / "问题二第一问_多情景运输架次.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(all_deliveries, ignore_index=True).to_csv(
        OUTPUT_DIR / "问题二第一问_多情景逐箱交付.csv",
        index=False,
        encoding="utf-8-sig",
    )

    main = results["及时性优先"]
    main_sorties = pd.DataFrame(main.sorties).sort_values(["开始时刻_s", "架次编号"])
    main_deliveries = pd.DataFrame(main.deliveries).sort_values(
        ["交付完成时刻_s", "货箱编号"]
    )
    main_sorties.to_csv(
        OUTPUT_DIR / "问题二第一问_主方案运输架次.csv",
        index=False,
        encoding="utf-8-sig",
    )
    main_deliveries.to_csv(
        OUTPUT_DIR / "问题二第一问_主方案逐箱交付.csv",
        index=False,
        encoding="utf-8-sig",
    )

    drone_timeline = main_sorties[
        ["无人机编号", "架次编号", "开始时刻_s", "起飞时刻_s", "返回O01时刻_s"]
    ].sort_values(["无人机编号", "开始时刻_s"])
    battery_timeline = main_sorties[
        [
            "电池编号",
            "架次编号",
            "开始时刻_s",
            "返回O01时刻_s",
            "充电完成时刻_s",
            "返航SOC_pct",
        ]
    ].sort_values(["电池编号", "开始时刻_s"])
    drone_timeline.to_csv(
        OUTPUT_DIR / "问题二第一问_无人机资源时间轴.csv",
        index=False,
        encoding="utf-8-sig",
    )
    battery_timeline.to_csv(
        OUTPUT_DIR / "问题二第一问_电池资源时间轴.csv",
        index=False,
        encoding="utf-8-sig",
    )

    validate_result(main).to_csv(
        OUTPUT_DIR / "问题二第一问_主方案检查.csv",
        index=False,
        encoding="utf-8-sig",
    )
    segment_table().to_csv(
        OUTPUT_DIR / "问题二第一问_全节点航段参数.csv",
        index=False,
        encoding="utf-8-sig",
    )


# ==============================
# 9. 主程序
# ==============================


def main() -> None:
    global BOXES, BOX_BY_ID, DRONES, SEGMENTS
    global DRONE_RESOURCES, BATTERY_RESOURCES, FULL_CHARGE_TIME

    print("读取货箱、机型、实体无人机和共享电池数据……")
    BOXES = read_boxes()
    BOX_BY_ID = {
        str(row["box_id"]): row.to_dict() for _, row in BOXES.iterrows()
    }
    DRONES = {drone.code: drone for drone in Q1.read_drone_types()}
    (
        DRONE_RESOURCES,
        BATTERY_RESOURCES,
        FULL_CHARGE_TIME,
    ) = read_transport_resources()

    center, _, nodes = read_nodes()
    print("按题设水平直线和DEM净空规则预计算全部节点间航段……")
    SEGMENTS = build_all_segments(center, nodes)

    print("构造多组初始货箱组批……")
    initial_partitions = [
        q1_initial_partition(),
        greedy_partition("deadline"),
        greedy_partition("mass"),
        greedy_partition("service"),
    ]
    initial_partitions = list(dict.fromkeys(initial_partitions))
    print(
        "初始方案架次数："
        + "，".join(str(len(partition)) for partition in initial_partitions)
    )

    results: Dict[str, ScheduleResult] = {}
    for scenario_index, scenario in enumerate(SCENARIOS, start=1):
        print(f"\n[{scenario_index}/{len(SCENARIOS)}] 求解{scenario}情景……")
        result = optimize_scenario(
            scenario,
            initial_partitions,
            RANDOM_SEED + 1000 * scenario_index,
        )
        validate_result(result)
        results[scenario] = result
        if result.partition not in initial_partitions:
            initial_partitions.append(result.partition)

    shared_partitions = list(
        dict.fromkeys(initial_partitions + [result.partition for result in results.values()])
    )
    print("\n使用全部已发现分组构成共享候选池，再进行一轮交叉复优化……")
    for scenario_index, scenario in enumerate(SCENARIOS, start=1):
        refined = optimize_scenario(
            scenario,
            shared_partitions,
            RANDOM_SEED + 10_000 + 1000 * scenario_index,
        )
        comparison_candidates = [refined, results[scenario]]
        comparison_candidates.extend(
            schedule_partition(partition, scenario)
            for partition in shared_partitions
        )
        best = min(comparison_candidates, key=objective_key)
        validate_result(best)
        results[scenario] = best
        if best.partition not in shared_partitions:
            shared_partitions.append(best.partition)

    write_results(results)
    print("\n问题二第一小问启发式优化完成。")
    print(pd.DataFrame([result_summary(result) for result in results.values()]).to_string(index=False))
    print(f"\n输出目录：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
