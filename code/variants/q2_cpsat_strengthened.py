"""问题二第一小问：候选池扩展 + CP-SAT 字典序强化求解。

该程序不是简单再次运行元启发式，而是采用数学规划型混合算法：

1. 读取原问题二脚本以及五算法已经发现的货箱分组；
2. 用随机大邻域搜索扩展候选组批；
3. 对当前较优解中的两架次进行局部穷举重组；
4. 在生成的有限候选架次集合上建立 CP-SAT 精确模型；
5. 依次精确最小化加权迟到、全部任务完成时间、总能耗和架次数；
6. 将精确解继续用于扩充候选池，重复若干轮。

CP-SAT 返回 OPTIMAL 时，表示在“当前候选架次集合”内已经证明最优。只有枚举了
原问题的全部可行架次，才可称为原问题全局最优；80箱全部子集无法直接暴力枚举，
因此本程序通过多轮候选池扩展逐步逼近该目标，并保留求解界和 gap 供说明。
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, permutations
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

try:
    import torch
except ImportError:
    torch = None


ROOT = Path(__file__).resolve().parent
BASE_SCRIPT = ROOT / "问题二_第一问.py"
BASE_RESULT_DIR = ROOT / "问题二第一问结果"
FIVE_RESULT_DIR = ROOT / "问题二第一问五算法对比结果"
OUTPUT_DIR = ROOT / "问题二第一问CP-SAT强化结果"

RANDOM_SEED = 20260923
MASTER_ROUNDS = 3
RANDOM_WALK_STEPS_PER_ROUND = 12_000
BRUTE_FORCE_MAX_UNION_BOXES = 12
BRUTE_FORCE_PAIR_LIMIT = 120
MAX_ROUTE_GROUPS = 1_500
CP_SAT_TIME_LIMIT_PER_STAGE_S = 900
CP_SAT_WORKERS = 8
HORIZON_S = 40_000
ENERGY_SCALE = 1_000_000
TOLERANCE = 1e-8
EXHAUSTIVE_GROUP_MAX_SIZE = 2
ENABLE_ROUTE_ORDER_VARIANTS = True
MAX_ROUTE_ORDERS_PER_DRONE = 4
ROUTE_ORDER_START_SAMPLES_S = (0, 1_800, 3_600, 7_200)
SOLVER_LOG = False
DEVICE_REQUEST = "auto"
COMPUTE_DEVICE = "cpu"
GPU_NAME = ""
GPU_PREFILTER_ENABLED = False
OBJECTIVE_MODE = "timeliness"
ZERO_TARDINESS_REQUIRED = False
MAX_SORTIES_LIMIT: Optional[int] = None
MAX_MAKESPAN_LIMIT_S: Optional[int] = None
MAX_STOPS_LIMIT = 5


Partition = Tuple[Tuple[str, ...], ...]


def configure_compute_device(requested: str) -> None:
    global DEVICE_REQUEST
    global COMPUTE_DEVICE
    global GPU_NAME
    global GPU_PREFILTER_ENABLED

    DEVICE_REQUEST = requested
    cuda_available = bool(torch is not None and torch.cuda.is_available())
    if requested == "gpu" and not cuda_available:
        raise RuntimeError(
            "指定了--device gpu，但当前Python环境未检测到可用CUDA。"
        )
    if requested in {"gpu", "auto"} and cuda_available:
        COMPUTE_DEVICE = "cuda"
        GPU_PREFILTER_ENABLED = True
        GPU_NAME = str(torch.cuda.get_device_name(0))
        print(f"GPU候选组批预筛选：启用，设备={GPU_NAME}")
    else:
        COMPUTE_DEVICE = "cpu"
        GPU_PREFILTER_ENABLED = False
        GPU_NAME = ""
        print("GPU候选组批预筛选：未启用，候选评分使用CPU。")
    print("CP-SAT求解器：使用CPU多线程（OR-Tools CP-SAT不支持CUDA求解）。")


def objective_priority_text() -> str:
    if ZERO_TARDINESS_REQUIRED:
        prefix = "加权迟到=0硬约束"
    else:
        prefix = "硬时限"
    names = {
        "timeliness": ["加权迟到", "全部任务完成时间", "总运输能耗", "运输架次数"],
        "makespan": ["全部任务完成时间", "加权迟到", "总运输能耗", "运输架次数"],
        "energy": ["总运输能耗", "全部任务完成时间", "运输架次数", "加权迟到"],
        "sortie": ["运输架次数", "全部任务完成时间", "总运输能耗", "加权迟到"],
    }
    return prefix + " → " + " → ".join(names[OBJECTIVE_MODE])


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_module("q2_first_base_for_cpsat", BASE_SCRIPT)


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    box_ids: Tuple[str, ...]
    drone_type: str
    service_order: Tuple[str, ...]
    option: object
    duration_s_int: int
    occupied_battery_s_int: int
    energy_int: int
    delivery_offset_int: Tuple[Tuple[str, int], ...]


@dataclass
class MasterSolution:
    result: object
    selected_candidates: List[Candidate]
    start_times: Dict[str, int]
    drone_assignment: Dict[str, str]
    battery_assignment: Dict[str, str]
    objective_tuple: Tuple[int, int, int, int]
    stage_log: List[dict]


def initialize_problem() -> None:
    BASE.BOXES = BASE.read_boxes()
    BASE.BOX_BY_ID = {
        str(row["box_id"]): row.to_dict()
        for _, row in BASE.BOXES.iterrows()
    }
    BASE.DRONES = {drone.code: drone for drone in BASE.Q1.read_drone_types()}
    (
        BASE.DRONE_RESOURCES,
        BASE.BATTERY_RESOURCES,
        BASE.FULL_CHARGE_TIME,
    ) = BASE.read_transport_resources()
    center, _, nodes = BASE.read_nodes()
    BASE.SEGMENTS = BASE.build_all_segments(center, nodes)
    BASE.feasible_route_options.cache_clear()
    candidate_route_options.cache_clear()


def route_option_key_at_start(option, box_ids: Sequence[str], start_time: float) -> Tuple[float, ...]:
    delivery_offsets = dict(option.delivery_offsets)
    hard_count = 0
    hard_lateness = 0.0
    weighted_tardiness = 0.0
    for box_id in box_ids:
        record = BASE.BOX_BY_ID[box_id]
        delivery_time = start_time + delivery_offsets[box_id]
        deadline = BASE.hard_deadline(record)
        if deadline is not None and delivery_time > deadline + TOLERANCE:
            hard_count += 1
            hard_lateness += delivery_time - deadline
        weighted_tardiness += float(record["priority"]) * max(
            0.0, delivery_time - float(record["desired_time_s"])
        )
    return (
        float(hard_count),
        hard_lateness,
        weighted_tardiness,
        start_time + option.duration_s,
        option.energy_kwh,
    )


@lru_cache(maxsize=100_000)
def candidate_route_options(box_ids_key: Tuple[str, ...]) -> Tuple[object, ...]:
    box_ids = tuple(sorted(box_ids_key))
    base_options = BASE.feasible_route_options(box_ids)
    if not ENABLE_ROUTE_ORDER_VARIANTS or not base_options:
        return base_options

    total_mass, total_volume, services = BASE.route_basic_totals(box_ids)
    boxes_by_service: Dict[str, List[dict]] = defaultdict(list)
    for record in BASE.route_box_records(box_ids):
        boxes_by_service[record["service_id"]].append(record)

    selected_options: List[object] = []
    for drone_type, drone in sorted(BASE.DRONES.items()):
        if total_mass > drone.max_payload_kg + TOLERANCE:
            continue
        if total_volume > drone.volume_m3 + TOLERANCE:
            continue

        all_options: List[object] = []
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
                segment = BASE.SEGMENTS[(current_node, service_id)]
                current_time += BASE.segment_flight_time(drone, segment)
                total_energy += BASE.Q1.segment_energy(
                    drone, segment.distance_m, segment.climb_m, remaining_mass
                )
                service_boxes = boxes_by_service[service_id]
                current_time += (
                    drone.handover_base_time_s
                    + len(service_boxes) * drone.handover_per_box_time_s
                )
                for record in service_boxes:
                    delivery_offsets[record["box_id"]] = current_time
                remaining_mass -= sum(
                    float(record["mass_kg"]) for record in service_boxes
                )
                current_node = service_id

            return_segment = BASE.SEGMENTS[(current_node, "O01")]
            current_time += BASE.segment_flight_time(drone, return_segment)
            total_energy += BASE.Q1.segment_energy(
                drone, return_segment.distance_m, return_segment.climb_m, 0.0
            )
            energy_limit = BASE.Q1.task_energy_limit(drone)
            if total_energy > energy_limit + TOLERANCE:
                continue
            return_soc = BASE.Q1.return_soc_ratio(drone, total_energy)
            all_options.append(
                BASE.RouteOption(
                    drone_type=drone_type,
                    service_order=tuple(service_order),
                    total_mass_kg=total_mass,
                    total_volume_m3=total_volume,
                    energy_kwh=total_energy,
                    energy_limit_kwh=energy_limit,
                    return_soc_pct=100.0 * return_soc,
                    duration_s=current_time,
                    takeoff_offset_s=takeoff_offset,
                    charge_time_s=BASE.battery_charge_time(drone_type, return_soc),
                    delivery_offsets=tuple(sorted(delivery_offsets.items())),
                )
            )

        if not all_options:
            continue

        win_count: Dict[Tuple[str, ...], int] = defaultdict(int)
        base_option = next(
            option for option in base_options if option.drone_type == drone_type
        )
        win_count[base_option.service_order] += 10
        win_count[min(all_options, key=lambda option: option.duration_s).service_order] += 2
        win_count[min(all_options, key=lambda option: option.energy_kwh).service_order] += 2
        for sample_start in ROUTE_ORDER_START_SAMPLES_S:
            winner = min(
                all_options,
                key=lambda option: route_option_key_at_start(
                    option, box_ids, sample_start
                ),
            )
            win_count[winner.service_order] += 3

        option_by_order = {option.service_order: option for option in all_options}
        ranked_orders = sorted(
            win_count,
            key=lambda order: (
                -win_count[order],
                option_by_order[order].duration_s,
                option_by_order[order].energy_kwh,
                order,
            ),
        )
        selected_options.extend(
            option_by_order[order]
            for order in ranked_orders[:MAX_ROUTE_ORDERS_PER_DRONE]
        )

    return tuple(selected_options)


def partition_from_frame(frame: pd.DataFrame) -> List[Partition]:
    if "货箱编号列表" not in frame.columns:
        return []
    grouping_columns = [column for column in ["算法", "优化情景"] if column in frame.columns]
    if not grouping_columns:
        grouped = [("single", frame)]
    else:
        grouped = list(frame.groupby(grouping_columns, dropna=False))

    partitions: List[Partition] = []
    for _, group in grouped:
        partition = BASE.normalize_partition(
            str(value).split("、") for value in group["货箱编号列表"]
        )
        if BASE.partition_is_physically_feasible(partition):
            partitions.append(partition)
    return partitions


def load_existing_partitions() -> List[Partition]:
    partitions = [
        BASE.q1_initial_partition(),
        BASE.greedy_partition("deadline"),
        BASE.greedy_partition("mass"),
        BASE.greedy_partition("service"),
    ]
    files = [
        BASE_RESULT_DIR / "问题二第一问_主方案运输架次.csv",
        BASE_RESULT_DIR / "问题二第一问_多情景运输架次.csv",
        FIVE_RESULT_DIR / "问题二第一问_五算法运输架次.csv",
        FIVE_RESULT_DIR / "问题二第一问_五算法最佳方案运输架次.csv",
        ROOT / "问题二第一问CP-SAT强化结果" / "问题二第一问_CP-SAT强化方案运输架次.csv",
    ]
    gpu_result_root = ROOT / "问题二第一问_GPU多方案实验结果"
    if gpu_result_root.exists():
        files.extend(gpu_result_root.glob("**/*运输架次.csv"))
    files.extend(ROOT.glob("问题二*/*运输架次.csv"))
    for path in files:
        if path.exists():
            partitions.extend(
                partition_from_frame(pd.read_csv(path, encoding="utf-8-sig"))
            )
    return list(dict.fromkeys(partitions))


class RouteGroupPool:
    def __init__(self):
        self.groups: set[Tuple[str, ...]] = set()
        self.mandatory: set[Tuple[str, ...]] = set()

    def add_group(self, box_ids: Iterable[str], mandatory: bool = False) -> bool:
        key = tuple(sorted(box_ids))
        if not key or not BASE.feasible_route_options(key):
            return False
        is_new = key not in self.groups
        self.groups.add(key)
        if mandatory:
            self.mandatory.add(key)
        return is_new

    def add_partition(self, partition: Partition, mandatory: bool = False) -> int:
        return sum(self.add_group(route, mandatory) for route in partition)

    @staticmethod
    def _group_features(group: Tuple[str, ...]) -> Tuple[float, ...]:
        options = BASE.feasible_route_options(group)
        if not options:
            return (0.0, 0.0, 1e9, 1e9, 1e9)
        best_energy = min(option.energy_kwh for option in options)
        best_duration_per_box = min(option.duration_s for option in options) / len(group)
        _, _, services = BASE.route_basic_totals(group)
        hard_count = sum(
            BASE.hard_deadline(BASE.BOX_BY_ID[box_id]) is not None
            for box_id in group
        )
        return (
            float(len(group)),
            float(hard_count),
            float(len(services)),
            float(best_duration_per_box),
            best_energy,
        )

    @staticmethod
    def _gpu_rank_groups(
        groups: Sequence[Tuple[str, ...]],
    ) -> List[Tuple[str, ...]]:
        features = np.asarray(
            [RouteGroupPool._group_features(group) for group in groups],
            dtype=np.float32,
        )
        if not GPU_PREFILTER_ENABLED or torch is None or len(groups) == 0:
            order = sorted(
                range(len(groups)),
                key=lambda index: (
                    -features[index, 0],
                    -features[index, 1],
                    features[index, 2],
                    features[index, 3],
                    features[index, 4],
                ),
            )
            return [groups[index] for index in order]

        tensor = torch.as_tensor(features, device="cuda")
        duration_scale = torch.clamp(tensor[:, 3].amax(), min=1.0)
        energy_scale = torch.clamp(tensor[:, 4].amax(), min=1.0)
        score = (
            -tensor[:, 0] * 1_000_000.0
            - tensor[:, 1] * 10_000.0
            + tensor[:, 2] * 100.0
            + tensor[:, 3] / duration_scale * 10.0
            + tensor[:, 4] / energy_scale
        )
        order = torch.argsort(score).detach().cpu().tolist()
        return [groups[index] for index in order]

    def prune(self) -> None:
        if len(self.groups) <= MAX_ROUTE_GROUPS:
            return
        optional = [group for group in self.groups if group not in self.mandatory]

        keep_optional = max(0, MAX_ROUTE_GROUPS - len(self.mandatory))
        ranked_optional = self._gpu_rank_groups(optional)
        selected = ranked_optional[:keep_optional]
        self.groups = set(self.mandatory) | set(selected)

    def candidates(self) -> List[Candidate]:
        candidates: List[Candidate] = []
        for group_index, group in enumerate(sorted(self.groups), start=1):
            for option_index, option in enumerate(
                candidate_route_options(group), start=1
            ):
                duration_int = int(math.ceil(option.duration_s - 1e-9))
                occupied_battery_int = int(
                    math.ceil(option.duration_s + option.charge_time_s - 1e-9)
                )
                candidates.append(
                    Candidate(
                        candidate_id=f"C{group_index:05d}-{option_index}",
                        box_ids=group,
                        drone_type=option.drone_type,
                        service_order=option.service_order,
                        option=option,
                        duration_s_int=duration_int,
                        occupied_battery_s_int=occupied_battery_int,
                        energy_int=int(round(option.energy_kwh * ENERGY_SCALE)),
                        delivery_offset_int=tuple(
                            (box_id, int(math.ceil(offset - 1e-9)))
                            for box_id, offset in option.delivery_offsets
                        ),
                    )
                )
        return candidates


def exhaustive_small_group_expand(
    pool: RouteGroupPool, max_group_size: int
) -> Tuple[int, int]:
    if max_group_size <= 1:
        return 0, 0
    box_ids = tuple(sorted(BASE.BOX_BY_ID))
    tested = 0
    added = 0
    for group_size in range(2, max_group_size + 1):
        for group in combinations(box_ids, group_size):
            tested += 1
            added += pool.add_group(group)
    return tested, added


def random_walk_expand(
    pool: RouteGroupPool,
    starts: Sequence[Partition],
    steps: int,
    rng: random.Random,
) -> int:
    current = rng.choice(list(starts))
    added = 0
    for step in range(steps):
        candidate = BASE.mutate_partition(current, rng)
        if candidate is not None:
            current = candidate
            added += pool.add_partition(candidate)
        if step % 200 == 199:
            current = rng.choice(list(starts))
    return added


def brute_force_two_route_repacking(
    pool: RouteGroupPool,
    partition: Partition,
    rng: random.Random,
) -> int:
    route_pairs = [
        (first, second)
        for first, second in combinations(partition, 2)
        if 2 <= len(set(first) | set(second)) <= BRUTE_FORCE_MAX_UNION_BOXES
    ]
    rng.shuffle(route_pairs)
    route_pairs = route_pairs[:BRUTE_FORCE_PAIR_LIMIT]
    added = 0

    for first, second in route_pairs:
        union = tuple(sorted(set(first) | set(second)))
        anchor = union[0]
        remaining = union[1:]
        for mask in range(1 << len(remaining)):
            group_a = [anchor]
            group_b = []
            for index, box_id in enumerate(remaining):
                if mask & (1 << index):
                    group_a.append(box_id)
                else:
                    group_b.append(box_id)
            if not group_b:
                continue
            if BASE.feasible_route_options(tuple(sorted(group_a))) and BASE.feasible_route_options(tuple(sorted(group_b))):
                added += pool.add_group(group_a)
                added += pool.add_group(group_b)
    return added


def build_master_model(candidates: Sequence[Candidate]):
    model = cp_model.CpModel()
    selected: Dict[int, object] = {}
    start: Dict[int, object] = {}
    task_end: Dict[int, object] = {}
    battery_end: Dict[int, object] = {}
    drone_intervals: Dict[str, List[object]] = {
        drone_type: [] for drone_type in BASE.DRONE_RESOURCES
    }
    battery_intervals: Dict[str, List[object]] = {
        drone_type: [] for drone_type in BASE.BATTERY_RESOURCES
    }

    for index, candidate in enumerate(candidates):
        selected[index] = model.NewBoolVar(f"select_{index}")
        start[index] = model.NewIntVar(0, HORIZON_S, f"start_{index}")
        task_end[index] = model.NewIntVar(
            0, HORIZON_S + candidate.duration_s_int, f"end_{index}"
        )
        model.Add(task_end[index] == start[index] + candidate.duration_s_int)
        model.Add(start[index] == 0).OnlyEnforceIf(selected[index].Not())

        drone_intervals[candidate.drone_type].append(
            model.NewOptionalIntervalVar(
                start[index],
                candidate.duration_s_int,
                task_end[index],
                selected[index],
                f"task_{index}",
            )
        )

        battery_end[index] = model.NewIntVar(
            0,
            HORIZON_S + candidate.occupied_battery_s_int,
            f"battery_end_{index}",
        )
        model.Add(
            battery_end[index]
            == start[index] + candidate.occupied_battery_s_int
        )
        battery_intervals[candidate.drone_type].append(
            model.NewOptionalIntervalVar(
                start[index],
                candidate.occupied_battery_s_int,
                battery_end[index],
                selected[index],
                f"battery_task_charge_{index}",
            )
        )

    for drone_type, intervals in drone_intervals.items():
        model.AddCumulative(
            intervals,
            [1] * len(intervals),
            len(BASE.DRONE_RESOURCES[drone_type]),
        )
    for drone_type, intervals in battery_intervals.items():
        model.AddCumulative(
            intervals,
            [1] * len(intervals),
            len(BASE.BATTERY_RESOURCES[drone_type]),
        )

    candidates_by_box: Dict[str, List[int]] = {
        box_id: [] for box_id in BASE.BOX_BY_ID
    }
    offset_by_candidate_box: Dict[Tuple[int, str], int] = {}
    for index, candidate in enumerate(candidates):
        offset_map = dict(candidate.delivery_offset_int)
        for box_id in candidate.box_ids:
            candidates_by_box[box_id].append(index)
            offset_by_candidate_box[(index, box_id)] = offset_map[box_id]

    delivery: Dict[str, object] = {}
    tardiness: Dict[str, object] = {}
    for box_id, candidate_indices in candidates_by_box.items():
        if not candidate_indices:
            raise ValueError(f"候选池没有覆盖货箱：{box_id}")
        model.Add(sum(selected[index] for index in candidate_indices) == 1)
        delivery[box_id] = model.NewIntVar(0, HORIZON_S, f"delivery_{box_id}")
        tardiness[box_id] = model.NewIntVar(0, HORIZON_S, f"tardy_{box_id}")
        record = BASE.BOX_BY_ID[box_id]
        desired = int(round(float(record["desired_time_s"])))
        model.Add(tardiness[box_id] >= delivery[box_id] - desired)
        if ZERO_TARDINESS_REQUIRED:
            model.Add(tardiness[box_id] == 0)

        deadline = BASE.hard_deadline(record)
        for index in candidate_indices:
            offset = offset_by_candidate_box[(index, box_id)]
            model.Add(
                delivery[box_id] == start[index] + offset
            ).OnlyEnforceIf(selected[index])
            if deadline is not None:
                model.Add(
                    start[index] + offset <= int(math.floor(deadline + 1e-9))
                ).OnlyEnforceIf(selected[index])

    makespan = model.NewIntVar(0, HORIZON_S, "makespan")
    for index in range(len(candidates)):
        model.Add(makespan >= task_end[index]).OnlyEnforceIf(selected[index])

    weighted_tardiness = sum(
        int(round(float(BASE.BOX_BY_ID[box_id]["priority"])))
        * tardiness[box_id]
        for box_id in BASE.BOX_BY_ID
    )
    total_energy = sum(
        candidates[index].energy_int * selected[index]
        for index in range(len(candidates))
    )
    sortie_count = sum(selected.values())
    if MAX_SORTIES_LIMIT is not None:
        model.Add(sortie_count <= MAX_SORTIES_LIMIT)
    if MAX_MAKESPAN_LIMIT_S is not None:
        model.Add(makespan <= MAX_MAKESPAN_LIMIT_S)

    variables = {
        "selected": selected,
        "start": start,
        "task_end": task_end,
        "battery_end": battery_end,
        "delivery": delivery,
        "tardiness": tardiness,
        "makespan": makespan,
    }
    objective_map = {
        "timeliness": [
            ("加权迟到", weighted_tardiness),
            ("全部任务完成时间", makespan),
            ("总运输能耗", total_energy),
            ("运输架次数", sortie_count),
        ],
        "makespan": [
            ("全部任务完成时间", makespan),
            ("加权迟到", weighted_tardiness),
            ("总运输能耗", total_energy),
            ("运输架次数", sortie_count),
        ],
        "energy": [
            ("总运输能耗", total_energy),
            ("全部任务完成时间", makespan),
            ("运输架次数", sortie_count),
            ("加权迟到", weighted_tardiness),
        ],
        "sortie": [
            ("运输架次数", sortie_count),
            ("全部任务完成时间", makespan),
            ("总运输能耗", total_energy),
            ("加权迟到", weighted_tardiness),
        ],
    }
    objectives = objective_map[OBJECTIVE_MODE]
    return model, variables, objectives


def add_result_hint(
    model,
    variables: dict,
    candidates: Sequence[Candidate],
    hint_result: Optional[object],
) -> None:
    if hint_result is None:
        return
    if (
        MAX_SORTIES_LIMIT is not None
        and hint_result.sortie_count > MAX_SORTIES_LIMIT
    ):
        return
    if (
        MAX_MAKESPAN_LIMIT_S is not None
        and hint_result.makespan_s > MAX_MAKESPAN_LIMIT_S + TOLERANCE
    ):
        return
    exact_index: Dict[Tuple[Tuple[str, ...], str, Tuple[str, ...]], int] = {}
    fallback_index: Dict[Tuple[Tuple[str, ...], str], int] = {}
    for index, candidate in enumerate(candidates):
        exact_index[
            (candidate.box_ids, candidate.drone_type, candidate.service_order)
        ] = index
        fallback_index.setdefault((candidate.box_ids, candidate.drone_type), index)

    matched: List[Tuple[dict, int]] = []
    used: set[int] = set()
    for sortie in sorted(
        hint_result.sorties, key=lambda row: float(row["开始时刻_s"])
    ):
        box_ids = tuple(sorted(str(sortie["货箱编号列表"]).split("、")))
        drone_type = str(sortie["机型编号"])
        service_order = tuple(str(sortie["访问服务区顺序"]).split("→"))
        index = exact_index.get((box_ids, drone_type, service_order))
        if index is None:
            index = fallback_index.get((box_ids, drone_type))
        if index is None or index in used:
            return
        used.add(index)
        matched.append((sortie, index))

    covered = {
        box_id
        for _, index in matched
        for box_id in candidates[index].box_ids
    }
    if covered != set(BASE.BOX_BY_ID):
        return

    drone_ready = {
        drone_id: 0
        for drone_ids in BASE.DRONE_RESOURCES.values()
        for drone_id in drone_ids
    }
    battery_ready = {
        battery_id: 0
        for battery_ids in BASE.BATTERY_RESOURCES.values()
        for battery_id in battery_ids
    }
    selected_start: Dict[int, int] = {}
    hinted_delivery: Dict[str, int] = {}

    for sortie, index in matched:
        candidate = candidates[index]
        drone_id = str(sortie["无人机编号"])
        battery_id = str(sortie["电池编号"])
        if drone_id not in BASE.DRONE_RESOURCES[candidate.drone_type]:
            return
        if battery_id not in BASE.BATTERY_RESOURCES[candidate.drone_type]:
            return
        original_start = int(math.ceil(float(sortie["开始时刻_s"]) - 1e-9))
        start_value = max(
            0,
            original_start,
            drone_ready[drone_id],
            battery_ready[battery_id],
        )
        if start_value > HORIZON_S:
            return
        selected_start[index] = start_value
        drone_ready[drone_id] = start_value + candidate.duration_s_int
        battery_ready[battery_id] = (
            start_value + candidate.occupied_battery_s_int
        )
        for box_id, offset in candidate.delivery_offset_int:
            delivery_time = start_value + offset
            deadline = BASE.hard_deadline(BASE.BOX_BY_ID[box_id])
            if deadline is not None and delivery_time > math.floor(deadline + 1e-9):
                return
            hinted_delivery[box_id] = delivery_time

    model.ClearHints()
    for index in range(len(candidates)):
        is_selected = index in selected_start
        start_value = selected_start.get(index, 0)
        model.AddHint(variables["selected"][index], int(is_selected))
        model.AddHint(variables["start"][index], start_value)
        model.AddHint(
            variables["task_end"][index],
            start_value + candidates[index].duration_s_int,
        )
        model.AddHint(
            variables["battery_end"][index],
            start_value + candidates[index].occupied_battery_s_int,
        )

    for box_id, variable in variables["delivery"].items():
        delivery_time = hinted_delivery[box_id]
        desired = int(round(float(BASE.BOX_BY_ID[box_id]["desired_time_s"])))
        model.AddHint(variable, delivery_time)
        model.AddHint(
            variables["tardiness"][box_id],
            max(0, delivery_time - desired),
        )
    model.AddHint(
        variables["makespan"],
        max(
            selected_start[index] + candidates[index].duration_s_int
            for index in selected_start
        ),
    )


def replace_hints_with_solution(model, variables: dict, solver) -> None:
    model.ClearHints()
    for variable_name in [
        "selected",
        "start",
        "task_end",
        "battery_end",
        "delivery",
        "tardiness",
    ]:
        for variable in variables[variable_name].values():
            model.AddHint(variable, solver.Value(variable))
    model.AddHint(variables["makespan"], solver.Value(variables["makespan"]))


def solve_lexicographic_master(
    candidates: Sequence[Candidate],
    round_number: int,
    hint_result: Optional[object] = None,
) -> Tuple[object, dict, List[dict], Tuple[int, int, int, int]]:
    model, variables, objectives = build_master_model(candidates)
    add_result_hint(model, variables, candidates, hint_result)
    stage_log: List[dict] = []
    final_solver = None

    for stage_number, (objective_name, objective) in enumerate(objectives, start=1):
        model.Minimize(objective)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = CP_SAT_TIME_LIMIT_PER_STAGE_S
        solver.parameters.num_search_workers = CP_SAT_WORKERS
        solver.parameters.random_seed = RANDOM_SEED + 100 * round_number + stage_number
        solver.parameters.log_search_progress = SOLVER_LOG
        started = perf_counter()
        status = solver.Solve(model)
        elapsed = perf_counter() - started
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError(
                f"第{round_number}轮{objective_name}阶段没有找到可行解，"
                f"状态={solver.StatusName(status)}。"
            )

        value = int(round(solver.ObjectiveValue()))
        bound = float(solver.BestObjectiveBound())
        gap = abs(value - bound) / max(1.0, abs(value))
        proven_optimal = status == cp_model.OPTIMAL or abs(value - bound) < 0.5
        stage_log.append(
            {
                "候选池轮次": round_number,
                "字典序阶段": stage_number,
                "目标": objective_name,
                "状态": solver.StatusName(status),
                "目标值": value,
                "最好界": bound,
                "相对gap": gap,
                "已证明最优": "是" if proven_optimal else "否",
                "求解时间_s": elapsed,
                "候选架次数": len(candidates),
            }
        )
        final_solver = solver
        if not proven_optimal:
            break
        model.Add(objective == value)
        replace_hints_with_solution(model, variables, solver)

    if final_solver is None:
        raise AssertionError("CP-SAT未执行任何字典序阶段。")
    objective_values = tuple(
        int(final_solver.Value(objective)) for _, objective in objectives
    )
    return final_solver, variables, stage_log, objective_values


def extract_master_solution(
    candidates: Sequence[Candidate],
    solver,
    variables: dict,
    stage_log: List[dict],
    objective_tuple: Tuple[int, int, int, int],
) -> MasterSolution:
    selected_indices = [
        index
        for index, variable in variables["selected"].items()
        if solver.BooleanValue(variable)
    ]
    selected_indices.sort(key=lambda index: solver.Value(variables["start"][index]))

    sorties: List[dict] = []
    deliveries: List[dict] = []
    selected_candidates: List[Candidate] = []
    start_times: Dict[str, int] = {}
    assigned_drone: Dict[str, str] = {}
    assigned_battery: Dict[str, str] = {}
    drone_ready = {
        drone_id: 0
        for drone_ids in BASE.DRONE_RESOURCES.values()
        for drone_id in drone_ids
    }
    battery_ready = {
        battery_id: 0
        for battery_ids in BASE.BATTERY_RESOURCES.values()
        for battery_id in battery_ids
    }

    for sortie_number, index in enumerate(selected_indices, start=1):
        candidate = candidates[index]
        option = candidate.option
        selected_candidates.append(candidate)
        start_time = int(solver.Value(variables["start"][index]))
        available_drones = [
            drone_id
            for drone_id in BASE.DRONE_RESOURCES[candidate.drone_type]
            if drone_ready[drone_id] <= start_time
        ]
        if not available_drones:
            raise AssertionError("累计无人机容量解无法还原为具体实体无人机。")
        drone_id = min(
            available_drones,
            key=lambda resource_id: (drone_ready[resource_id], resource_id),
        )
        available_batteries = [
            battery_id
            for battery_id in BASE.BATTERY_RESOURCES[candidate.drone_type]
            if battery_ready[battery_id] <= start_time
        ]
        if not available_batteries:
            raise AssertionError("累计电池容量解无法还原为具体共享电池。")
        battery_id = min(
            available_batteries,
            key=lambda resource_id: (battery_ready[resource_id], resource_id),
        )
        drone_ready[drone_id] = start_time + candidate.duration_s_int
        battery_ready[battery_id] = (
            start_time + candidate.occupied_battery_s_int
        )
        sortie_id = f"Q2-CP-{sortie_number:03d}"
        exact_return_time = start_time + option.duration_s
        exact_charge_complete = exact_return_time + option.charge_time_s
        start_times[candidate.candidate_id] = start_time
        assigned_drone[candidate.candidate_id] = drone_id
        assigned_battery[candidate.candidate_id] = battery_id

        sorties.append(
            {
                "架次编号": sortie_id,
                "候选编号": candidate.candidate_id,
                "无人机编号": drone_id,
                "机型编号": candidate.drone_type,
                "电池编号": battery_id,
                "开始时刻_s": start_time,
                "起飞时刻_s": start_time + option.takeoff_offset_s,
                "访问服务区顺序": "→".join(candidate.service_order),
                "完整路线": "O01→" + "→".join(candidate.service_order) + "→O01",
                "货箱编号列表": "、".join(candidate.box_ids),
                "货箱数量": len(candidate.box_ids),
                "总质量_kg": option.total_mass_kg,
                "总体积_m3": option.total_volume_m3,
                "返回O01时刻_s": exact_return_time,
                "架次持续时间_s": option.duration_s,
                "架次能耗_kWh": option.energy_kwh,
                "任务能量上限_kWh": option.energy_limit_kwh,
                "返航SOC_pct": option.return_soc_pct,
                "充电完成时刻_s": exact_charge_complete,
            }
        )

        delivery_offsets = dict(option.delivery_offsets)
        for box_id in candidate.box_ids:
            record = BASE.BOX_BY_ID[box_id]
            delivery_time = start_time + delivery_offsets[box_id]
            deadline = BASE.hard_deadline(record)
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
                    "交付完成时刻_s": delivery_time,
                    "硬时限是否满足": "是" if deadline is None or delivery_time <= deadline + TOLERANCE else "否",
                    "期望时间迟到_s": max(
                        0.0, delivery_time - float(record["desired_time_s"])
                    ),
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
    first_rows = delivery_frame[delivery_frame["是否首批保障"].eq("是")]
    medical_rows = delivery_frame[delivery_frame["物资类型"].eq("医疗物资")]
    partition = BASE.normalize_partition(candidate.box_ids for candidate in selected_candidates)

    result = BASE.ScheduleResult(
        scenario="CP-SAT强化字典序方案",
        priority=objective_priority_text(),
        partition=partition,
        sorties=sorties,
        deliveries=deliveries,
        hard_violation_count=int((hard_lateness > TOLERANCE).sum()),
        hard_lateness_s=float(hard_lateness.sum()),
        weighted_tardiness=weighted_tardiness,
        makespan_s=max(row["返回O01时刻_s"] for row in sorties),
        total_energy_kwh=sum(row["架次能耗_kWh"] for row in sorties),
        sortie_count=len(sorties),
        desired_on_time_count=int(
            (delivery_frame["期望时间迟到_s"] <= TOLERANCE).sum()
        ),
        first_batch_on_time_count=int(
            first_rows["硬时限是否满足"].eq("是").sum()
        ),
        medical_on_time_count=int(
            medical_rows["硬时限是否满足"].eq("是").sum()
        ),
    )
    BASE.validate_result(result)
    return MasterSolution(
        result=result,
        selected_candidates=selected_candidates,
        start_times=start_times,
        drone_assignment=assigned_drone,
        battery_assignment=assigned_battery,
        objective_tuple=objective_tuple,
        stage_log=stage_log,
    )


def write_solution(
    best: MasterSolution,
    all_stage_logs: Sequence[dict],
    round_records: Sequence[dict],
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(best.result.sorties).to_csv(
        OUTPUT_DIR / "问题二第一问_CP-SAT强化方案运输架次.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(best.result.deliveries).to_csv(
        OUTPUT_DIR / "问题二第一问_CP-SAT强化方案逐箱交付.csv",
        index=False,
        encoding="utf-8-sig",
    )
    BASE.validate_result(best.result).to_csv(
        OUTPUT_DIR / "问题二第一问_CP-SAT强化方案检查.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(all_stage_logs).to_csv(
        OUTPUT_DIR / "问题二第一问_CP-SAT字典序求解记录.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(round_records).to_csv(
        OUTPUT_DIR / "问题二第一问_CP-SAT候选池迭代记录.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        [
            {
                "加权迟到_优先系数乘秒": best.result.weighted_tardiness,
                "全部任务完成时间_s": best.result.makespan_s,
                "总运输能耗_kWh": best.result.total_energy_kwh,
                "运输架次数": best.result.sortie_count,
                "期望时间内送达箱数": best.result.desired_on_time_count,
                "首批按时箱数": best.result.first_batch_on_time_count,
                "医疗物资按时箱数": best.result.medical_on_time_count,
                "候选池整数目标": str(best.objective_tuple),
                "已完成字典序阶段数": len(best.stage_log),
                "已证明最优阶段数": sum(
                    row["已证明最优"] == "是" for row in best.stage_log
                ),
                "当前候选池四阶段全部证明最优": (
                    "是"
                    if len(best.stage_log) == 4
                    and all(row["已证明最优"] == "是" for row in best.stage_log)
                    else "否"
                ),
                "最优性口径": "仅对当前有限候选架次集合成立，不代表80箱原问题全局最优",
            }
        ]
    ).to_csv(
        OUTPUT_DIR / "问题二第一问_CP-SAT强化方案指标.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with (OUTPUT_DIR / "问题二第一问_CP-SAT运行配置.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            {
                "随机种子": RANDOM_SEED,
                "候选评分设备": COMPUTE_DEVICE,
                "GPU预筛选是否启用": GPU_PREFILTER_ENABLED,
                "GPU名称": GPU_NAME,
                "CP-SAT求解器": "CPU多线程",
                "目标模式": OBJECTIVE_MODE,
                "零迟到硬约束": ZERO_TARDINESS_REQUIRED,
                "字典序阶段记录数": len(best.stage_log),
                "主参数": {
                    "MASTER_ROUNDS": MASTER_ROUNDS,
                    "RANDOM_WALK_STEPS_PER_ROUND": RANDOM_WALK_STEPS_PER_ROUND,
                    "BRUTE_FORCE_MAX_UNION_BOXES": BRUTE_FORCE_MAX_UNION_BOXES,
                    "BRUTE_FORCE_PAIR_LIMIT": BRUTE_FORCE_PAIR_LIMIT,
                    "MAX_ROUTE_GROUPS": MAX_ROUTE_GROUPS,
                    "CP_SAT_TIME_LIMIT_PER_STAGE_S": CP_SAT_TIME_LIMIT_PER_STAGE_S,
                    "CP_SAT_WORKERS": CP_SAT_WORKERS,
                    "EXHAUSTIVE_GROUP_MAX_SIZE": EXHAUSTIVE_GROUP_MAX_SIZE,
                    "MAX_ROUTE_ORDERS_PER_DRONE": MAX_ROUTE_ORDERS_PER_DRONE,
                },
            },
            file,
            ensure_ascii=False,
            indent=2,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="问题二第一问：穷举小组批、LNS候选扩展与CP-SAT强化求解"
    )
    parser.add_argument("--rounds", type=int, default=MASTER_ROUNDS)
    parser.add_argument(
        "--walk-steps", type=int, default=RANDOM_WALK_STEPS_PER_ROUND
    )
    parser.add_argument(
        "--brute-pair-limit", type=int, default=BRUTE_FORCE_PAIR_LIMIT
    )
    parser.add_argument(
        "--max-union-boxes", type=int, default=BRUTE_FORCE_MAX_UNION_BOXES
    )
    parser.add_argument("--max-route-groups", type=int, default=MAX_ROUTE_GROUPS)
    parser.add_argument(
        "--time-limit", type=float, default=CP_SAT_TIME_LIMIT_PER_STAGE_S
    )
    parser.add_argument("--workers", type=int, default=CP_SAT_WORKERS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--exhaustive-size",
        type=int,
        choices=[1, 2, 3],
        default=EXHAUSTIVE_GROUP_MAX_SIZE,
        help="穷举所有不超过该箱数的可行组批；3会明显增加预处理时间",
    )
    parser.add_argument(
        "--base-route-orders-only",
        action="store_true",
        help="只使用原脚本为每个机型保留的单一服务区访问顺序",
    )
    parser.add_argument(
        "--max-orders-per-drone",
        type=int,
        default=MAX_ROUTE_ORDERS_PER_DRONE,
    )
    parser.add_argument(
        "--max-stops",
        type=int,
        default=MAX_STOPS_LIMIT,
        help="单架次最多访问的服务区数；题面未规定上限，较大取值会增加航序搜索量",
    )
    parser.add_argument(
        "--solver-log", action="store_true", help="显示OR-Tools底层搜索日志"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu"],
        default=DEVICE_REQUEST,
        help="候选组批评分设备；CP-SAT始终使用CPU",
    )
    parser.add_argument(
        "--objective",
        choices=["timeliness", "makespan", "energy", "sortie"],
        default=OBJECTIVE_MODE,
        help="字典序目标：及时性、完工时间、能耗或架次",
    )
    parser.add_argument(
        "--zero-tardiness",
        action="store_true",
        help="将所有货箱的期望时间迟到固定为0，再优化其余目标",
    )
    parser.add_argument(
        "--max-sorties",
        type=int,
        default=None,
        help="限制运输架次数不超过该值，用于目标可行性检验",
    )
    parser.add_argument(
        "--max-makespan",
        type=int,
        default=None,
        help="限制最后一架无人机返航O01的时刻不超过该秒数",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="本次运行的独立输出目录",
    )
    return parser.parse_args()


def apply_arguments(args: argparse.Namespace) -> None:
    global RANDOM_SEED
    global MASTER_ROUNDS
    global RANDOM_WALK_STEPS_PER_ROUND
    global BRUTE_FORCE_MAX_UNION_BOXES
    global BRUTE_FORCE_PAIR_LIMIT
    global MAX_ROUTE_GROUPS
    global CP_SAT_TIME_LIMIT_PER_STAGE_S
    global CP_SAT_WORKERS
    global EXHAUSTIVE_GROUP_MAX_SIZE
    global ENABLE_ROUTE_ORDER_VARIANTS
    global MAX_ROUTE_ORDERS_PER_DRONE
    global SOLVER_LOG
    global OUTPUT_DIR
    global OBJECTIVE_MODE
    global ZERO_TARDINESS_REQUIRED
    global MAX_SORTIES_LIMIT
    global MAX_MAKESPAN_LIMIT_S
    global MAX_STOPS_LIMIT

    RANDOM_SEED = args.seed
    MASTER_ROUNDS = max(1, args.rounds)
    RANDOM_WALK_STEPS_PER_ROUND = max(0, args.walk_steps)
    BRUTE_FORCE_PAIR_LIMIT = max(0, args.brute_pair_limit)
    BRUTE_FORCE_MAX_UNION_BOXES = max(2, args.max_union_boxes)
    MAX_ROUTE_GROUPS = max(100, args.max_route_groups)
    CP_SAT_TIME_LIMIT_PER_STAGE_S = max(1.0, args.time_limit)
    CP_SAT_WORKERS = max(1, args.workers)
    EXHAUSTIVE_GROUP_MAX_SIZE = args.exhaustive_size
    ENABLE_ROUTE_ORDER_VARIANTS = not args.base_route_orders_only
    MAX_ROUTE_ORDERS_PER_DRONE = max(1, args.max_orders_per_drone)
    MAX_STOPS_LIMIT = max(1, min(15, args.max_stops))
    BASE.MAX_STOPS_PER_SORTIE = MAX_STOPS_LIMIT
    SOLVER_LOG = args.solver_log
    OUTPUT_DIR = Path(args.output_dir).resolve()
    OBJECTIVE_MODE = args.objective
    ZERO_TARDINESS_REQUIRED = args.zero_tardiness
    MAX_SORTIES_LIMIT = None if args.max_sorties is None else max(1, args.max_sorties)
    MAX_MAKESPAN_LIMIT_S = (
        None if args.max_makespan is None else max(1, args.max_makespan)
    )
    configure_compute_device(args.device)


def solution_comparison_key(solution: MasterSolution) -> Tuple[float, ...]:
    result = solution.result
    return solution.objective_tuple + (
        result.weighted_tardiness,
        result.makespan_s,
        result.total_energy_kwh,
        float(result.sortie_count),
    )


def main() -> None:
    args = parse_args()
    apply_arguments(args)

    print("初始化问题二数据、DEM航段、实体无人机和共享电池参数……")
    initialize_problem()
    partitions = load_existing_partitions()
    if not partitions:
        raise RuntimeError("没有找到可用的初始货箱分区。")

    seed_scenario = {
        "timeliness": "及时性优先",
        "makespan": "完工时间优先",
        "energy": "能耗优先",
        "sortie": "架次优先",
    }[OBJECTIVE_MODE]
    seed_results = [
        BASE.schedule_partition(partition, seed_scenario)
        for partition in partitions
    ]
    feasible_seed_results = [
        result
        for result in seed_results
        if result.hard_violation_count == 0
        and (
            not ZERO_TARDINESS_REQUIRED
            or result.weighted_tardiness <= TOLERANCE
        )
        and (
            MAX_SORTIES_LIMIT is None
            or result.sortie_count <= MAX_SORTIES_LIMIT
        )
        and (
            MAX_MAKESPAN_LIMIT_S is None
            or result.makespan_s <= MAX_MAKESPAN_LIMIT_S + TOLERANCE
        )
    ]
    anchor_result = min(
        feasible_seed_results if feasible_seed_results else seed_results,
        key=BASE.objective_key,
    )
    print(
        "已有方案最好值："
        f"加权迟到={anchor_result.weighted_tardiness:.3f}，"
        f"完工时间={anchor_result.makespan_s:.2f}s，"
        f"能耗={anchor_result.total_energy_kwh:.6f}kWh，"
        f"架次={anchor_result.sortie_count}。"
    )

    pool = RouteGroupPool()
    for box_id in BASE.BOX_BY_ID:
        pool.add_group([box_id], mandatory=True)
    for partition in partitions:
        pool.add_partition(partition, mandatory=True)

    tested, exhaustive_added = exhaustive_small_group_expand(
        pool, EXHAUSTIVE_GROUP_MAX_SIZE
    )
    print(
        f"小规模穷举完成：测试{tested}组，新增{exhaustive_added}个可行组批。"
    )

    best_solution: Optional[MasterSolution] = None
    all_stage_logs: List[dict] = []
    round_records: List[dict] = []
    rng = random.Random(RANDOM_SEED)

    for round_number in range(1, MASTER_ROUNDS + 1):
        print(f"\n[{round_number}/{MASTER_ROUNDS}] 扩展候选组批……")
        brute_added = brute_force_two_route_repacking(
            pool, anchor_result.partition, rng
        )
        random_added = random_walk_expand(
            pool,
            partitions + [anchor_result.partition],
            RANDOM_WALK_STEPS_PER_ROUND,
            rng,
        )
        before_prune = len(pool.groups)
        pool.prune()
        candidates = pool.candidates()
        print(
            f"候选组批：剪枝前{before_prune}，剪枝后{len(pool.groups)}；"
            f"具体机型/航序候选架次={len(candidates)}。"
        )

        solver, variables, logs, objective_tuple = solve_lexicographic_master(
            candidates,
            round_number,
            anchor_result,
        )
        solution = extract_master_solution(
            candidates,
            solver,
            variables,
            logs,
            objective_tuple,
        )
        all_stage_logs.extend(logs)
        all_stages_optimal = (
            len(logs) == 4
            and all(row["已证明最优"] == "是" for row in logs)
        )
        round_records.append(
            {
                "候选池轮次": round_number,
                "剪枝前组批数": before_prune,
                "剪枝后组批数": len(pool.groups),
                "候选架次数": len(candidates),
                "随机扩展新增组数": random_added,
                "暴力重组新增组数": brute_added,
                "初始穷举新增组数": exhaustive_added if round_number == 1 else 0,
                "加权迟到_优先系数乘秒": solution.result.weighted_tardiness,
                "全部任务完成时间_s": solution.result.makespan_s,
                "总运输能耗_kWh": solution.result.total_energy_kwh,
                "运输架次数": solution.result.sortie_count,
                "完成字典序阶段数": len(logs),
                "四阶段全部证明最优": "是" if all_stages_optimal else "否",
            }
        )

        if (
            best_solution is None
            or solution_comparison_key(solution)
            < solution_comparison_key(best_solution)
        ):
            best_solution = solution

        partitions.append(solution.result.partition)
        partitions = list(dict.fromkeys(partitions))
        pool.add_partition(solution.result.partition, mandatory=True)
        if best_solution is None:
            raise AssertionError("本轮没有得到可行解。")
        anchor_result = best_solution.result
        write_solution(best_solution, all_stage_logs, round_records)

        last_log = logs[-1]
        print(
            f"本轮结果：加权迟到={solution.result.weighted_tardiness:.3f}，"
            f"完工时间={solution.result.makespan_s:.2f}s，"
            f"能耗={solution.result.total_energy_kwh:.6f}kWh，"
            f"架次={solution.result.sortie_count}；"
            f"最深阶段={last_log['目标']}，状态={last_log['状态']}，"
            f"gap={last_log['相对gap']:.6g}。"
        )

    if best_solution is None:
        raise AssertionError("CP-SAT没有返回任何可行方案。")
    print("\n强化求解完成：")
    print(
        f"加权迟到={best_solution.result.weighted_tardiness:.3f}，"
        f"完工时间={best_solution.result.makespan_s:.2f}s，"
        f"能耗={best_solution.result.total_energy_kwh:.6f}kWh，"
        f"架次={best_solution.result.sortie_count}。"
    )
    print(f"输出目录：{OUTPUT_DIR}")
    print(
        "注意：OPTIMAL只表示当前有限候选架次集合内最优；"
        "除非穷举全部可行架次，否则不能宣称80箱原问题全局最优。"
    )


if __name__ == "__main__":
    main()
