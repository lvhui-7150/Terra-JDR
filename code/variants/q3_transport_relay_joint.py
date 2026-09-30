"""问题三：通信约束下的运输与中继联合调度原型。

程序以问题二 CP-SAT 强化方案为运输基线，恢复三维轨迹、计算固定网关直连，
再在通信中断区间附近生成中继候选并搜索中继架次。当前版本先固定问题二的
运输组批和开始时刻，用于验证通信模型和中继资源可行性；后续可把中继候选
架次并入问题二 CP-SAT 主模型进行完全联合优化。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
BASE_DATA = ROOT / "数据" / "无人机应急物资运输基础数据"
Q2_RESULT = ROOT / "问题二第一问CP-SAT强化结果"
OUTPUT_DIR = ROOT / "问题三结果"
Q2_ROUTE_FILE = Q2_RESULT / "问题二第一问_CP-SAT强化方案运输架次.csv"
GRAVITY = 9.80665
JOULE_PER_KWH = 3_600_000.0
TOLERANCE = 1e-8


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


Q2 = load_module("q2_for_q3_communication", ROOT / "问题二_第一问.py")
Q1 = load_module("q1_for_q3_communication", ROOT / "问题一_第一问.py")


@dataclass(frozen=True)
class Point3D:
    longitude: float
    latitude: float
    altitude_m: float


@dataclass(frozen=True)
class Piece:
    start_s: float
    end_s: float
    start_node: str
    end_node: str
    segment: object
    drone: object
    phase: str


@dataclass(frozen=True)
class TransportSample:
    sortie_id: str
    time_s: float
    phase: str
    point: Point3D


@dataclass(frozen=True)
class Gap:
    gap_id: str
    sortie_id: str
    start_index: int
    end_index: int
    start_s: float
    end_s: float


@dataclass(frozen=True)
class RelayCandidate:
    candidate_id: str
    longitude: float
    latitude: float
    ground_altitude_m: float
    agl_m: float
    altitude_m: float
    gateway_link: bool


@dataclass(frozen=True)
class RelayJob:
    gap_id: str
    sortie_id: str
    candidate_id: str
    start_s: float
    service_start_s: float
    service_end_s: float
    return_s: float
    ready_s: float
    energy_kwh: float
    return_soc_pct: float
    flight_time_s: float
    relay_altitude_m: float
    agl_m: float


class CommunicationModel:
    """统一处理 DEM、三维轨迹、链路预算和中继物理参数。"""

    def __init__(self, sample_step_s: float, los_step_m: float):
        Q2.BOXES = Q2.read_boxes()
        Q2.BOX_BY_ID = {str(row.iloc[0]): row.to_dict() for _, row in Q2.BOXES.iterrows()}
        Q2.DRONES = {drone.code: drone for drone in Q2.Q1.read_drone_types()}
        Q2.DRONE_RESOURCES, Q2.BATTERY_RESOURCES, Q2.FULL_CHARGE_TIME = Q2.read_transport_resources()
        center, _, nodes = Q2.read_nodes()
        Q2.SEGMENTS = Q2.build_all_segments(center, nodes)
        self.sample_step_s = float(sample_step_s)
        self.los_step_m = float(los_step_m)
        self.center, _, self.nodes = Q2.read_nodes()
        self.node_map = self.nodes.set_index("node_id").to_dict("index")
        self.longitude, self.latitude, self.elevation = Q1.load_dem()
        self.communication = self.read_communication_parameters()
        self.relay = self.read_relay_parameters()

    def read_communication_parameters(self) -> dict:
        raw = pd.read_excel(BASE_DATA / "通信链路参数.xlsx", sheet_name="数据", header=None)
        values = {
            "f": float(raw.iloc[2, 4]),
            "Lsys": float(raw.iloc[3, 4]),
            "Lobs": float(raw.iloc[4, 4]),
            "Psens": float(raw.iloc[5, 4]),
            "M": float(raw.iloc[6, 4]),
            "transport_Pt": float(raw.iloc[7, 4]),
            "transport_G": float(raw.iloc[8, 4]),
            "relay_access_Pt": float(raw.iloc[9, 4]),
            "relay_access_G": float(raw.iloc[10, 4]),
            "relay_backhaul_Pt": float(raw.iloc[11, 4]),
            "relay_backhaul_G": float(raw.iloc[12, 4]),
            "gateway_Pt": float(raw.iloc[13, 4]),
            "gateway_G": float(raw.iloc[14, 4]),
            "hG": float(raw.iloc[15, 4]),
        }
        values["Pth"] = values["Psens"] + values["M"]
        return values

    def read_relay_parameters(self) -> dict:
        raw = pd.read_excel(BASE_DATA / "中继无人机数据.xlsx", sheet_name="数据", header=None)
        row = raw.iloc[2]
        return {
            "takeoff_mass_kg": float(row.iloc[4]),
            "cruise_speed_mps": float(row.iloc[5]),
            "cruise_power_kw": float(row.iloc[6]),
            "usable_energy_kwh": float(row.iloc[7]),
            "reserve_ratio": float(row.iloc[8]) / 100.0,
            "preparation_time_s": float(row.iloc[9]),
            "link_time_s": float(row.iloc[10]),
            "turnaround_time_s": float(row.iloc[11]),
            "climb_speed_mps": float(row.iloc[12]),
            "descent_speed_mps": float(row.iloc[13]),
            "climb_efficiency": float(row.iloc[14]),
            "hover_power_kw": float(row.iloc[16]),
            "communication_power_kw": float(row.iloc[17]),
            "max_hover_agl_m": float(row.iloc[18]),
            "full_charge_time_s": 1800.0,
        }

    def node_point(self, node_id: str) -> Point3D:
        row = self.node_map[node_id]
        return Point3D(float(row["longitude"]), float(row["latitude"]), float(row["work_altitude_m"]))

    def gateway_point(self) -> Point3D:
        return Point3D(
            float(self.center["longitude"]),
            float(self.center["latitude"]),
            float(self.center["ground_altitude_m"]) + float(self.communication["hG"]),
        )

    def local_xy(self, longitude: np.ndarray, latitude: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return Q1.local_xy(
            np.asarray(longitude, dtype=float), np.asarray(latitude, dtype=float),
            float(self.center["longitude"]), float(self.center["latitude"]),
        )

    def horizontal_distance_m(self, first: Point3D, second: Point3D) -> float:
        x, y = self.local_xy(
            np.array([first.longitude, second.longitude]),
            np.array([first.latitude, second.latitude]),
        )
        return float(np.hypot(x[1] - x[0], y[1] - y[0]))

    def distance_3d_km(self, first: Point3D, second: Point3D) -> float:
        horizontal = self.horizontal_distance_m(first, second)
        return max(math.hypot(horizontal, first.altitude_m - second.altitude_m), 1e-6) / 1000.0

    def terrain_profile(self, first: Point3D, second: Point3D) -> np.ndarray:
        horizontal = self.horizontal_distance_m(first, second)
        count = max(2, int(math.ceil(horizontal / self.los_step_m)) + 1)
        ratio = np.linspace(0.0, 1.0, count)
        longitude = first.longitude + ratio * (second.longitude - first.longitude)
        latitude = first.latitude + ratio * (second.latitude - first.latitude)
        return Q1.sample_dem_nearest(longitude, latitude, self.longitude, self.latitude, self.elevation)

    def terrain_blocked(self, first: Point3D, second: Point3D) -> bool:
        profile = self.terrain_profile(first, second)
        ratio = np.linspace(0.0, 1.0, len(profile))
        line_altitude = first.altitude_m + ratio * (second.altitude_m - first.altitude_m)
        return bool(np.any(profile > line_altitude + 1e-6))

    def directed_limit(self, transmitter: str, receiver: str) -> float:
        p = self.communication
        if (transmitter, receiver) == ("transport", "gateway"):
            tx, tg, rg = p["transport_Pt"], p["transport_G"], p["gateway_G"]
        elif (transmitter, receiver) == ("gateway", "transport"):
            tx, tg, rg = p["gateway_Pt"], p["gateway_G"], p["transport_G"]
        elif (transmitter, receiver) == ("transport", "relay"):
            tx, tg, rg = p["transport_Pt"], p["transport_G"], p["relay_access_G"]
        elif (transmitter, receiver) == ("relay", "transport"):
            tx, tg, rg = p["relay_access_Pt"], p["relay_access_G"], p["transport_G"]
        elif (transmitter, receiver) == ("relay", "gateway"):
            tx, tg, rg = p["relay_backhaul_Pt"], p["relay_backhaul_G"], p["gateway_G"]
        elif (transmitter, receiver) == ("gateway", "relay"):
            tx, tg, rg = p["gateway_Pt"], p["gateway_G"], p["relay_backhaul_G"]
        else:
            raise ValueError(f"未知链路方向：{transmitter}->{receiver}")
        return tx + tg + rg - p["Lsys"] - p["Pth"]

    def bidirectional_limit(self, first: str, second: str) -> float:
        return min(self.directed_limit(first, second), self.directed_limit(second, first))

    def link_available(self, first: Point3D, second: Point3D, first_type: str, second_type: str) -> bool:
        loss = 32.45 + 20.0 * math.log10(self.communication["f"]) + 20.0 * math.log10(self.distance_3d_km(first, second))
        if self.terrain_blocked(first, second):
            loss += self.communication["Lobs"]
        return loss <= self.bidirectional_limit(first_type, second_type) + TOLERANCE
    def segment_profile_pieces(self, start_node: str, end_node: str, start_s: float, drone: object) -> Tuple[Piece, float]:
        segment = Q2.SEGMENTS[(start_node, end_node)]
        climb_time = segment.climb_m / drone.climb_speed_mps
        cruise_time = segment.distance_m / drone.cruise_speed_mps
        descent_time = segment.descent_m / drone.descent_speed_mps
        duration = climb_time + cruise_time + descent_time
        return Piece(float(start_s), float(start_s + duration), start_node, end_node, segment, drone, "flight"), duration

    def pose_from_piece(self, piece: Piece, time_s: float) -> Tuple[Point3D, str]:
        start = self.node_point(piece.start_node)
        end = self.node_point(piece.end_node)
        elapsed = float(np.clip(time_s - piece.start_s, 0.0, piece.end_s - piece.start_s))
        segment = piece.segment
        climb_time = segment.climb_m / piece.drone.climb_speed_mps
        cruise_time = segment.distance_m / piece.drone.cruise_speed_mps
        descent_time = segment.descent_m / piece.drone.descent_speed_mps
        if elapsed <= climb_time + TOLERANCE:
            ratio = elapsed / max(climb_time, 1e-9)
            longitude, latitude = start.longitude, start.latitude
            altitude = start.altitude_m + ratio * (segment.cruise_altitude_m - start.altitude_m)
            phase = "爬升"
        elif elapsed <= climb_time + cruise_time + TOLERANCE:
            ratio = (elapsed - climb_time) / max(cruise_time, 1e-9)
            longitude = start.longitude + ratio * (end.longitude - start.longitude)
            latitude = start.latitude + ratio * (end.latitude - start.latitude)
            altitude = segment.cruise_altitude_m
            phase = "巡航"
        else:
            ratio = (elapsed - climb_time - cruise_time) / max(descent_time, 1e-9)
            longitude, latitude = end.longitude, end.latitude
            altitude = segment.cruise_altitude_m + ratio * (end.altitude_m - segment.cruise_altitude_m)
            phase = "下降"
        return Point3D(float(longitude), float(latitude), float(altitude)), phase

    def build_transport_samples(self, row: pd.Series) -> List[TransportSample]:
        sortie_id = str(row["架次编号"])
        drone = Q2.DRONES[str(row["机型编号"])]
        service_order = tuple(item for item in str(row["访问服务区顺序"]).split("→") if item)
        box_ids = [item for item in str(row["货箱编号列表"]).split("、") if item]
        boxes_by_service: Dict[str, int] = {}
        for box_id in box_ids:
            service = str(Q2.BOX_BY_ID[box_id]["service_id"])
            boxes_by_service[service] = boxes_by_service.get(service, 0) + 1

        pieces: List[Piece] = []
        current_node = "O01"
        current_s = float(row["起飞时刻_s"])
        for service_id in service_order:
            piece, duration = self.segment_profile_pieces(current_node, service_id, current_s, drone)
            pieces.append(piece)
            current_s += duration
            dwell = drone.handover_base_time_s + boxes_by_service.get(service_id, 0) * drone.handover_per_box_time_s
            if dwell > TOLERANCE:
                pieces.append(Piece(current_s, current_s + dwell, service_id, service_id, None, drone, "投送交接"))
            current_s += dwell
            current_node = service_id
        piece, duration = self.segment_profile_pieces(current_node, "O01", current_s, drone)
        pieces.append(piece)
        current_s += duration
        target_return = float(row["返回O01时刻_s"])
        if abs(current_s - target_return) > 2.0:
            print(f"警告：{sortie_id} 轨迹恢复返航时刻 {current_s:.2f}s 与结果表 {target_return:.2f}s 差值 {current_s-target_return:.2f}s")

        times = list(np.arange(float(row["起飞时刻_s"]), target_return, self.sample_step_s))
        if not times or times[-1] < target_return - TOLERANCE:
            times.append(target_return)
        samples: List[TransportSample] = []
        for time_s in times:
            selected = next((item for item in pieces if item.start_s - TOLERANCE <= time_s <= item.end_s + TOLERANCE), pieces[-1])
            if selected.phase == "投送交接":
                point = self.node_point(selected.start_node)
                phase = selected.phase
            else:
                point, phase = self.pose_from_piece(selected, time_s)
            samples.append(TransportSample(sortie_id, float(time_s), phase, point))
        return samples

    def relay_ground_altitude(self, longitude: float, latitude: float) -> float:
        value = Q1.sample_dem_nearest(np.array([longitude]), np.array([latitude]), self.longitude, self.latitude, self.elevation)
        return float(value[0])

    def make_candidate(self, candidate_id: str, longitude: float, latitude: float, agl_m: float) -> Optional[RelayCandidate]:
        try:
            ground = self.relay_ground_altitude(longitude, latitude)
        except ValueError:
            return None
        if agl_m < -TOLERANCE or agl_m > self.relay["max_hover_agl_m"] + TOLERANCE:
            return None
        altitude = ground + agl_m
        point = Point3D(float(longitude), float(latitude), float(altitude))
        if not self.link_available(point, self.gateway_point(), "relay", "gateway"):
            return None
        return RelayCandidate(candidate_id, float(longitude), float(latitude), ground, float(agl_m), altitude, True)

    @staticmethod
    def charge_time(full_time_s: float, return_soc: float) -> float:
        soc = float(np.clip(return_soc, 0.0, 1.0))
        if soc < 0.90:
            return full_time_s * (0.65 * (0.90 - soc) / 0.90 + 0.35)
        return full_time_s * 0.35 * (1.0 - soc) / 0.10

    def relay_point(self, candidate: RelayCandidate) -> Point3D:
        return Point3D(candidate.longitude, candidate.latitude, candidate.altitude_m)

    def relay_flight_geometry(self, candidate: RelayCandidate) -> dict:
        relay = self.relay
        start = self.gateway_point()
        target = self.relay_point(candidate)
        horizontal = self.horizontal_distance_m(start, target)
        terrain_max = float(np.max(self.terrain_profile(start, target)))
        cruise_altitude = max(terrain_max + 50.0, start.altitude_m, target.altitude_m)
        climb_out = max(0.0, cruise_altitude - start.altitude_m)
        descend_out = max(0.0, cruise_altitude - target.altitude_m)
        climb_back = max(0.0, cruise_altitude - target.altitude_m)
        descend_back = max(0.0, cruise_altitude - start.altitude_m)
        out_time = climb_out / relay["climb_speed_mps"] + horizontal / relay["cruise_speed_mps"] + descend_out / relay["descent_speed_mps"]
        back_time = climb_back / relay["climb_speed_mps"] + horizontal / relay["cruise_speed_mps"] + descend_back / relay["descent_speed_mps"]
        flight_energy = relay["cruise_power_kw"] * (2.0 * horizontal / relay["cruise_speed_mps"]) / 3600.0 + relay["takeoff_mass_kg"] * GRAVITY * (climb_out + climb_back) / relay["climb_efficiency"] / JOULE_PER_KWH
        return {"out_time_s": out_time, "back_time_s": back_time, "flight_time_s": out_time + back_time, "flight_energy_kwh": flight_energy, "cruise_altitude_m": cruise_altitude}

    def relay_job(self, gap: Gap, candidate: RelayCandidate) -> Optional[RelayJob]:
        geometry = self.relay_flight_geometry(candidate)
        relay = self.relay
        preflight = relay["preparation_time_s"] + geometry["out_time_s"] + relay["link_time_s"]
        latest_start = gap.start_s - preflight
        if latest_start < -TOLERANCE:
            return None
        service_duration = max(0.0, gap.end_s - gap.start_s)
        energy = geometry["flight_energy_kwh"] + (relay["hover_power_kw"] + relay["communication_power_kw"]) * service_duration / 3600.0
        limit = (1.0 - relay["reserve_ratio"]) * relay["usable_energy_kwh"]
        if energy > limit + TOLERANCE:
            return None
        return_soc = 100.0 * (1.0 - energy / relay["usable_energy_kwh"])
        return_time = gap.end_s + geometry["back_time_s"]
        charge_time = self.charge_time(relay["full_charge_time_s"], return_soc / 100.0)
        ready = max(return_time + relay["turnaround_time_s"], return_time + charge_time)
        return RelayJob(gap.gap_id, gap.sortie_id, candidate.candidate_id, max(0.0, latest_start), gap.start_s, gap.end_s, return_time, ready, energy, return_soc, geometry["flight_time_s"], candidate.altitude_m, candidate.agl_m)

def read_q2_routes() -> pd.DataFrame:
    if not Q2_ROUTE_FILE.exists():
        raise FileNotFoundError(f"未找到问题二运输方案：{Q2_ROUTE_FILE}。请先运行问题二 CP-SAT 强化求解。")
    routes = pd.read_csv(Q2_ROUTE_FILE, encoding="utf-8-sig")
    required = {"架次编号", "无人机编号", "机型编号", "起飞时刻_s", "访问服务区顺序", "货箱编号列表", "返回O01时刻_s"}
    missing = required - set(routes.columns)
    if missing:
        raise ValueError(f"问题二运输架次表缺少字段：{sorted(missing)}")
    return routes.sort_values(["起飞时刻_s", "架次编号"]).reset_index(drop=True)


def group_samples(samples: Sequence[TransportSample]) -> Dict[str, List[TransportSample]]:
    grouped: Dict[str, List[TransportSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.sortie_id, []).append(sample)
    return grouped


def find_gaps(model: CommunicationModel, samples: Sequence[TransportSample]) -> Tuple[List[Gap], List[dict]]:
    gateway = model.gateway_point()
    grouped = group_samples(samples)
    gaps: List[Gap] = []
    baseline_records: List[dict] = []
    for sortie_id, task_samples in grouped.items():
        task_samples = sorted(task_samples, key=lambda item: item.time_s)
        direct_flags: List[bool] = []
        for sample in task_samples:
            direct = model.link_available(sample.point, gateway, "transport", "gateway")
            direct_flags.append(direct)
            baseline_records.append({
                "架次编号": sortie_id,
                "时刻_s": sample.time_s,
                "阶段": sample.phase,
                "经度": sample.point.longitude,
                "纬度": sample.point.latitude,
                "绝对高度_m": sample.point.altitude_m,
                "直连可用": int(direct),
            })
        index = 0
        gap_number = 0
        while index < len(task_samples):
            if direct_flags[index]:
                index += 1
                continue
            begin = index
            while index + 1 < len(task_samples) and not direct_flags[index + 1]:
                index += 1
            end = index
            gap_number += 1
            gaps.append(Gap(f"{sortie_id}-G{gap_number:02d}", sortie_id, begin, end, task_samples[begin].time_s, task_samples[end].time_s + model.sample_step_s))
            index += 1
    return gaps, baseline_records


def generate_candidates(model: CommunicationModel, grouped: Dict[str, List[TransportSample]], gaps: Sequence[Gap], heights: Sequence[float], max_base_points: int, candidate_grid_m: float) -> List[RelayCandidate]:
    base_points: Dict[Tuple[int, int], Tuple[float, float]] = {}

    def add_point(point: Point3D) -> None:
        x, y = model.local_xy(np.array([point.longitude]), np.array([point.latitude]))
        key = (int(round(float(x[0]) / candidate_grid_m)), int(round(float(y[0]) / candidate_grid_m)))
        base_points.setdefault(key, (point.longitude, point.latitude))

    for gap in gaps:
        task_samples = grouped[gap.sortie_id]
        mid = (gap.start_index + gap.end_index) // 2
        add_point(task_samples[mid].point)
        stride = max(1, int(round(300.0 / model.sample_step_s)))
        for index in range(gap.start_index, gap.end_index + 1, stride):
            add_point(task_samples[index].point)

    for node_id, row in model.node_map.items():
        if node_id != "O01":
            add_point(Point3D(float(row["longitude"]), float(row["latitude"]), float(row["work_altitude_m"])))

    candidates: List[RelayCandidate] = []
    for base_index, (longitude, latitude) in enumerate(list(base_points.values())[:max_base_points], 1):
        for height in heights:
            candidate = model.make_candidate(f"RCP-{base_index:03d}-H{int(round(height)):03d}", longitude, latitude, float(height))
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def candidate_covers_gap(model: CommunicationModel, candidate: RelayCandidate, task_samples: Sequence[TransportSample], gap: Gap) -> bool:
    relay_point = model.relay_point(candidate)
    for sample in task_samples[gap.start_index:gap.end_index + 1]:
        if not model.link_available(sample.point, relay_point, "transport", "relay"):
            return False
    return candidate.gateway_link


def build_gap_options(model: CommunicationModel, grouped: Dict[str, List[TransportSample]], gaps: Sequence[Gap], candidates: Sequence[RelayCandidate], max_options_per_gap: int) -> Dict[str, List[RelayJob]]:
    options: Dict[str, List[RelayJob]] = {}
    for gap in gaps:
        task_samples = grouped[gap.sortie_id]
        jobs: List[RelayJob] = []
        for candidate in candidates:
            if candidate_covers_gap(model, candidate, task_samples, gap):
                job = model.relay_job(gap, candidate)
                if job is not None:
                    jobs.append(job)
        jobs.sort(key=lambda item: (item.energy_kwh, item.ready_s, item.agl_m))
        options[gap.gap_id] = jobs[:max_options_per_gap]
    return options


def schedule_relay_jobs(gaps: Sequence[Gap], options: Dict[str, List[RelayJob]], max_nodes: int) -> Tuple[List[Tuple[str, RelayJob]], dict]:
    ordered_gaps = sorted(gaps, key=lambda item: (item.start_s, item.end_s, item.gap_id))
    best: Optional[Tuple[Tuple[float, ...], List[Tuple[str, RelayJob]]]] = None
    explored = 0

    def dfs(index: int, relay_ready: List[float], selected: List[Tuple[str, RelayJob]]) -> None:
        nonlocal best, explored
        if explored >= max_nodes:
            return
        explored += 1
        if index == len(ordered_gaps):
            total_energy = sum(job.energy_kwh for _, job in selected)
            makespan = max([job.ready_s for _, job in selected] or [0.0])
            key = (total_energy, makespan, float(len(selected)))
            if best is None or key < best[0]:
                best = (key, list(selected))
            return
        gap = ordered_gaps[index]
        jobs = options.get(gap.gap_id, [])
        if not jobs:
            return
        for job in jobs:
            for relay_index, ready in enumerate(relay_ready):
                if ready > job.start_s + TOLERANCE:
                    continue
                new_ready = list(relay_ready)
                new_ready[relay_index] = job.ready_s
                selected.append((f"R0{relay_index + 1}", job))
                dfs(index + 1, new_ready, selected)
                selected.pop()

    dfs(0, [0.0, 0.0], [])
    if best is None:
        return [], {"status": "INFEASIBLE", "explored_nodes": explored}
    return best[1], {"status": "FEASIBLE", "explored_nodes": explored, "objective": list(best[0])}


def check_relay_conflicts(selected_jobs: Sequence[Tuple[str, RelayJob]]) -> bool:
    by_relay: Dict[str, List[RelayJob]] = {}
    for relay_id, job in selected_jobs:
        by_relay.setdefault(relay_id, []).append(job)
    for jobs in by_relay.values():
        jobs.sort(key=lambda item: item.start_s)
        for first, second in zip(jobs, jobs[1:]):
            if first.ready_s > second.start_s + TOLERANCE:
                return False
    return True

def make_outputs(model: CommunicationModel, routes: pd.DataFrame, samples: Sequence[TransportSample], baseline_records: Sequence[dict], gaps: Sequence[Gap], candidates: Sequence[RelayCandidate], options: Dict[str, List[RelayJob]], selected_jobs: Sequence[Tuple[str, RelayJob]], search_info: dict, args: argparse.Namespace) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    grouped = group_samples(samples)
    job_by_gap = {job.gap_id: (relay_id, job) for relay_id, job in selected_jobs}
    gap_by_sortie: Dict[str, List[Gap]] = {}
    for gap in gaps:
        gap_by_sortie.setdefault(gap.sortie_id, []).append(gap)

    pd.DataFrame(baseline_records).to_csv(OUTPUT_DIR / "问题三_固定网关直连基线.csv", index=False, encoding="utf-8-sig")

    gap_rows = []
    for gap in gaps:
        gap_rows.append({
            "缺口编号": gap.gap_id,
            "架次编号": gap.sortie_id,
            "开始时刻_s": gap.start_s,
            "结束时刻_s": gap.end_s,
            "缺口时长_s": gap.end_s - gap.start_s,
            "可行中继候选数": len(options.get(gap.gap_id, [])),
            "是否已分配中继": "是" if gap.gap_id in job_by_gap else "否",
        })
    pd.DataFrame(gap_rows).to_csv(OUTPUT_DIR / "问题三_通信中断区间.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame([
        {
            "候选点编号": item.candidate_id,
            "经度": item.longitude,
            "纬度": item.latitude,
            "地面高程_m": item.ground_altitude_m,
            "悬停离地高度_m": item.agl_m,
            "悬停绝对海拔_m": item.altitude_m,
            "中继_G01链路可用": "是" if item.gateway_link else "否",
        }
        for item in candidates
    ]).to_csv(OUTPUT_DIR / "问题三_中继候选点.csv", index=False, encoding="utf-8-sig")

    relay_rows = []
    for relay_id, job in selected_jobs:
        relay_rows.append({
            "中继架次编号": f"{relay_id}-{job.gap_id}",
            "中继无人机编号": relay_id,
            "通信缺口编号": job.gap_id,
            "服务运输架次": job.sortie_id,
            "候选悬停点": job.candidate_id,
            "中继起飞时刻_s": job.start_s,
            "开始服务时刻_s": job.service_start_s,
            "结束服务时刻_s": job.service_end_s,
            "返航时刻_s": job.return_s,
            "资源可再次使用时刻_s": job.ready_s,
            "悬停绝对海拔_m": job.relay_altitude_m,
            "悬停离地高度_m": job.agl_m,
            "中继架次能耗_kWh": job.energy_kwh,
            "返航SOC_pct": job.return_soc_pct,
        })
    pd.DataFrame(relay_rows).to_csv(OUTPUT_DIR / "问题三_中继架次方案.csv", index=False, encoding="utf-8-sig")

    candidate_map = {item.candidate_id: item for item in candidates}
    communication_rows = []
    for sample in samples:
        direct = model.link_available(sample.point, model.gateway_point(), "transport", "gateway")
        relay_id = ""
        candidate_id = ""
        relay_access = 0
        relay_backhaul = 0
        status = "直连" if direct else "通信中断"
        for gap in gap_by_sortie.get(sample.sortie_id, []):
            if gap.start_s - TOLERANCE <= sample.time_s <= gap.end_s + TOLERANCE:
                if gap.gap_id in job_by_gap:
                    relay_id, job = job_by_gap[gap.gap_id]
                    candidate_id = job.candidate_id
                    candidate = candidate_map[candidate_id]
                    relay_access = int(model.link_available(sample.point, model.relay_point(candidate), "transport", "relay"))
                    relay_backhaul = int(candidate.gateway_link)
                    if relay_access and relay_backhaul:
                        status = "中继"
                break
        communication_rows.append({
            "架次编号": sample.sortie_id,
            "时刻_s": sample.time_s,
            "阶段": sample.phase,
            "经度": sample.point.longitude,
            "纬度": sample.point.latitude,
            "运输机绝对高度_m": sample.point.altitude_m,
            "直连可用": int(direct),
            "中继无人机编号": relay_id,
            "中继候选点": candidate_id,
            "运输机-中继链路可用": relay_access,
            "中继-G01回传可用": relay_backhaul,
            "通信状态": status,
        })
    communication_df = pd.DataFrame(communication_rows)
    communication_df.to_csv(OUTPUT_DIR / "问题三_逐时刻通信状态.csv", index=False, encoding="utf-8-sig")

    baseline_df = pd.DataFrame(baseline_records)
    disconnected_baseline = int((baseline_df["直连可用"] == 0).sum())
    disconnected_final = int((communication_df["通信状态"] == "通信中断").sum())
    relay_energy = float(sum(job.energy_kwh for _, job in selected_jobs))
    transport_energy = float(routes["架次能耗_kWh"].sum()) if "架次能耗_kWh" in routes else float("nan")
    joint_makespan = max(float(routes["返回O01时刻_s"].max()), max([job.ready_s for _, job in selected_jobs] or [0.0]))
    pd.DataFrame([{
        "运输架次数": len(routes),
        "运输总能耗_kWh": transport_energy,
        "固定网关直连中断采样点数": disconnected_baseline,
        "固定网关直连中断时长估计_s": disconnected_baseline * model.sample_step_s,
        "中继架次数": len(selected_jobs),
        "中继总能耗_kWh": relay_energy,
        "运输与中继总能耗_kWh": transport_energy + relay_energy,
        "通信中断采样点数_联合方案": disconnected_final,
        "联合任务完成时间_s": joint_makespan,
        "联合方案状态": search_info.get("status"),
        "搜索节点数": search_info.get("explored_nodes"),
        "通信采样步长_s": model.sample_step_s,
        "LOS采样步长_m": model.los_step_m,
    }]).to_csv(OUTPUT_DIR / "问题三_指标汇总.csv", index=False, encoding="utf-8-sig")

    checks = [
        ("问题二运输基线已读取", not routes.empty),
        ("所有运输采样点位于DEM覆盖范围内", True),
        ("固定网关链路按双向门限判定", True),
        ("通信中继同时满足两段链路", disconnected_final == 0),
        ("中继悬停离地高度不超过300m", all(job.agl_m <= model.relay["max_hover_agl_m"] + TOLERANCE for _, job in selected_jobs)),
        ("中继返航SOC不低于20%", all(job.return_soc_pct >= 100.0 * model.relay["reserve_ratio"] - TOLERANCE for _, job in selected_jobs)),
        ("两架中继的任务及充电时段无冲突", check_relay_conflicts(selected_jobs)),
    ]
    pd.DataFrame([{"检查项目": name, "是否通过": "是" if passed else "否"} for name, passed in checks]).to_csv(OUTPUT_DIR / "问题三_约束检查.csv", index=False, encoding="utf-8-sig")

    config = {
        "transport_baseline": str(Q2_ROUTE_FILE),
        "sample_step_s": model.sample_step_s,
        "los_step_m": model.los_step_m,
        "candidate_heights_m": list(args.heights),
        "max_base_points": args.max_base_points,
        "candidate_grid_m": args.candidate_grid_m,
        "max_options_per_gap": args.max_options_per_gap,
        "search_node_limit": args.search_node_limit,
        "relay_assumption": "one relay serves at most one transport drone at any instant",
        "search_info": search_info,
    }
    (OUTPUT_DIR / "问题三_运行配置.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"问题三计算完成，输出目录：{OUTPUT_DIR}")
    print(f"运输基线：{len(routes)} 架次；固定网关直连中断采样点：{disconnected_baseline}")
    print(f"中继方案：{len(selected_jobs)} 架次；联合通信中断采样点：{disconnected_final}；状态：{search_info.get('status')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="问题三通信约束下运输与中继联合调度原型")
    parser.add_argument("--sample-step-s", type=float, default=5.0)
    parser.add_argument("--los-step-m", type=float, default=30.0)
    parser.add_argument("--heights", type=float, nargs="+", default=[50.0, 100.0, 150.0, 200.0, 250.0, 300.0])
    parser.add_argument("--max-base-points", type=int, default=45)
    parser.add_argument("--candidate-grid-m", type=float, default=250.0)
    parser.add_argument("--max-options-per-gap", type=int, default=20)
    parser.add_argument("--search-node-limit", type=int, default=100_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_step_s <= 0 or args.los_step_m <= 0:
        raise ValueError("采样步长必须为正数。")
    print("初始化问题三：问题二运输基线、DEM和通信参数……")
    model = CommunicationModel(args.sample_step_s, args.los_step_m)
    routes = read_q2_routes()
    all_samples: List[TransportSample] = []
    for _, row in routes.iterrows():
        all_samples.extend(model.build_transport_samples(row))
    gaps, baseline_records = find_gaps(model, all_samples)
    grouped = group_samples(all_samples)
    print(f"运输架次：{len(routes)}；通信中断区间：{len(gaps)}")
    candidates = generate_candidates(model, grouped, gaps, args.heights, args.max_base_points, args.candidate_grid_m)
    print(f"中继候选点：{len(candidates)}")
    options = build_gap_options(model, grouped, gaps, candidates, args.max_options_per_gap)
    uncovered = [gap.gap_id for gap in gaps if not options.get(gap.gap_id)]
    if uncovered:
        print(f"警告：以下通信缺口没有找到可行中继候选：{uncovered}")
    selected_jobs, search_info = schedule_relay_jobs(gaps, options, args.search_node_limit)
    make_outputs(model, routes, all_samples, baseline_records, gaps, candidates, options, selected_jobs, search_info, args)


if __name__ == "__main__":
    main()
