"""求解问题一第二小问：优化单服务区往返的货箱组批。

运行前先运行“问题一_第一问.py”，生成可行候选组批表。
本脚本不重新计算航线能耗，只在第一小问基于题设固定水平直线航段、
并已验证安全性的候选组批中优化选择。
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix, vstack


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "数据" / "无人机应急物资运输基础数据"
FIRST_RESULT_DIR = ROOT / "问题一第一问结果"
OUTPUT_DIR = ROOT / "问题一第二问结果"
DIAGONAL_OUTPUT_DIR = ROOT / "问题一第二问斜飞结果"

CANDIDATE_FILE = FIRST_RESULT_DIR / "可行候选组批.csv"
FIRST_PLAN_FILE = FIRST_RESULT_DIR / "第一问可行方案.csv"
SAFE_PAYLOAD_FILE = FIRST_RESULT_DIR / "最大安全载荷.csv"
BOX_FILE = INPUT_DIR / "物资需求与配送时限.xlsx"
DRONE_FILE = INPUT_DIR / "运输无人机数据.xlsx"
ROUTE_FILE = FIRST_RESULT_DIR / "航段参数.csv"

ENERGY_TOLERANCE_KWH = 1e-8
TIME_LIMIT_SECONDS = 600
RESERVE_RATIO_OVERRIDE: Optional[float] = None
RUN_EXTENDED_SCENARIOS = True
RUN_TIME_COMPARISON = True


def read_inputs() -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """读取第一小问候选组批、货箱、机型和最大安全载荷数据。"""

    for input_file in [
        CANDIDATE_FILE,
        FIRST_PLAN_FILE,
        SAFE_PAYLOAD_FILE,
        ROUTE_FILE,
        BOX_FILE,
        DRONE_FILE,
    ]:
        if not input_file.exists():
            raise FileNotFoundError(
                f"缺少输入文件：{input_file}\n请先运行问题一_第一问.py，或检查附件路径。"
            )

    candidates = pd.read_csv(CANDIDATE_FILE, encoding="utf-8-sig")
    first_plan = pd.read_csv(FIRST_PLAN_FILE, encoding="utf-8-sig")
    safe_payloads = pd.read_csv(SAFE_PAYLOAD_FILE, encoding="utf-8-sig")
    boxes = pd.read_excel(BOX_FILE, sheet_name="逐箱货箱清单", header=0)
    drone_data = pd.read_excel(DRONE_FILE, sheet_name="数据", header=1)
    drones = drone_data[
        drone_data["机型编号"].isin(["A", "B", "C"])
        & drone_data["最大载货质量（kg）"].notna()
    ].copy()
    if RESERVE_RATIO_OVERRIDE is not None:
        if not 0.0 <= RESERVE_RATIO_OVERRIDE < 1.0:
            raise ValueError("返航安全余量必须位于[0,1)内。")
        drones["返航电量下限（%）"] = 100.0 * RESERVE_RATIO_OVERRIDE

    required_candidate_columns = {
        "服务区编号",
        "机型类型",
        "货箱编号列表",
        "货箱数量",
        "总质量_kg",
        "总体积_m3",
        "往返能耗_kWh",
        "任务能量上限_kWh",
        "返航剩余能量_kWh",
        "返航SOC_pct",
        "往返飞行时间_s",
        "估计架次作业时间_s",
    }
    missing_columns = required_candidate_columns.difference(candidates.columns)
    if missing_columns:
        raise ValueError(f"候选组批表缺少字段：{sorted(missing_columns)}")

    if len(boxes) != 80 or boxes["货箱编号"].nunique() != 80:
        raise ValueError("逐箱货箱清单应包含80个唯一货箱，请检查输入附件。")
    if len(candidates) == 0:
        raise ValueError("候选组批表为空，无法建立优化模型。")
    if set(drones["机型编号"]) != {"A", "B", "C"}:
        raise ValueError("未从运输无人机数据中完整读取 A、B、C 三种机型。")

    return candidates, first_plan, safe_payloads, boxes, drones


def diagonal_flight_time(
    distance_m: float,
    climb_height_m: float,
    descent_height_m: float,
    cruise_speed_mps: float,
    climb_speed_mps: float,
    descent_speed_mps: float,
) -> float:
    """计算严谨同步斜飞的最短航段时间。

    爬升、水平运动和下降按顺序分为三个阶段，但爬升/下降阶段允许
    同时具有水平速度分量。水平位移可在三个阶段之间分配，故最短时间为
    max(水平位移所需时间，竖直运动所需总时间)。
    """

    horizontal_time = distance_m / cruise_speed_mps
    vertical_time = (
        climb_height_m / climb_speed_mps
        + descent_height_m / descent_speed_mps
    )
    return max(horizontal_time, vertical_time)


def diagonal_operation_times(
    candidates: pd.DataFrame, drones: pd.DataFrame, routes: pd.DataFrame
) -> pd.DataFrame:
    """按斜向爬升/下降模型重算候选架次的时间，不改动能耗。"""

    drone_table = drones.set_index("机型编号")
    route_table = routes.set_index("服务区编号")
    diagonal_flight_times = []
    diagonal_operation_times = []

    for _, row in candidates.iterrows():
        drone = drone_table.loc[str(row["机型类型"])]
        route = route_table.loc[str(row["服务区编号"])]
        outbound = diagonal_flight_time(
            float(route["水平距离_m"]),
            float(route["去程爬升高度_m"]),
            float(route["去程下降高度_m"]),
            float(drone["计划巡航速度（m/s）"]),
            float(drone["最大爬升速度（m/s）"]),
            float(drone["最大下降速度（m/s）"]),
        )
        return_leg = diagonal_flight_time(
            float(route["水平距离_m"]),
            float(route["返程爬升高度_m"]),
            float(route["返程下降高度_m"]),
            float(drone["计划巡航速度（m/s）"]),
            float(drone["最大爬升速度（m/s）"]),
            float(drone["最大下降速度（m/s）"]),
        )
        flight_time = outbound + return_leg
        box_count = int(row["货箱数量"])
        operation_time = (
            flight_time
            + float(drone["工位固定准备时间（s）"])
            + box_count * float(drone["每箱装载时间（s）"])
            + float(drone["接收点基础交接时间（s）"])
            + box_count * float(drone["每箱增加交接时间（s）"])
        )
        diagonal_flight_times.append(flight_time)
        diagonal_operation_times.append(operation_time)

    result = candidates.copy()
    result["斜飞往返飞行时间_s"] = diagonal_flight_times
    result["斜飞架次作业时间_s"] = diagonal_operation_times
    return result


def validate_candidates(
    candidates: pd.DataFrame,
    safe_payloads: pd.DataFrame,
    boxes: pd.DataFrame,
    drones: pd.DataFrame,
) -> Tuple[List[List[str]], csc_matrix, List[str]]:
    """核验每个候选架次，并建立货箱—候选组批稀疏关联矩阵。"""

    box_ids = boxes["货箱编号"].astype(str).tolist()
    box_position = {box_id: index for index, box_id in enumerate(box_ids)}
    box_service = boxes.set_index("货箱编号")["服务区编号"].astype(str).to_dict()
    box_mass = boxes.set_index("货箱编号")["单箱质量（kg）"].astype(float).to_dict()
    box_volume = boxes.set_index("货箱编号")["单箱体积（m³）"].astype(float).to_dict()

    safe_payload_table = safe_payloads.set_index("服务区编号")
    drone_table = drones.set_index("机型编号")

    group_box_ids: List[List[str]] = []
    matrix_rows: List[int] = []
    matrix_columns: List[int] = []

    numeric_columns = [
        "货箱数量",
        "总质量_kg",
        "总体积_m3",
        "往返能耗_kWh",
        "任务能量上限_kWh",
        "返航剩余能量_kWh",
        "返航SOC_pct",
        "往返飞行时间_s",
        "估计架次作业时间_s",
    ]
    if not np.isfinite(candidates[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("候选组批表中存在空值或非有限数值。")

    for candidate_index, row in candidates.iterrows():
        service_id = str(row["服务区编号"])
        drone_code = str(row["机型类型"])
        selected_box_ids = str(row["货箱编号列表"]).split("、")

        if len(selected_box_ids) != int(row["货箱数量"]):
            raise ValueError(f"候选组批第 {candidate_index + 1} 行的货箱数与编号列表不一致。")
        if len(selected_box_ids) != len(set(selected_box_ids)):
            raise ValueError(f"候选组批第 {candidate_index + 1} 行重复包含同一货箱。")
        if any(box_id not in box_position for box_id in selected_box_ids):
            raise ValueError(f"候选组批第 {candidate_index + 1} 行含有未知货箱编号。")
        if {box_service[box_id] for box_id in selected_box_ids} != {service_id}:
            raise ValueError(f"候选组批第 {candidate_index + 1} 行跨服务区或服务区编号错误。")
        if service_id not in safe_payload_table.index or drone_code not in drone_table.index:
            raise ValueError(f"候选组批第 {candidate_index + 1} 行的服务区或机型无参数记录。")

        actual_mass = sum(box_mass[box_id] for box_id in selected_box_ids)
        actual_volume = sum(box_volume[box_id] for box_id in selected_box_ids)
        reported_mass = float(row["总质量_kg"])
        reported_volume = float(row["总体积_m3"])
        drone = drone_table.loc[drone_code]
        safe_payload = float(safe_payload_table.loc[service_id, f"{drone_code}型最大安全载荷_kg"])

        if not np.isclose(actual_mass, reported_mass, atol=1e-8, rtol=0.0):
            raise ValueError(f"候选组批第 {candidate_index + 1} 行的货物质量与逐箱清单不一致。")
        if not np.isclose(actual_volume, reported_volume, atol=1e-8, rtol=0.0):
            raise ValueError(f"候选组批第 {candidate_index + 1} 行的货物体积与逐箱清单不一致。")
        if actual_mass > safe_payload + 1e-8:
            raise ValueError(f"候选组批第 {candidate_index + 1} 行超过最大安全载荷。")
        if actual_volume > float(drone["可用装载体积（m³）"]) + 1e-8:
            raise ValueError(f"候选组批第 {candidate_index + 1} 行超过机型装载体积。")

        usable_energy = float(drone["电池可用能量（kWh）"])
        reserve_ratio = float(drone["返航电量下限（%）"]) / 100.0
        trip_energy = float(row["往返能耗_kWh"])
        energy_limit = (1.0 - reserve_ratio) * usable_energy
        remaining_energy = usable_energy - trip_energy
        return_soc_pct = 100.0 * remaining_energy / usable_energy

        if trip_energy > energy_limit + ENERGY_TOLERANCE_KWH:
            raise ValueError(f"候选组批第 {candidate_index + 1} 行违反整架次能耗上限。")
        if return_soc_pct + 1e-7 < 100.0 * reserve_ratio:
            raise ValueError(f"候选组批第 {candidate_index + 1} 行返航SOC低于下限。")
        if not np.isclose(
            float(row["任务能量上限_kWh"]), energy_limit, atol=1e-8, rtol=0.0
        ):
            raise ValueError(f"候选组批第 {candidate_index + 1} 行能耗上限与机型数据不一致。")
        if not np.isclose(
            float(row["返航剩余能量_kWh"]), remaining_energy, atol=1e-8, rtol=0.0
        ):
            raise ValueError(f"候选组批第 {candidate_index + 1} 行返航剩余能量不一致。")
        if not np.isclose(
            float(row["返航SOC_pct"]), return_soc_pct, atol=1e-7, rtol=0.0
        ):
            raise ValueError(f"候选组批第 {candidate_index + 1} 行返航SOC计算不一致。")

        count = len(selected_box_ids)
        recomputed_time = (
            float(row["往返飞行时间_s"])
            + float(drone["工位固定准备时间（s）"])
            + count * float(drone["每箱装载时间（s）"])
            + float(drone["接收点基础交接时间（s）"])
            + count * float(drone["每箱增加交接时间（s）"])
        )
        if not np.isclose(
            float(row["估计架次作业时间_s"]), recomputed_time, atol=1e-7, rtol=0.0
        ):
            raise ValueError(f"候选组批第 {candidate_index + 1} 行累计架次作业时间不一致。")

        group_box_ids.append(selected_box_ids)
        matrix_rows.extend(box_position[box_id] for box_id in selected_box_ids)
        matrix_columns.extend([candidate_index] * count)

    coverage_matrix = csc_matrix(
        (np.ones(len(matrix_rows)), (matrix_rows, matrix_columns)),
        shape=(len(box_ids), len(candidates)),
    )
    return group_box_ids, coverage_matrix, box_ids


def solve_stage(
    stage_name: str,
    objective: np.ndarray,
    constraints: Sequence[LinearConstraint],
    integrality: np.ndarray,
    bounds: Bounds,
    allow_infeasible: bool = False,
) -> Tuple[object, float]:
    """运行一个 MILP 阶段；可按需将已证明不可行作为情景结果返回。"""

    started = perf_counter()
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=bounds,
        constraints=list(constraints),
        options={"disp": False, "presolve": True, "time_limit": TIME_LIMIT_SECONDS, "mip_rel_gap": 0.0},
    )
    elapsed = perf_counter() - started

    if allow_infeasible and result.status == 2 and result.x is None:
        return result, elapsed
    if not result.success or result.x is None:
        raise RuntimeError(
            f"{stage_name} 未证明最优：status={result.status}, message={result.message}, "
            f"耗时={elapsed:.2f}s。当前程序不会把未证明最优的解标成最优方案。"
        )
    if not np.allclose(result.x, np.rint(result.x), atol=1e-6, rtol=0.0):
        raise RuntimeError(f"{stage_name} 返回的决策变量不是整数解。")

    return result, elapsed


def record_stage(
    stage_log: List[Dict[str, object]],
    scenario_name: str,
    stage_name: str,
    result: object,
    elapsed: float,
) -> None:
    """保存各优化情景中每个字典序阶段的求解状态。"""

    status = "已证明最优" if result.success else "不可行" if result.status == 2 else "未证明最优"
    stage_log.append(
        {
            "优化情景": scenario_name,
            "阶段": stage_name,
            "状态": status,
            "求解器状态码": int(result.status),
            "目标值": float(result.fun) if result.fun is not None else np.nan,
            "MIP_gap": float(getattr(result, "mip_gap", np.nan)),
            "求解时间_s": elapsed,
            "求解信息": str(result.message),
        }
    )


def solve_lexicographic(
    scenario_name: str,
    objectives: Sequence[Tuple[str, np.ndarray]],
    base_constraints: Sequence[LinearConstraint],
    integrality: np.ndarray,
    bounds: Bounds,
    stage_log: List[Dict[str, object]],
    allow_infeasible: bool = False,
) -> Optional[object]:
    """按给定优先级逐层优化，并固定每一层已取得的最优值。"""

    active_constraints = list(base_constraints)
    final_result = None

    for stage_number, (objective_name, objective) in enumerate(objectives, start=1):
        stage_name = f"第{stage_number}层：最小化{objective_name}"
        result, elapsed = solve_stage(
            f"{scenario_name}—{stage_name}",
            objective,
            active_constraints,
            integrality,
            bounds,
            allow_infeasible=allow_infeasible,
        )
        record_stage(stage_log, scenario_name, stage_name, result, elapsed)
        if result.status == 2 and result.x is None:
            return None

        final_result = result
        if stage_number < len(objectives):
            objective_value = float(np.dot(objective, np.rint(result.x)))
            objective_row = csc_matrix(np.asarray(objective, dtype=float).reshape(1, -1))
            if objective_name == "架次数":
                fixed_value = int(round(objective_value))
                active_constraints.append(
                    LinearConstraint(objective_row, lb=fixed_value, ub=fixed_value)
                )
            else:
                active_constraints.append(
                    LinearConstraint(objective_row, lb=-np.inf, ub=objective_value)
                )

    return final_result


def selected_rows(candidates: pd.DataFrame, solution: np.ndarray) -> pd.DataFrame:
    """将求解器的0-1变量转换为架次清单。"""

    indices = np.flatnonzero(solution > 0.5)
    chosen = candidates.iloc[indices].copy().reset_index(drop=True)
    chosen.insert(0, "方案架次编号", [f"Q1-2-{index:03d}" for index in range(1, len(chosen) + 1)])
    return chosen


def summarize_plan(
    name: str,
    plan: pd.DataFrame,
    priority: str,
    drone_volumes: Dict[str, float],
    feasible: bool = True,
) -> Dict[str, object]:
    """汇总方案的运输、时间、机型构成和货舱容积利用指标。"""

    if not feasible or plan.empty:
        return {
            "方案": name,
            "目标优先级": priority,
            "可行性": "不可行",
            "架次数": np.nan,
            "A型架次数": np.nan,
            "B型架次数": np.nan,
            "C型架次数": np.nan,
            "总运输能耗_kWh": np.nan,
            "累计架次作业时间_s": np.nan,
            "总装载体积_m3": np.nan,
            "总可用货舱容积_m3": np.nan,
            "总闲置货舱容积_m3": np.nan,
            "总体积利用率_pct": np.nan,
            "最小返航SOC_pct": np.nan,
        }

    loaded_volume = float(plan["总体积_m3"].sum())
    available_volume = float(
        plan["机型类型"].astype(str).map(drone_volumes).sum()
    )
    unused_volume = available_volume - loaded_volume
    if unused_volume < -1e-7:
        raise AssertionError(f"方案“{name}”的总装载体积超过可用货舱容积。")

    drone_counts = plan["机型类型"].astype(str).value_counts()
    return {
        "方案": name,
        "目标优先级": priority,
        "可行性": "可行",
        "架次数": len(plan),
        "A型架次数": int(drone_counts.get("A", 0)),
        "B型架次数": int(drone_counts.get("B", 0)),
        "C型架次数": int(drone_counts.get("C", 0)),
        "总运输能耗_kWh": float(plan["往返能耗_kWh"].sum()),
        "累计架次作业时间_s": float(plan["估计架次作业时间_s"].sum()),
        "总装载体积_m3": loaded_volume,
        "总可用货舱容积_m3": available_volume,
        "总闲置货舱容积_m3": max(0.0, unused_volume),
        "总体积利用率_pct": 100.0 * loaded_volume / available_volume,
        "最小返航SOC_pct": float(plan["返航SOC_pct"].min()),
    }


def check_exact_coverage(
    plan: pd.DataFrame,
    group_box_ids: Sequence[Sequence[str]],
    box_ids: Sequence[str],
) -> pd.DataFrame:
    """确认方案中每个货箱恰好出现一次。"""

    counts = {box_id: 0 for box_id in box_ids}
    for candidate_row in plan["货箱编号列表"].astype(str):
        for box_id in candidate_row.split("、"):
            if box_id not in counts:
                raise AssertionError(f"方案含有未知货箱：{box_id}")
            counts[box_id] += 1

    audit = pd.DataFrame(
        {
            "货箱编号": list(counts),
            "方案中出现次数": list(counts.values()),
        }
    )
    if not audit["方案中出现次数"].eq(1).all():
        raise AssertionError("精确覆盖复核失败：存在遗漏或重复货箱。")
    return audit


def main() -> None:
    candidates, first_plan, safe_payloads, boxes, drones = read_inputs()
    group_box_ids, coverage_matrix, box_ids = validate_candidates(
        candidates, safe_payloads, boxes, drones
    )

    candidate_count = len(candidates)
    integrality = np.ones(candidate_count, dtype=np.uint8)
    lower_bounds = np.zeros(candidate_count)
    upper_bounds = np.ones(candidate_count)
    bounds = Bounds(lower_bounds, upper_bounds)

    number_of_sorties = np.ones(candidate_count)
    energy = candidates["往返能耗_kWh"].to_numpy(dtype=float)
    operation_time = candidates["估计架次作业时间_s"].to_numpy(dtype=float)
    drone_volumes = (
        drones.set_index("机型编号")["可用装载体积（m³）"].astype(float).to_dict()
    )
    candidate_capacity = (
        candidates["机型类型"].astype(str).map(drone_volumes).to_numpy(dtype=float)
    )
    loaded_volume = candidates["总体积_m3"].to_numpy(dtype=float)
    unused_volume = candidate_capacity - loaded_volume
    if np.any(unused_volume < -1e-8):
        raise ValueError("候选组批中存在超过机型可用装载体积的组批。")
    unused_volume = np.maximum(unused_volume, 0.0)

    coverage_constraint = LinearConstraint(
        coverage_matrix,
        lb=np.ones(len(box_ids)),
        ub=np.ones(len(box_ids)),
    )
    sortie_row = csc_matrix(np.ones((1, candidate_count)))
    stage_log: List[Dict[str, object]] = []

    print(f"候选架次：{candidate_count:,}；货箱：{len(box_ids)}")
    print("阶段 1：计算全机型混合情况下的最少架次数……")
    result_sorties, time_sorties = solve_stage(
        "最少架次数优化",
        number_of_sorties,
        [coverage_constraint],
        integrality,
        bounds,
    )
    record_stage(stage_log, "共享基准", "最少架次数", result_sorties, time_sorties)
    minimum_sorties = int(round(float(result_sorties.fun)))

    exact_sortie_count = LinearConstraint(
        sortie_row,
        lb=float(minimum_sorties),
        ub=float(minimum_sorties),
    )

    main_result = solve_lexicographic(
        "架次优先主方案",
        [("总能耗", energy), ("累计架次作业时间", operation_time)],
        [coverage_constraint, exact_sortie_count],
        integrality,
        bounds,
        stage_log,
    )
    minimum_sortie_time_result = main_result
    if RUN_TIME_COMPARISON:
        minimum_sortie_time_result = solve_lexicographic(
            "最少架次下时间优先对照",
            [("累计架次作业时间", operation_time), ("总能耗", energy)],
            [coverage_constraint, exact_sortie_count],
            integrality,
            bounds,
            stage_log,
        )
    time_first_result: Optional[object] = None
    energy_first_result: Optional[object] = None
    volume_first_result: Optional[object] = None
    single_type_results: Dict[str, Optional[object]] = {}
    if RUN_EXTENDED_SCENARIOS:
        time_first_result = solve_lexicographic(
            "累计时间优先",
            [
                ("累计架次作业时间", operation_time),
                ("架次数", number_of_sorties),
                ("总能耗", energy),
            ],
            [coverage_constraint],
            integrality,
            bounds,
            stage_log,
        )
        energy_first_result = solve_lexicographic(
            "能耗优先",
            [
                ("总能耗", energy),
                ("架次数", number_of_sorties),
                ("累计架次作业时间", operation_time),
            ],
            [coverage_constraint],
            integrality,
            bounds,
            stage_log,
        )
        volume_first_result = solve_lexicographic(
            "最少架次下容积利用率优先",
            [
                ("总闲置货舱容积", unused_volume),
                ("总能耗", energy),
                ("累计架次作业时间", operation_time),
            ],
            [coverage_constraint, exact_sortie_count],
            integrality,
            bounds,
            stage_log,
        )

        for drone_code in ["A", "B", "C"]:
            type_upper_bounds = candidates["机型类型"].astype(str).eq(drone_code).to_numpy(dtype=float)
            type_bounds = Bounds(lower_bounds, type_upper_bounds)
            single_type_results[drone_code] = solve_lexicographic(
                f"仅使用{drone_code}型",
                [
                    ("架次数", number_of_sorties),
                    ("总能耗", energy),
                    ("累计架次作业时间", operation_time),
                ],
                [coverage_constraint],
                integrality,
                type_bounds,
                stage_log,
                allow_infeasible=True,
            )

    if main_result is None or minimum_sortie_time_result is None:
        raise AssertionError("混合机型下的基准情景应当可行。")

    main_plan = selected_rows(candidates, main_result.x)
    minimum_sortie_time_plan = selected_rows(candidates, minimum_sortie_time_result.x)
    if len(main_plan) != minimum_sorties or len(minimum_sortie_time_plan) != minimum_sorties:
        raise AssertionError("主方案或最少架次时间对照方案的架次数不一致。")

    scenario_definitions = [
        (
            "第一小问可行方案（基线）",
            "仅检验可行性，未进行三指标优化",
            "BASE",
            first_plan,
        ),
        (
            "架次优先主方案",
            "架次数 → 总能耗 → 累计架次作业时间",
            "SORTIE",
            main_plan,
        ),
    ]
    if RUN_TIME_COMPARISON:
        scenario_definitions.append(
            (
                "最少架次下时间优先对照",
                "固定最少架次 → 累计架次作业时间 → 总能耗",
                "TIMEFIX",
                minimum_sortie_time_plan,
            )
        )
    if RUN_EXTENDED_SCENARIOS:
        scenario_definitions.extend(
            [
                (
                    "累计时间优先",
                    "累计架次作业时间 → 架次数 → 总能耗",
                    "TIME",
                    selected_rows(candidates, time_first_result.x) if time_first_result else None,
                ),
                (
                    "能耗优先",
                    "总能耗 → 架次数 → 累计架次作业时间",
                    "ENERGY",
                    selected_rows(candidates, energy_first_result.x) if energy_first_result else None,
                ),
                (
                    "最少架次下容积利用率优先",
                    "固定最少架次 → 最小化总闲置货舱容积 → 总能耗 → 累计作业时间",
                    "VOLUME",
                    selected_rows(candidates, volume_first_result.x) if volume_first_result else None,
                ),
            ]
        )
        for drone_code in ["A", "B", "C"]:
            result = single_type_results[drone_code]
            scenario_definitions.append(
            (
                f"仅使用{drone_code}型",
                "限定单一机型 → 架次数 → 总能耗 → 累计架次作业时间",
                f"TYPE-{drone_code}",
                selected_rows(candidates, result.x) if result is not None else None,
            )
            )

    summaries: List[Dict[str, object]] = []
    scenario_plans: List[pd.DataFrame] = []
    coverage_audits: List[pd.DataFrame] = []
    for scenario_name, priority, scenario_code, plan in scenario_definitions:
        if plan is None:
            summaries.append(
                summarize_plan(
                    scenario_name,
                    pd.DataFrame(),
                    priority,
                    drone_volumes,
                    feasible=False,
                )
            )
            continue

        audit = check_exact_coverage(plan, group_box_ids, box_ids)
        audit.insert(0, "优化情景", scenario_name)
        coverage_audits.append(audit)

        summaries.append(
            summarize_plan(scenario_name, plan, priority, drone_volumes)
        )
        scenario_frame = plan.copy()
        scenario_frame.insert(0, "优化情景", scenario_name)
        scenario_frame.insert(1, "目标优先级", priority)
        scenario_frame["方案架次编号"] = [
            f"{scenario_code}-{index:03d}"
            for index in range(1, len(scenario_frame) + 1)
        ]
        scenario_plans.append(scenario_frame)

    if not coverage_audits:
        raise AssertionError("没有生成任何逐箱覆盖检查结果。")
    all_solutions = pd.concat(scenario_plans, ignore_index=True)
    all_coverage = pd.concat(coverage_audits, ignore_index=True)
    main_coverage = check_exact_coverage(main_plan, group_box_ids, box_ids)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    main_plan.to_csv(
        OUTPUT_DIR / "问题一第二问_字典序主方案.csv",
        index=False,
        encoding="utf-8-sig",
    )
    minimum_sortie_time_plan.to_csv(
        OUTPUT_DIR / "问题一第二问_最短时间对照方案.csv",
        index=False,
        encoding="utf-8-sig",
    )
    all_solutions.to_csv(
        OUTPUT_DIR / "问题一第二问_多情景方案.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(summaries).to_csv(
        OUTPUT_DIR / "问题一第二问_指标比较.csv",
        index=False,
        encoding="utf-8-sig",
    )
    main_coverage.to_csv(
        OUTPUT_DIR / "问题一第二问_逐箱覆盖检查.csv",
        index=False,
        encoding="utf-8-sig",
    )
    all_coverage.to_csv(
        OUTPUT_DIR / "问题一第二问_多情景逐箱覆盖检查.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(stage_log).to_csv(
        OUTPUT_DIR / "问题一第二问_求解阶段记录.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n求解完成。所有可行情景均要求求解器证明每个优先级阶段最优。")
    print(f"最少往返架次数：{minimum_sorties}")
    print(f"字典序主方案总能耗：{main_plan['往返能耗_kWh'].sum():.6f} kWh")
    print(
        "字典序主方案累计架次作业时间："
        f"{main_plan['估计架次作业时间_s'].sum():.2f} s"
    )
    print(
        "固定最少架次的时间优先对照："
        f"能耗 {minimum_sortie_time_plan['往返能耗_kWh'].sum():.6f} kWh，"
        f"累计架次作业时间 {minimum_sortie_time_plan['估计架次作业时间_s'].sum():.2f} s"
    )
    print(f"多情景指标及货舱利用率已写入：{OUTPUT_DIR / '问题一第二问_指标比较.csv'}")
    print(f"输出目录：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
