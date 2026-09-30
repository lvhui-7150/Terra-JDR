"""问题一第三小问：返航安全余量敏感性分析。

对返航SOC下限10%、12%、……、30%（步长2%）逐一执行完整计算链：
1. 重新计算最大安全载荷；
2. 重新枚举可行候选组批；
3. 重新求解架次数、能耗、累计作业时间字典序优化；
4. 汇总安全性与运输效率变化。
"""
from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "问题一第三问结果"
RESERVE_LEVELS = [percent / 100.0 for percent in range(10, 31, 2)]
EXPECTED_BOX_COUNT = 80
MIP_GAP_TOLERANCE = 1e-9


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_scenario(reserve_ratio: float) -> dict:
    label = f"SOC_{int(round(100 * reserve_ratio)):02d}pct"
    scenario_dir = OUTPUT_DIR / label
    first_dir = scenario_dir / "第一问重算结果"
    second_dir = scenario_dir / "第二问优化结果"
    scenario_dir.mkdir(parents=True, exist_ok=True)

    q1 = load_module(f"q1_first_{label}", ROOT / "问题一_第一问.py")
    original_read_drone_types = q1.read_drone_types

    def read_drone_types_with_scenario():
        return [
            replace(drone, reserve_ratio=reserve_ratio)
            for drone in original_read_drone_types()
        ]

    q1.read_drone_types = read_drone_types_with_scenario
    q1.OUTPUT_DIR = first_dir
    q1.SUBMISSION_OUTPUT = scenario_dir / "结果提交_问题一第一问.xlsx"
    print(f"\n[{label}] 重新计算安全载荷和候选组批……")
    q1.main()

    q2 = load_module(f"q1_second_{label}", ROOT / "问题一_第二问.py")
    q2.FIRST_RESULT_DIR = first_dir
    q2.OUTPUT_DIR = second_dir
    q2.CANDIDATE_FILE = first_dir / "可行候选组批.csv"
    q2.FIRST_PLAN_FILE = first_dir / "第一问可行方案.csv"
    q2.SAFE_PAYLOAD_FILE = first_dir / "最大安全载荷.csv"
    q2.RESERVE_RATIO_OVERRIDE = reserve_ratio
    q2.RUN_EXTENDED_SCENARIOS = False
    q2.RUN_TIME_COMPARISON = False
    print(f"[{label}] 重新求解字典序优化方案……")
    q2.main()

    return read_scenario_result(reserve_ratio)


def read_scenario_result(reserve_ratio: float) -> dict:
    """读取并复核一个已完成情景的核心结果。"""

    label = f"SOC_{int(round(100 * reserve_ratio)):02d}pct"
    scenario_dir = OUTPUT_DIR / label
    first_dir = scenario_dir / "第一问重算结果"
    second_dir = scenario_dir / "第二问优化结果"

    metrics = pd.read_csv(
        second_dir / "问题一第二问_指标比较.csv", encoding="utf-8-sig"
    )
    main_rows = metrics.loc[
        metrics["方案"].isin(["架次优先主方案", "第二小问字典序主方案"])
    ]
    if len(main_rows) != 1:
        raise ValueError(f"{label} 无法唯一识别字典序主方案。")
    main_row = main_rows.iloc[0]
    candidates = pd.read_csv(first_dir / "可行候选组批.csv", encoding="utf-8-sig")
    safe_payload = pd.read_csv(first_dir / "最大安全载荷.csv", encoding="utf-8-sig")
    main_plan = pd.read_csv(
        second_dir / "问题一第二问_字典序主方案.csv", encoding="utf-8-sig"
    )
    stage_log = pd.read_csv(
        second_dir / "问题一第二问_求解阶段记录.csv", encoding="utf-8-sig"
    )
    coverage = pd.read_csv(
        second_dir / "问题一第二问_逐箱覆盖检查.csv", encoding="utf-8-sig"
    )

    payload_columns = [c for c in safe_payload.columns if c.endswith("最大安全载荷_kg")]
    payload_values = safe_payload[payload_columns].apply(pd.to_numeric, errors="coerce")

    if (main_plan["返航SOC_pct"] + 1e-7 < 100.0 * reserve_ratio).any():
        raise AssertionError(f"{label} 的最优方案存在返航SOC低于情景下限的架次。")

    mip_gaps = pd.to_numeric(stage_log["MIP_gap"], errors="coerce")
    solver_codes = pd.to_numeric(stage_log["求解器状态码"], errors="coerce")
    if (
        not stage_log["状态"].eq("已证明最优").all()
        or not solver_codes.eq(0).all()
        or mip_gaps.isna().any()
        or mip_gaps.abs().max() > MIP_GAP_TOLERANCE
    ):
        raise AssertionError(f"{label} 存在未通过最优性校验的求解阶段。")

    coverage_count = pd.to_numeric(coverage["方案中出现次数"], errors="coerce")
    if len(coverage) != EXPECTED_BOX_COUNT or not coverage_count.eq(1).all():
        raise AssertionError(f"{label} 未通过{EXPECTED_BOX_COUNT}箱精确覆盖校验。")

    return {
        "返航安全余量_pct": 100.0 * reserve_ratio,
        "是否存在完整最优方案": True,
        "可行候选组批数量": len(candidates),
        "最少往返架次数": int(main_row["架次数"]),
        "总运输能耗_kWh": float(main_row["总运输能耗_kWh"]),
        "累计架次作业时间_s": float(main_row["累计架次作业时间_s"]),
        "最优方案最低返航SOC_pct": float(main_row["最小返航SOC_pct"]),
        "最优方案SOC冗余_pct": float(main_row["最小返航SOC_pct"])
        - 100.0 * reserve_ratio,
        "最大MIP_gap": float(mip_gaps.abs().max()),
        "安全载荷均值_kg": float(payload_values.stack().mean()),
        "安全载荷最小值_kg": float(payload_values.stack().min()),
        "安全载荷最大值_kg": float(payload_values.stack().max()),
        "结果目录": str(scenario_dir),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for reserve_ratio in RESERVE_LEVELS:
        try:
            label = f"SOC_{int(round(100 * reserve_ratio)):02d}pct"
            scenario_dir = OUTPUT_DIR / label
            required_files = [
                scenario_dir / "第一问重算结果" / "可行候选组批.csv",
                scenario_dir / "第一问重算结果" / "最大安全载荷.csv",
                scenario_dir / "第二问优化结果" / "问题一第二问_指标比较.csv",
                scenario_dir / "第二问优化结果" / "问题一第二问_字典序主方案.csv",
                scenario_dir / "第二问优化结果" / "问题一第二问_求解阶段记录.csv",
                scenario_dir / "第二问优化结果" / "问题一第二问_逐箱覆盖检查.csv",
            ]
            if all(path.exists() for path in required_files):
                print(f"\n[{label}] 复用现有已完成结果。")
                rows.append(read_scenario_result(reserve_ratio))
            else:
                rows.append(run_scenario(reserve_ratio))
        except Exception as exc:
            rows.append(
                {
                    "返航安全余量_pct": 100.0 * reserve_ratio,
                    "是否存在完整最优方案": False,
                    "失败原因": str(exc),
                }
            )
            raise

    summary = pd.DataFrame(rows).sort_values("返航安全余量_pct")
    summary.to_csv(
        OUTPUT_DIR / "问题一第三问_安全余量敏感性汇总.csv",
        index=False,
        encoding="utf-8-sig",
    )

    baseline = summary.loc[summary["返航安全余量_pct"] == 20.0].iloc[0]
    comparison = summary.copy()
    comparison["相对20pct架次数变化"] = (
        comparison["最少往返架次数"] - baseline["最少往返架次数"]
    )
    comparison["相对20pct能耗变化_pct"] = 100.0 * (
        comparison["总运输能耗_kWh"] / baseline["总运输能耗_kWh"] - 1.0
    )
    comparison["相对20pct时间变化_pct"] = 100.0 * (
        comparison["累计架次作业时间_s"] / baseline["累计架次作业时间_s"] - 1.0
    )
    comparison["相对20pct候选组批变化_pct"] = 100.0 * (
        comparison["可行候选组批数量"] / baseline["可行候选组批数量"] - 1.0
    )
    comparison.to_csv(
        OUTPUT_DIR / "问题一第三问_相对20pct基准比较.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n第三小问敏感性分析完成：")
    print(summary.to_string(index=False))
    print(f"\n输出目录：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()

