"""以可变运输架次联合优化通信覆盖和两架中继无人机调度。"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "新算法复核" / "问题三_可变架次逐任务资源联合求解.py"
DEFAULTS = [
    "--seed", "20260924",
    "--alns-steps", "1200",
    "--alns-segment", "100",
    "--pair-limit", "800",
    "--max-route-groups", "300",
    "--max-route-candidates", "400",
    "--singleton-options-per-box", "3",
    "--mandatory-options-per-group", "3",
    "--max-orders-per-drone", "8",
    "--seed-route-file", str(ROOT / "_问题三独立模型正式可行结果_修复输出" / "问题三独立模型_运输架次.csv"),
    "--sample-step", "0.5",
    "--los-step", "30",
    "--heights", "100", "200", "300",
    "--relay-grid", "500",
    "--max-relay-base-points", "240",
    "--dynamic-points-per-gap", "10",
    "--relay-test-limit", "100",
    "--max-relay-options", "8",
    "--max-relay-block", "3600",
    "--relay-drone-count", "2",
    "--relay-component-count", "6",
    "--horizon", "40000",
    "--time-limit", "300",
    "--feasibility-time", "180",
    "--seed-time-limit", "120",
    "--seed-feasibility-time", "120",
    "--cover-time-limit", "120",
    "--transport-seed-time-limit", "180",
    "--transport-seed-stages", "3",
    "--seed-iterations", "20",
    "--master-time-limit", "60",
    "--max-stages", "4",
    "--workers", "8",
    "--objective", "timeliness",
    "--output-dir", str(ROOT / "问题三_全量高分辨率联合结果"),
]


def main() -> None:
    spec = importlib.util.spec_from_file_location("q3_joint_core", CORE)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载联合求解主程序：{CORE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    sys.argv = [str(CORE), *DEFAULTS, *sys.argv[1:]]
    module.main()


if __name__ == "__main__":
    main()
