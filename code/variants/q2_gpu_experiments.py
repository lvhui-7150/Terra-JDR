"""问题二第一问：GPU候选预筛选 + 多种强化策略自动对比。

说明：
1. OR-Tools CP-SAT本身只能使用CPU求解；本程序通过主求解脚本的--device gpu
   启用CUDA候选组批评分，CP-SAT仍使用CPU多线程。
2. 每个实验使用独立输出目录，避免结果互相覆盖。
3. 默认依次尝试候选池扩张、多航序、深度LNS、三箱穷举和综合强化方案。
4. 三箱穷举和综合强化可能运行很久，建议先使用--smoke验证流程。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Dict, List

import pandas as pd


ROOT = Path(__file__).resolve().parent
SOLVER = ROOT / "问题二_第一问_CP-SAT强化精确求解.py"
OUTPUT_ROOT = ROOT / "问题二第一问_GPU多方案实验结果"


PROFILES: List[Dict[str, object]] = [
    {
        "name": "01_GPU候选评分基线",
        "description": "保持当前参数，仅启用GPU候选组批评分",
        "args": [
            "--rounds", "3", "--walk-steps", "12000",
            "--brute-pair-limit", "120", "--max-union-boxes", "12",
            "--max-route-groups", "1500", "--time-limit", "900",
            "--exhaustive-size", "2", "--max-orders-per-drone", "4",
        ],
    },
    {
        "name": "02_GPU候选池扩张",
        "description": "候选组批上限由1500扩张到3000",
        "args": [
            "--rounds", "3", "--walk-steps", "12000",
            "--brute-pair-limit", "180", "--max-union-boxes", "12",
            "--max-route-groups", "3000", "--time-limit", "900",
            "--exhaustive-size", "2", "--max-orders-per-drone", "4",
        ],
    },
    {
        "name": "03_GPU多航序扩展",
        "description": "候选池扩张并将每种机型保留航序增加到8种",
        "args": [
            "--rounds", "3", "--walk-steps", "16000",
            "--brute-pair-limit", "240", "--max-union-boxes", "12",
            "--max-route-groups", "3000", "--time-limit", "900",
            "--exhaustive-size", "2", "--max-orders-per-drone", "8",
        ],
    },
    {
        "name": "04_GPU深度LNS",
        "description": "增加轮数、随机大邻域步数和两架次重组规模",
        "args": [
            "--rounds", "5", "--walk-steps", "30000",
            "--brute-pair-limit", "300", "--max-union-boxes", "14",
            "--max-route-groups", "3000", "--time-limit", "1200",
            "--exhaustive-size", "2", "--max-orders-per-drone", "6",
        ],
    },
    {
        "name": "05_GPU三箱穷举",
        "description": "穷举所有不超过三箱的可行组批，并进行强化搜索",
        "args": [
            "--rounds", "2", "--walk-steps", "16000",
            "--brute-pair-limit", "240", "--max-union-boxes", "12",
            "--max-route-groups", "3000", "--time-limit", "900",
            "--exhaustive-size", "3", "--max-orders-per-drone", "6",
        ],
    },
    {
        "name": "06_GPU综合最强",
        "description": "扩大候选池、增加航序、深度LNS并延长CP-SAT搜索",
        "args": [
            "--rounds", "6", "--walk-steps", "50000",
            "--brute-pair-limit", "500", "--max-union-boxes", "14",
            "--max-route-groups", "5000", "--time-limit", "1800",
            "--exhaustive-size", "2", "--max-orders-per-drone", "8",
        ],
    },
    {
        "name": "07_GPU零迟到最低能耗",
        "description": "固定所有货箱期望时间内送达，再优先最小化总能耗",
        "args": [
            "--objective", "energy", "--zero-tardiness",
            "--rounds", "5", "--walk-steps", "30000",
            "--brute-pair-limit", "300", "--max-union-boxes", "14",
            "--max-route-groups", "3000", "--time-limit", "1200",
            "--exhaustive-size", "2", "--max-orders-per-drone", "6",
        ],
    },
    {
        "name": "08_GPU零迟到最少架次",
        "description": "固定所有货箱期望时间内送达，再优先最小化运输架次",
        "args": [
            "--objective", "sortie", "--zero-tardiness",
            "--rounds", "5", "--walk-steps", "30000",
            "--brute-pair-limit", "300", "--max-union-boxes", "14",
            "--max-route-groups", "3000", "--time-limit", "1200",
            "--exhaustive-size", "2", "--max-orders-per-drone", "6",
        ],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="依次运行问题二第一问的GPU候选评分和多种强化配置"
    )
    parser.add_argument(
        "--only",
        nargs="*",
        help="只运行指定实验名称，例如 02_GPU候选池扩张 04_GPU深度LNS",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="快速冒烟模式，仅验证所有配置能运行，不代表正式结果",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="候选评分设备；CP-SAT始终使用CPU",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="某个实验失败后立即停止",
    )
    return parser.parse_args()


def smoke_args(profile: Dict[str, object]) -> List[str]:
    profile_args = [str(value) for value in profile["args"]]
    objective_args: List[str] = []
    if "--objective" in profile_args:
        index = profile_args.index("--objective")
        objective_args = profile_args[index : index + 2]
    if "--zero-tardiness" in profile_args:
        objective_args.append("--zero-tardiness")
    return objective_args + [
        "--rounds", "1", "--walk-steps", "30",
        "--brute-pair-limit", "3", "--max-union-boxes", "8",
        "--max-route-groups", "300", "--time-limit", "5",
        "--exhaustive-size", "1", "--max-orders-per-drone", "3",
    ]


def read_metrics(output_dir: Path) -> Dict[str, object]:
    path = output_dir / "问题二第一问_CP-SAT强化方案指标.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if frame.empty:
        return {}
    return frame.iloc[0].to_dict()


def run_profile(profile: Dict[str, object], args: argparse.Namespace) -> Dict[str, object]:
    name = str(profile["name"])
    output_dir = OUTPUT_ROOT / name
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(SOLVER),
        *(["--device", args.device]),
        "--output-dir",
        str(output_dir),
    ]
    if args.smoke:
        command.extend(smoke_args(profile))
    else:
        command.extend(str(value) for value in profile["args"])

    print(f"\n========== {name} ==========")
    print(profile["description"])
    print("命令：" + " ".join(command))
    started = perf_counter()
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    elapsed = perf_counter() - started
    log_path = output_dir / "运行日志.txt"
    log_path.write_text(completed.stdout, encoding="utf-8")
    print(completed.stdout[-4000:])

    metrics = read_metrics(output_dir)
    record: Dict[str, object] = {
        "实验名称": name,
        "实验说明": profile["description"],
        "退出码": completed.returncode,
        "运行时间_s": elapsed,
        "输出目录": str(output_dir),
        "状态": "成功" if completed.returncode == 0 else "失败",
    }
    record.update(metrics)
    return record


def main() -> None:
    args = parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    selected = PROFILES
    if args.only:
        wanted = set(args.only)
        selected = [profile for profile in PROFILES if profile["name"] in wanted]
        missing = wanted - {str(profile["name"]) for profile in selected}
        if missing:
            raise ValueError(f"未找到实验名称：{sorted(missing)}")

    all_records: List[Dict[str, object]] = []
    for profile in selected:
        record = run_profile(profile, args)
        all_records.append(record)
        if args.stop_on_error and record["退出码"] != 0:
            break

    summary = pd.DataFrame(all_records)
    summary_path = OUTPUT_ROOT / "问题二第一问_GPU多方案指标汇总.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    (OUTPUT_ROOT / "实验配置.json").write_text(
        json.dumps(
            {
                "求解脚本": str(SOLVER),
                "候选评分设备": args.device,
                "CP-SAT说明": "OR-Tools CP-SAT使用CPU多线程，GPU仅用于候选组批评分预筛选",
                "实验数量": len(selected),
                "冒烟模式": args.smoke,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\n========== 全部实验汇总 ==========")
    columns = [
        column
        for column in [
            "实验名称",
            "状态",
            "加权迟到_优先系数乘秒",
            "全部任务完成时间_s",
            "总运输能耗_kWh",
            "运输架次数",
            "运行时间_s",
        ]
        if column in summary.columns
    ]
    if not summary.empty:
        print(summary[columns].to_string(index=False))
    print(f"汇总文件：{summary_path}")


if __name__ == "__main__":
    main()
