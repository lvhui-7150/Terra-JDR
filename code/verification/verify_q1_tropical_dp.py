from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "问题一第一问结果" / "可行候选组批.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "问题一_热带半环DP"
BOX_FILE = ROOT / "数据" / "无人机应急物资运输基础数据" / "物资需求与配送时限.xlsx"


def solve_service(service_id: str, rows: pd.DataFrame) -> list[dict]:
    box_ids = sorted({box for field in rows["货箱编号列表"] for box in str(field).split("、")})
    box_index = {box: index for index, box in enumerate(box_ids)}
    all_mask = (1 << len(box_ids)) - 1
    candidates: list[tuple[int, dict]] = []
    by_bit: list[list[int]] = [[] for _ in box_ids]
    for _, source in rows.iterrows():
        if str(source["服务区编号"]) != service_id:
            raise ValueError(f"候选行服务区与分组不符：{source['服务区编号']} != {service_id}")
        selected_boxes = str(source["货箱编号列表"]).split("、")
        if len(selected_boxes) != len(set(selected_boxes)):
            raise ValueError(f"候选组批内部有重复货箱：{selected_boxes}")
        if any(box not in box_index for box in selected_boxes):
            raise ValueError(f"候选组批含不属于本服务区的货箱：{selected_boxes}")
        mask = sum(1 << box_index[box] for box in selected_boxes)
        record = source.to_dict()
        candidates.append((mask, record))
        for bit in range(len(box_ids)):
            if mask & (1 << bit):
                by_bit[bit].append(len(candidates) - 1)

    @lru_cache(maxsize=None)
    def best(covered: int) -> tuple[float, float, float, int | None, int | None]:
        if covered == all_mask:
            return 0.0, 0.0, 0.0, None, None
        first = next(i for i in range(len(box_ids)) if not covered & (1 << i))
        optimum = (float("inf"), float("inf"), float("inf"), None, None)
        for candidate_index in by_bit[first]:
            mask, row = candidates[candidate_index]
            if mask & covered:
                continue
            tail = best(covered | mask)
            score = (tail[0] + 1.0, tail[1] + float(row["往返能耗_kWh"]),
                     tail[2] + float(row["估计架次作业时间_s"]))
            if score < optimum[:3]:
                optimum = (*score, covered | mask, candidate_index)
        if optimum[3] is None:
            raise ValueError(f"{service_id} 无法覆盖全部货箱；不可行状态={covered:#x}")
        return optimum

    best(0)
    selected = []
    covered = 0
    while covered != all_mask:
        candidate_index = best(covered)[4]
        if candidate_index is None:
            raise AssertionError("DP回溯记录损坏")
        mask, row = candidates[candidate_index]
        selected.append(row)
        covered |= mask
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="服务区独立的字典序 min-plus 精确覆盖动态规划")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    candidates = pd.read_csv(args.input, encoding="utf-8-sig")
    required = {"服务区编号", "货箱编号列表", "往返能耗_kWh", "估计架次作业时间_s"}
    if not required.issubset(candidates.columns):
        raise ValueError(f"候选表缺少字段：{sorted(required - set(candidates.columns))}")
    plan = []
    for service, group in candidates.groupby("服务区编号", sort=True):
        chosen = solve_service(str(service), group)
        plan.extend(chosen)
        print(f"{service}: {len(chosen)}架次，能耗={sum(float(r['往返能耗_kWh']) for r in chosen):.9f} kWh")
    result = pd.DataFrame(plan)
    delivered = [box for field in result["货箱编号列表"] for box in str(field).split("、")]
    if not BOX_FILE.is_file():
        raise FileNotFoundError(f"找不到题目原始货箱清单：{BOX_FILE}")
    source_boxes = pd.read_excel(BOX_FILE, sheet_name="逐箱货箱清单")
    if "货箱编号" not in source_boxes:
        raise ValueError("题目原始货箱清单缺少“货箱编号”列")
    expected = set(source_boxes["货箱编号"].dropna().astype(str))
    if len(expected) != 80:
        raise ValueError(f"题目原始清单应有80个唯一货箱，实际为{len(expected)}")
    if len(delivered) != len(set(delivered)) or set(delivered) != expected:
        raise AssertionError("DP结果未实现货箱无重复、无遗漏覆盖")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "DP字典序最优组批.csv", index=False, encoding="utf-8-sig")
    summary = {"候选架次数": len(candidates), "服务区数": candidates["服务区编号"].nunique(),
               "覆盖货箱数": len(delivered), "架次数": len(result),
               "总能耗_kWh": float(result["往返能耗_kWh"].sum()),
               "累计作业时间_s": float(result["估计架次作业时间_s"].sum()), "覆盖检查": "PASS"}
    pd.DataFrame([summary]).to_csv(args.output_dir / "DP汇总.csv", index=False, encoding="utf-8-sig")
    print("DP结果：" + "；".join(f"{key}={value}" for key, value in summary.items()))
    print(f"输出目录：{args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
