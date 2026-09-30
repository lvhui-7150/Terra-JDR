"""问题二第一小问：五种元启发式算法对比。

统一调用“问题二_第一问.py”中的数据读取、DEM航段、逐段剩余载荷能耗、
实体无人机与共享电池调度、目标函数和约束检查，仅替换上层搜索算法。

对比算法：
1. 模拟退火（SA）；
2. 禁忌搜索（TS）；
3. 遗传算法（GA）；
4. 离散灰狼优化（DGWO）；
5. 离散粒子群优化（DPSO）。

五种算法使用相同初始解池、相同目标场景和相同物理约束。灰狼与粒子群
采用适用于离散货箱分组的“向领导解迁移”算子，而不是直接套用连续位置公式。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import importlib.util
import math
from pathlib import Path
import random
import sys
from time import perf_counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parent
BASE_SCRIPT = ROOT / "问题二_第一问.py"
BASE_RESULT_DIR = ROOT / "问题二第一问结果"
OUTPUT_DIR = ROOT / "问题二第一问五算法对比结果"

TARGET_SCENARIO = "及时性优先"
RANDOM_SEED = 20260923

SA_ITERATIONS = 600
TS_ITERATIONS = 350
TS_NEIGHBOR_COUNT = 24
TS_TABU_TENURE = 45
GA_GENERATIONS = 180
GA_POPULATION_SIZE = 20
GA_MUTATION_RATE = 0.80
GWO_ITERATIONS = 260
GWO_POPULATION_SIZE = 16
PSO_ITERATIONS = 260
PSO_POPULATION_SIZE = 16


Partition = Tuple[Tuple[str, ...], ...]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_module("q2_first_base_for_metaheuristics", BASE_SCRIPT)


@dataclass
class AlgorithmOutput:
    algorithm: str
    result: object
    runtime_s: float
    evaluations: int
    trace: List[dict]


class Evaluator:
    def __init__(self, scenario: str):
        self.scenario = scenario
        self.cache: Dict[Partition, object] = {}

    def evaluate(self, partition: Partition):
        if partition not in self.cache:
            self.cache[partition] = BASE.schedule_partition(partition, self.scenario)
        return self.cache[partition]

    @staticmethod
    def key(result) -> Tuple[float, ...]:
        return BASE.objective_key(result)

    @staticmethod
    def scalar(result) -> float:
        return BASE.objective_scalar(result)


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


def load_existing_main_partition() -> Optional[Partition]:
    path = BASE_RESULT_DIR / "问题二第一问_主方案运输架次.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if "货箱编号列表" not in frame.columns:
        return None
    partition = BASE.normalize_partition(
        str(value).split("、") for value in frame["货箱编号列表"]
    )
    return partition if BASE.partition_is_physically_feasible(partition) else None


def initial_partition_pool() -> List[Partition]:
    partitions = [
        BASE.q1_initial_partition(),
        BASE.greedy_partition("deadline"),
        BASE.greedy_partition("mass"),
        BASE.greedy_partition("service"),
    ]
    existing = load_existing_main_partition()
    if existing is not None:
        partitions.append(existing)
    return list(dict.fromkeys(partitions))


def best_of(partitions: Sequence[Partition], evaluator: Evaluator):
    return min((evaluator.evaluate(partition) for partition in partitions), key=evaluator.key)


def trace_row(iteration: int, result) -> dict:
    return {
        "迭代": iteration,
        "硬时限违约箱数": result.hard_violation_count,
        "加权迟到": result.weighted_tardiness,
        "全部任务完成时间_s": result.makespan_s,
        "总运输能耗_kWh": result.total_energy_kwh,
        "运输架次数": result.sortie_count,
    }


def route_map(partition: Partition) -> Dict[str, frozenset[str]]:
    mapping: Dict[str, frozenset[str]] = {}
    for route in partition:
        route_set = frozenset(route)
        for box_id in route:
            mapping[box_id] = route_set
    return mapping


def guided_move(
    current: Partition,
    guide: Partition,
    rng: random.Random,
) -> Partition:
    current_map = route_map(current)
    guide_map = route_map(guide)
    different = [
        box_id
        for box_id in BASE.BOX_BY_ID
        if current_map[box_id] != guide_map[box_id]
    ]
    if not different:
        candidate = BASE.mutate_partition(current, rng)
        return current if candidate is None else candidate

    box_id = rng.choice(different)
    target_companions = set(guide_map[box_id]) - {box_id}
    routes = [list(route) for route in current]
    source_index = next(index for index, route in enumerate(routes) if box_id in route)
    routes[source_index].remove(box_id)

    destination_order = sorted(
        range(len(routes)),
        key=lambda index: len(set(routes[index]) & target_companions),
        reverse=True,
    )
    for destination in destination_order:
        if destination == source_index:
            continue
        proposed = tuple(sorted(routes[destination] + [box_id]))
        if BASE.feasible_route_options(proposed):
            routes[destination].append(box_id)
            routes = [route for route in routes if route]
            return BASE.normalize_partition(routes)

    routes = [route for route in routes if route]
    routes.append([box_id])
    candidate = BASE.normalize_partition(routes)
    if BASE.partition_is_physically_feasible(candidate):
        return candidate
    fallback = BASE.mutate_partition(current, rng)
    return current if fallback is None else fallback


def repair_partition(routes: Iterable[Iterable[str]], rng: random.Random) -> Partition:
    accepted: List[List[str]] = []
    used: set[str] = set()
    for route in routes:
        unique = [box_id for box_id in route if box_id not in used]
        if not unique:
            continue
        key = tuple(sorted(unique))
        if BASE.feasible_route_options(key):
            accepted.append(list(key))
            used.update(key)

    missing = [box_id for box_id in BASE.BOX_BY_ID if box_id not in used]
    rng.shuffle(missing)
    for box_id in missing:
        choices: List[Tuple[Tuple[float, ...], Optional[int], Tuple[str, ...]]] = []
        for index, route in enumerate(accepted):
            candidate = tuple(sorted(route + [box_id]))
            options = BASE.feasible_route_options(candidate)
            if options:
                option = min(options, key=lambda item: (item.energy_kwh, item.duration_s))
                choices.append(((0.0, option.energy_kwh, option.duration_s), index, candidate))
        single = (box_id,)
        options = BASE.feasible_route_options(single)
        if options:
            option = min(options, key=lambda item: (item.energy_kwh, item.duration_s))
            choices.append(((1.0, option.energy_kwh, option.duration_s), None, single))
        if not choices:
            raise RuntimeError(f"修复过程中货箱{box_id}无法加入任何架次。")
        _, index, selected = min(choices, key=lambda item: item[0])
        if index is None:
            accepted.append(list(selected))
        else:
            accepted[index] = list(selected)

    partition = BASE.normalize_partition(accepted)
    if not BASE.partition_is_physically_feasible(partition):
        raise AssertionError("交叉或修复后的分区未通过物理可行性检查。")
    return partition


def crossover(parent_a: Partition, parent_b: Partition, rng: random.Random) -> Partition:
    selected = [list(route) for route in parent_a if rng.random() < 0.45]
    used = {box_id for route in selected for box_id in route}
    for route in parent_b:
        remaining = [box_id for box_id in route if box_id not in used]
        if remaining and BASE.feasible_route_options(tuple(sorted(remaining))):
            selected.append(remaining)
            used.update(remaining)
    return repair_partition(selected, rng)


def expanded_population(
    seeds: Sequence[Partition],
    population_size: int,
    rng: random.Random,
) -> List[Partition]:
    population = list(dict.fromkeys(seeds))
    while len(population) < population_size:
        parent = rng.choice(population)
        child = BASE.mutate_partition(parent, rng)
        if child is not None and child not in population:
            population.append(child)
    return population[:population_size]


# ==============================
# 1. 模拟退火
# ==============================


def run_simulated_annealing(
    seeds: Sequence[Partition], evaluator: Evaluator, seed: int
) -> AlgorithmOutput:
    rng = random.Random(seed)
    current = best_of(seeds, evaluator)
    best = current
    trace = [trace_row(0, best)]
    started = perf_counter()
    initial_temperature = max(1.0, evaluator.scalar(current) * 0.003)

    for iteration in range(1, SA_ITERATIONS + 1):
        candidate_partition = BASE.mutate_partition(current.partition, rng)
        if candidate_partition is None:
            continue
        candidate = evaluator.evaluate(candidate_partition)
        delta = evaluator.scalar(candidate) - evaluator.scalar(current)
        temperature = max(
            1e-8,
            initial_temperature * (1.0 - iteration / (SA_ITERATIONS + 1)) ** 2,
        )
        if delta <= 0.0 or rng.random() < math.exp(-min(delta / temperature, 700.0)):
            current = candidate
        if evaluator.key(candidate) < evaluator.key(best):
            best = candidate
        trace.append(trace_row(iteration, best))

    return AlgorithmOutput(
        algorithm="模拟退火SA",
        result=best,
        runtime_s=perf_counter() - started,
        evaluations=len(evaluator.cache),
        trace=trace,
    )


# ==============================
# 2. 禁忌搜索
# ==============================


def run_tabu_search(
    seeds: Sequence[Partition], evaluator: Evaluator, seed: int
) -> AlgorithmOutput:
    rng = random.Random(seed)
    current = best_of(seeds, evaluator)
    best = current
    tabu = deque([current.partition], maxlen=TS_TABU_TENURE)
    trace = [trace_row(0, best)]
    started = perf_counter()

    for iteration in range(1, TS_ITERATIONS + 1):
        neighborhood: List[object] = []
        seen: set[Partition] = set()
        for _ in range(TS_NEIGHBOR_COUNT * 3):
            candidate_partition = BASE.mutate_partition(current.partition, rng)
            if candidate_partition is None or candidate_partition in seen:
                continue
            seen.add(candidate_partition)
            candidate = evaluator.evaluate(candidate_partition)
            aspiration = evaluator.key(candidate) < evaluator.key(best)
            if candidate_partition not in tabu or aspiration:
                neighborhood.append(candidate)
            if len(neighborhood) >= TS_NEIGHBOR_COUNT:
                break
        if not neighborhood:
            current = evaluator.evaluate(rng.choice(seeds))
            tabu.append(current.partition)
            trace.append(trace_row(iteration, best))
            continue

        current = min(neighborhood, key=evaluator.key)
        tabu.append(current.partition)
        if evaluator.key(current) < evaluator.key(best):
            best = current
        trace.append(trace_row(iteration, best))

    return AlgorithmOutput(
        algorithm="禁忌搜索TS",
        result=best,
        runtime_s=perf_counter() - started,
        evaluations=len(evaluator.cache),
        trace=trace,
    )


# ==============================
# 3. 遗传算法
# ==============================


def tournament_select(population: Sequence[Partition], evaluator: Evaluator, rng: random.Random) -> Partition:
    competitors = rng.sample(list(population), k=min(3, len(population)))
    return min(competitors, key=lambda partition: evaluator.key(evaluator.evaluate(partition)))


def run_genetic_algorithm(
    seeds: Sequence[Partition], evaluator: Evaluator, seed: int
) -> AlgorithmOutput:
    rng = random.Random(seed)
    population = expanded_population(seeds, GA_POPULATION_SIZE, rng)
    best = best_of(population, evaluator)
    trace = [trace_row(0, best)]
    started = perf_counter()

    for generation in range(1, GA_GENERATIONS + 1):
        ranked = sorted(
            population,
            key=lambda partition: evaluator.key(evaluator.evaluate(partition)),
        )
        next_population = ranked[:2]
        attempts = 0
        while len(next_population) < GA_POPULATION_SIZE and attempts < GA_POPULATION_SIZE * 20:
            attempts += 1
            parent_a = tournament_select(population, evaluator, rng)
            parent_b = tournament_select(population, evaluator, rng)
            child = crossover(parent_a, parent_b, rng)
            if rng.random() < GA_MUTATION_RATE:
                mutated = BASE.mutate_partition(child, rng)
                if mutated is not None:
                    child = mutated
            if child not in next_population:
                next_population.append(child)
        while len(next_population) < GA_POPULATION_SIZE:
            parent = rng.choice(ranked[: max(2, len(ranked) // 2)])
            child = BASE.mutate_partition(parent, rng)
            if child is not None and child not in next_population:
                next_population.append(child)
        population = next_population
        generation_best = best_of(population, evaluator)
        if evaluator.key(generation_best) < evaluator.key(best):
            best = generation_best
        trace.append(trace_row(generation, best))

    return AlgorithmOutput(
        algorithm="遗传算法GA",
        result=best,
        runtime_s=perf_counter() - started,
        evaluations=len(evaluator.cache),
        trace=trace,
    )


# ==============================
# 4. 离散灰狼优化
# ==============================


def run_discrete_grey_wolf(
    seeds: Sequence[Partition], evaluator: Evaluator, seed: int
) -> AlgorithmOutput:
    rng = random.Random(seed)
    wolves = expanded_population(seeds, GWO_POPULATION_SIZE, rng)
    best = best_of(wolves, evaluator)
    trace = [trace_row(0, best)]
    started = perf_counter()

    for iteration in range(1, GWO_ITERATIONS + 1):
        ranked = sorted(
            wolves,
            key=lambda partition: evaluator.key(evaluator.evaluate(partition)),
        )
        alpha = ranked[0]
        beta = ranked[min(1, len(ranked) - 1)]
        delta = ranked[min(2, len(ranked) - 1)]
        leaders = [alpha, beta, delta]
        parameter_a = 2.0 * (1.0 - iteration / GWO_ITERATIONS)
        new_wolves = [alpha, beta, delta]

        for wolf in ranked[3:]:
            candidate = wolf
            guide_steps = 1 + int(2.0 - parameter_a / 2.0)
            for _ in range(guide_steps):
                guide = rng.choices(leaders, weights=[0.50, 0.30, 0.20], k=1)[0]
                candidate = guided_move(candidate, guide, rng)
            if rng.random() < parameter_a / 2.0:
                exploratory = BASE.mutate_partition(candidate, rng)
                if exploratory is not None:
                    candidate = exploratory

            current_result = evaluator.evaluate(wolf)
            candidate_result = evaluator.evaluate(candidate)
            if (
                evaluator.key(candidate_result) < evaluator.key(current_result)
                or rng.random() < 0.08 * parameter_a
            ):
                new_wolves.append(candidate)
            else:
                new_wolves.append(wolf)

        wolves = list(dict.fromkeys(new_wolves))
        wolves = expanded_population(wolves, GWO_POPULATION_SIZE, rng)
        iteration_best = best_of(wolves, evaluator)
        if evaluator.key(iteration_best) < evaluator.key(best):
            best = iteration_best
        trace.append(trace_row(iteration, best))

    return AlgorithmOutput(
        algorithm="离散灰狼DGWO",
        result=best,
        runtime_s=perf_counter() - started,
        evaluations=len(evaluator.cache),
        trace=trace,
    )


# ==============================
# 5. 离散粒子群优化
# ==============================


def run_discrete_particle_swarm(
    seeds: Sequence[Partition], evaluator: Evaluator, seed: int
) -> AlgorithmOutput:
    rng = random.Random(seed)
    particles = expanded_population(seeds, PSO_POPULATION_SIZE, rng)
    personal_best = list(particles)
    global_best_result = best_of(personal_best, evaluator)
    global_best = global_best_result.partition
    trace = [trace_row(0, global_best_result)]
    started = perf_counter()

    for iteration in range(1, PSO_ITERATIONS + 1):
        inertia = 0.85 - 0.50 * iteration / PSO_ITERATIONS
        next_particles: List[Partition] = []
        for index, particle in enumerate(particles):
            candidate = particle
            if rng.random() < inertia:
                mutated = BASE.mutate_partition(candidate, rng)
                if mutated is not None:
                    candidate = mutated
            if rng.random() < 0.70:
                candidate = guided_move(candidate, personal_best[index], rng)
            if rng.random() < 0.85:
                candidate = guided_move(candidate, global_best, rng)
            if rng.random() < 0.35:
                candidate = guided_move(candidate, global_best, rng)

            candidate_result = evaluator.evaluate(candidate)
            personal_result = evaluator.evaluate(personal_best[index])
            if evaluator.key(candidate_result) < evaluator.key(personal_result):
                personal_best[index] = candidate
            next_particles.append(candidate)

        particles = next_particles
        personal_best_result = best_of(personal_best, evaluator)
        if evaluator.key(personal_best_result) < evaluator.key(global_best_result):
            global_best_result = personal_best_result
            global_best = personal_best_result.partition
        trace.append(trace_row(iteration, global_best_result))

    return AlgorithmOutput(
        algorithm="离散粒子群DPSO",
        result=global_best_result,
        runtime_s=perf_counter() - started,
        evaluations=len(evaluator.cache),
        trace=trace,
    )


# ==============================
# 结果输出
# ==============================


def output_summary(output: AlgorithmOutput) -> dict:
    result = output.result
    return {
        "算法": output.algorithm,
        "目标场景": TARGET_SCENARIO,
        "硬时限违约箱数": result.hard_violation_count,
        "硬时限总迟到_s": result.hard_lateness_s,
        "加权迟到_优先系数乘秒": result.weighted_tardiness,
        "全部任务完成时间_s": result.makespan_s,
        "总运输能耗_kWh": result.total_energy_kwh,
        "运输架次数": result.sortie_count,
        "期望时间内送达箱数": result.desired_on_time_count,
        "运行时间_s": output.runtime_s,
        "已评估不同分区数": output.evaluations,
    }


def write_outputs(outputs: Sequence[AlgorithmOutput]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([output_summary(output) for output in outputs]).to_csv(
        OUTPUT_DIR / "问题二第一问_五算法指标比较.csv",
        index=False,
        encoding="utf-8-sig",
    )

    sortie_frames = []
    delivery_frames = []
    trace_frames = []
    for output in outputs:
        sorties = pd.DataFrame(output.result.sorties)
        sorties.insert(0, "算法", output.algorithm)
        deliveries = pd.DataFrame(output.result.deliveries)
        deliveries.insert(0, "算法", output.algorithm)
        trace = pd.DataFrame(output.trace)
        trace.insert(0, "算法", output.algorithm)
        sortie_frames.append(sorties)
        delivery_frames.append(deliveries)
        trace_frames.append(trace)

    pd.concat(sortie_frames, ignore_index=True).to_csv(
        OUTPUT_DIR / "问题二第一问_五算法运输架次.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(delivery_frames, ignore_index=True).to_csv(
        OUTPUT_DIR / "问题二第一问_五算法逐箱交付.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(trace_frames, ignore_index=True).to_csv(
        OUTPUT_DIR / "问题二第一问_五算法收敛记录.csv",
        index=False,
        encoding="utf-8-sig",
    )

    best_output = min(outputs, key=lambda output: BASE.objective_key(output.result))
    pd.DataFrame(best_output.result.sorties).to_csv(
        OUTPUT_DIR / "问题二第一问_五算法最佳方案运输架次.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(best_output.result.deliveries).to_csv(
        OUTPUT_DIR / "问题二第一问_五算法最佳方案逐箱交付.csv",
        index=False,
        encoding="utf-8-sig",
    )
    BASE.validate_result(best_output.result).to_csv(
        OUTPUT_DIR / "问题二第一问_五算法最佳方案检查.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    print("初始化问题二数据、DEM航段和资源参数……")
    initialize_problem()
    seeds = initial_partition_pool()

    seed_evaluator = Evaluator(TARGET_SCENARIO)
    seed_best = best_of(seeds, seed_evaluator)
    if seed_best.hard_violation_count != 0:
        print("公共初始解池尚无零硬违约方案，调用基础搜索生成共同可行种子……")
        recovered = BASE.optimize_scenario(
            TARGET_SCENARIO,
            seeds,
            RANDOM_SEED + 999,
        )
        BASE.validate_result(recovered)
        seeds.append(recovered.partition)
        seeds = list(dict.fromkeys(seeds))

    algorithms = [
        ("模拟退火SA", run_simulated_annealing),
        ("禁忌搜索TS", run_tabu_search),
        ("遗传算法GA", run_genetic_algorithm),
        ("离散灰狼DGWO", run_discrete_grey_wolf),
        ("离散粒子群DPSO", run_discrete_particle_swarm),
    ]
    outputs: List[AlgorithmOutput] = []
    for index, (name, algorithm) in enumerate(algorithms, start=1):
        print(f"\n[{index}/{len(algorithms)}] 运行{name}……")
        evaluator = Evaluator(TARGET_SCENARIO)
        output = algorithm(seeds, evaluator, RANDOM_SEED + index * 10_000)
        BASE.validate_result(output.result)
        outputs.append(output)
        summary = output_summary(output)
        print(
            f"{name}完成：加权迟到={summary['加权迟到_优先系数乘秒']:.3f}，"
            f"完工时间={summary['全部任务完成时间_s']:.2f}s，"
            f"能耗={summary['总运输能耗_kWh']:.6f}kWh，"
            f"架次={summary['运输架次数']}，运行={summary['运行时间_s']:.2f}s。"
        )

    write_outputs(outputs)
    print("\n五算法对比完成：")
    print(pd.DataFrame([output_summary(output) for output in outputs]).to_string(index=False))
    print(f"\n输出目录：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
