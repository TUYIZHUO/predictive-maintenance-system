# -*- coding: utf-8 -*-
"""
算法模块 3：遗传算法求解设备维护优先级排程
（对应《制造智能技术》"智能优化 / 进化计算"技术方向）。

问题建模：
    N 台设备需要维护，每台设备有故障风险、停机损失、资源需求、维护耗时。
    每天可用维护资源有限，因此需要排定维护优先级，使整体损失最小。

染色体编码：
    一个 0..N-1 的排列，表示维护优先级（越靠前越先维护）。

适应度（越小越好）：
    按染色体顺序贪心分配每天的维护资源；资源超出的设备顺延到下一天。
    每台设备损失 = 风险 × 停机损失 × (维护耗时 + 推迟天数 × 每日推迟惩罚)。
    总损失为所有设备损失之和。
    —— 这就是"把高风险、高停机损失的设备优先维护能减少总损失"的量化表达，
       也是本模块在系统里的实际作用（而非装饰性调用）。

算法流程：
    初始化随机种群 -> 逐代：计算适应度 -> 精英保留 -> 锦标赛选择 ->
    PMX 交叉 -> 交换变异，迭代收敛。
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Device:
    """一台待维护设备。"""
    device_id: str
    risk: float             # 故障风险概率 0~1（来自分类器 failure_prob）
    downtime_cost: float    # 单位时间停机损失（元/小时）
    resource: float         # 维护所需资源（工时）
    maintenance_time: float  # 维护耗时（小时）


def risk_to_level(prob: float, thresholds: Tuple[float, ...] = (0.3, 0.6)) -> int:
    """
    把故障概率映射为风险等级：0=低 1=中 2=高。
    供后端 /schedule 接口把分类器输出转成可读的风险档位。
    """
    return sum(1 for t in thresholds if prob >= t)


def build_devices(
    ids: List[str],
    risks: List[float],
    downtime_costs: List[float],
    resources: List[float],
    maintenance_times: List[float],
) -> List[Device]:
    """工厂函数：由平行列表构造 Device 列表。"""
    return [
        Device(did, r, c, res, t)
        for did, r, c, res, t in zip(ids, risks, downtime_costs, resources, maintenance_times)
    ]


class MaintenanceScheduler:
    """自实现遗传算法，求解维护优先级排序。"""

    def __init__(
        self,
        devices: List[Device],
        resource_capacity: float,
        pop_size: int = 50,
        n_generations: int = 200,
        elite_size: int = 2,
        crossover_prob: float = 0.8,
        mutation_prob: float = 0.2,
        penalty_per_day: float = 5.0,
        seed: Optional[int] = 42,
    ) -> None:
        if not devices:
            raise ValueError("设备列表不能为空")
        self.devices = devices
        self.n = len(devices)
        self.resource_capacity = resource_capacity
        self.pop_size = max(pop_size, 4)
        self.n_generations = n_generations
        self.elite_size = min(elite_size, self.pop_size)
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.penalty_per_day = penalty_per_day
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # 适应度：按优先级贪心调度，计算总损失
    # ------------------------------------------------------------------
    def _fitness(self, order: List[int]) -> float:
        """
        order : 0..n-1 的排列，表示维护优先级。
        贪心调度：依次排入当天，资源不足则顺延一天，损失随推迟天数线性放大。
        """
        day = 0
        used = 0.0
        total_loss = 0.0
        for idx in order:
            d = self.devices[idx]
            # 当前天资源不足 -> 开启下一天
            if used + d.resource > self.resource_capacity:
                day += 1
                used = 0.0
            delay = day
            loss = d.risk * d.downtime_cost * (d.maintenance_time + delay * self.penalty_per_day)
            total_loss += loss
            used += d.resource
        return total_loss

    # ------------------------------------------------------------------
    # 遗传算子
    # ------------------------------------------------------------------
    def _init_population(self) -> List[List[int]]:
        pop = []
        base = list(range(self.n))
        for _ in range(self.pop_size):
            order = base[:]
            self._rng.shuffle(order)
            pop.append(order)
        return pop

    def _tournament_select(self, pop: List[List[int]], fitnesses: List[float], k: int = 3) -> List[int]:
        """锦标赛选择：随机挑 k 个，返回其中适应度最小的个体。"""
        candidates = self._rng.sample(range(len(pop)), k)
        best = min(candidates, key=lambda i: fitnesses[i])
        return pop[best][:]

    def _pmx_crossover(self, p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
        """部分映射交叉（PMX），保证子代仍是合法排列。"""
        n = self.n
        a, b = sorted(self._rng.sample(range(n), 2))

        child1: List[Optional[int]] = [None] * n
        child2: List[Optional[int]] = [None] * n
        child1[a:b + 1] = p1[a:b + 1]
        child2[a:b + 1] = p2[a:b + 1]

        # 填充 child1：其余位置取 p2 的值，冲突时沿映射链替换
        for i in range(n):
            if child1[i] is not None:
                continue
            val = p2[i]
            while val in child1:
                val = p2[p1.index(val)]
            child1[i] = val

        # 填充 child2：对称
        for i in range(n):
            if child2[i] is not None:
                continue
            val = p1[i]
            while val in child2:
                val = p1[p2.index(val)]
            child2[i] = val

        return child1, child2

    def _swap_mutation(self, order: List[int]) -> List[int]:
        """交换变异：随机交换两个位置。"""
        order = order[:]
        i, j = self._rng.sample(range(self.n), 2)
        order[i], order[j] = order[j], order[i]
        return order

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def run(self) -> Tuple[List[int], float, List[float]]:
        """
        运行遗传算法。
        返回：
            best_order  : 最优维护优先级（设备下标排列）
            best_fitness: 最优适应度（总损失）
            history     : 每代最优适应度（用于观察收敛）
        """
        if self.n == 1:
            return [0], self._fitness([0]), [self._fitness([0])]

        pop = self._init_population()
        fitnesses = [self._fitness(p) for p in pop]
        history: List[float] = []

        for _ in range(self.n_generations):
            # 记录当前代最优
            best_i = min(range(len(pop)), key=lambda i: fitnesses[i])
            history.append(fitnesses[best_i])

            # 精英保留
            sorted_idx = sorted(range(len(pop)), key=lambda i: fitnesses[i])
            new_pop = [pop[i][:] for i in sorted_idx[:self.elite_size]]

            # 生成剩余个体
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_select(pop, fitnesses)
                p2 = self._tournament_select(pop, fitnesses)
                if self._rng.random() < self.crossover_prob:
                    c1, c2 = self._pmx_crossover(p1, p2)
                else:
                    c1, c2 = p1, p2
                if self._rng.random() < self.mutation_prob:
                    c1 = self._swap_mutation(c1)
                if self._rng.random() < self.mutation_prob:
                    c2 = self._swap_mutation(c2)
                new_pop.append(c1)
                if len(new_pop) < self.pop_size:
                    new_pop.append(c2)

            pop = new_pop
            fitnesses = [self._fitness(p) for p in pop]

        best_i = min(range(len(pop)), key=lambda i: fitnesses[i])
        history.append(fitnesses[best_i])
        return pop[best_i][:], fitnesses[best_i], history


def schedule_to_plan(
    devices: List[Device],
    order: List[int],
    resource_capacity: float,
) -> List[dict]:
    """
    把最优排序解析为可读的维护计划：设备ID、优先级、风险等级、
    建议维护批次（第几天）。
    """
    plan = []
    day = 0
    used = 0.0
    for rank, idx in enumerate(order, start=1):
        d = devices[idx]
        if used + d.resource > resource_capacity:
            day += 1
            used = 0.0
        used += d.resource
        plan.append({
            "device_id": d.device_id,
            "priority": rank,
            "risk_level": risk_to_level(d.risk),
            "risk": d.risk,
            "batch_day": day,
        })
    return plan
