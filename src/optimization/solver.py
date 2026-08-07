from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
import pulp

from config.pipeline import load_config


def _player_vars(df: pd.DataFrame, problem: pulp.LpProblem) -> Dict[int, pulp.LpVariable]:
    return {int(row["element_id"]): pulp.LpVariable(f"p_{int(row['element_id'])}", cat="Binary") for _, row in df.iterrows()}


def solve_squad_selection(
    df: pd.DataFrame,
    horizon_gws: int = 5,
    budget: float = 100.0,
) -> pulp.LpProblem:
    config = load_config()["optimization"]
    position_constraints = config.get("position_constraints", {})
    max_per_team = config.get("max_players_per_team", 3)

    problem = pulp.LpProblem("FPL_Squad_Selection", pulp.LpMaximize)
    players = _player_vars(df, problem)

    df["horizon_xPts"] = df.get("xPts", 0.0) * horizon_gws
    problem += pulp.lpSum(players[int(r["element_id"])] * float(r["horizon_xPts"]) for _, r in df.iterrows())

    problem += pulp.lpSum(players[p] * float(df.loc[df["element_id"] == p, "now_cost"].values[0]) for p in players) <= budget * 10

    for pos, limit in position_constraints.items():
        problem += (
            pulp.lpSum(players[int(r["element_id"])] for _, r in df.iterrows() if r.get("position") == pos)
            == limit
        )

    for team in df["team"].unique():
        problem += (
            pulp.lpSum(players[int(r["element_id"])] for _, r in df.iterrows() if r.get("team") == team)
            <= max_per_team
        )

    problem += pulp.lpSum(players.values()) == config.get("squad_size", 15)

    logger.info("MILP squad selection problem formulated with {} players", len(players))
    return problem


def solve_transfer_plan(
    current_squad: pd.DataFrame,
    candidate_df: pd.DataFrame,
    max_transfers: int = 1,
    transfer_penalty: float = 4.0,
    horizon_gws: int = 5,
) -> pulp.LpProblem:
    config = load_config()["optimization"]
    problem = pulp.LpProblem("FPL_Transfer_Plan", pulp.LpMaximize)

    current_ids = {int(r["element_id"]) for _, r in current_squad.iterrows()}
    candidate_ids = {int(r["element_id"]) for _, r in candidate_df.iterrows()}
    all_ids = current_ids | candidate_ids

    in_vars = {pid: pulp.LpVariable(f"in_{pid}", cat="Binary") for pid in candidate_ids}
    out_vars = {pid: pulp.LpVariable(f"out_{pid}", cat="Binary") for pid in current_ids}

    value_map = {int(r["element_id"]): float(r.get("xPts", 0.0)) * horizon_gws for _, r in candidate_df.iterrows()}
    cost_map = {int(r["element_id"]): float(r.get("now_cost", 0.0)) for _, r in candidate_df.iterrows()}
    current_cost_map = {int(r["element_id"]): float(r.get("now_cost", 0.0)) for _, r in current_squad.iterrows()}
    current_value_map = {int(r["element_id"]): float(r.get("xPts", 0.0)) * horizon_gws for _, r in current_squad.iterrows()}

    problem += (
        pulp.lpSum(value_map.get(pid, 0.0) * in_vars[pid] for pid in candidate_ids)
        - pulp.lpSum(current_value_map.get(pid, 0.0) * out_vars[pid] for pid in current_ids)
        - transfer_penalty * pulp.lpSum(out_vars.values())
    )

    problem += pulp.lpSum(out_vars.values()) <= max_transfers
    problem += pulp.lpSum(in_vars.values()) == pulp.lpSum(out_vars.values())

    current_budget = float(current_squad["now_cost"].sum()) if "now_cost" in current_squad.columns else 100.0 * 10
    problem += (
        pulp.lpSum(cost_map.get(pid, 0.0) * in_vars[pid] for pid in candidate_ids)
        - pulp.lpSum(current_cost_map.get(pid, 0.0) * out_vars[pid] for pid in current_ids)
        <= 0
    )

    logger.info("Transfer plan problem formulated")
    return problem


def solve_problem(problem: pulp.LpProblem) -> Optional[Dict[str, float]]:
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=60)
    result = problem.solve(solver)
    if pulp.LpStatus[result] != "Optimal":
        logger.warning("Solver status: {}", pulp.LpStatus[result])
        return None
    solution = {v.name: v.value() for v in problem.variables() if v.value() is not None and v.value() > 0.5}
    logger.info("Optimal objective value: {:.4f}", pulp.value(problem.objective))
    return solution
