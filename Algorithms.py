"""
algorithms.py
=============
Agent classes for solving CSP problems.

Progress-report deliverable: RandomAgent only.
All future algorithm classes (Backtracking, MinConflicts, SimulatedAnnealing,
GeneticAlgorithm, ConstraintWeightingTabu) will be added here without
modifying problems.py.

Shared agent interface
----------------------
Every agent class must implement:

    agent.solve(problem, time_limit=60) -> (state, metrics)

Where:
    problem   : any problem object from problems.py
    time_limit: maximum wall-clock seconds allowed
    state     : dict {variable: value} — best assignment found
    metrics   : dict with keys
                  'cost'               (float)
                  'violations'         (int)
                  'constraint_check'  (int)
                  'runtime_s'          (float)
                  'is_goal'            (bool)
"""

import random
import time


class RandomAgent:
    """
    Baseline agent.  Repeatedly samples a random complete assignment and
    returns the best one found within the time limit.

    Works on any problem that implements random_state(), cost(), 
    constraint_violations(), and is_goal().
    """

    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)

    def solve(self, problem, time_limit=10):
        """
        Parameters
        ----------
        problem    : problem object from problems.py
        time_limit : seconds before stopping (default 10s)

        Returns
        -------
        best_state : dict {variable: value}
        metrics    : dict
        """
        start = time.time()
        constraint_check = 0

        best_state = None
        best_cost = float("inf")
        iterations = 0

        while time.time() - start < time_limit:
            state = problem.random_state()
            c = problem.cost(state)
            constraint_check += 1          
            iterations += 1

            if c < best_cost:
                best_cost = c
                best_state = state

            if problem.is_goal(best_state):
                break

        runtime = time.time() - start

        metrics = {
            "cost": best_cost,
            "violations": problem.constraint_violations(best_state),
            "constraint_check": constraint_check,
            "runtime_s": round(runtime, 4),
            "iterations": iterations,
            "is_goal": problem.is_goal(best_state),
        }

        return best_state, metrics