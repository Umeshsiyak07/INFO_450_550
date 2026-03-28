"""
demo.py
=======
Demonstrates a RandomAgent solving each of the three CSP problems.

Run:
    python demo.py

Each demo is capped at a short time limit so the script finishes quickly.
"""

from problems import NQueensProblem, JobShopScheduling, AirportLoadBalancing
from Algorithms import RandomAgent

TIME_LIMIT = 5  


def print_header(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_metrics(metrics):
    print(f"  Solved (zero violations) : {metrics['is_goal']}")
    print(f"  Constraint violations    : {metrics['violations']}")
    print(f"  Best cost                : {metrics['cost']:.4f}")
    print(f"  Iterations               : {metrics['iterations']}")
    print(f"  Constraint check        : {metrics['constraint_check']}")
    print(f"  Runtime                  : {metrics['runtime_s']:.4f}s")


def demo_nqueens():
    print_header("Problem 1: N-Queens  (Random Agent)")
    agent = RandomAgent(seed=42)

    for n in [8, 20]:
        problem = NQueensProblem(n=n)
        print(f"\n  N = {n}  |  time limit = {TIME_LIMIT}s")
        state, metrics = agent.solve(problem, time_limit=TIME_LIMIT)
        print_metrics(metrics)

        if n == 8 and metrics["is_goal"]:
            print("\n  Solution board:")
            problem.display(state)


def demo_jobshop():
    print_header("Problem 2: Job Shop Scheduling  (Random Agent)")
    agent = RandomAgent(seed=42)

    problem = JobShopScheduling()   # ft06 by default
    print(f"\n  Instance : ft06 (6 jobs × 6 machines)")
    print(f"  Time limit: {TIME_LIMIT}s")
    state, metrics = agent.solve(problem, time_limit=TIME_LIMIT)
    print_metrics(metrics)

    if metrics["is_goal"]:
        print(f"  Makespan : {problem.makespan(state)}")


def demo_airport():
    print_header("Problem 3: Airport Traffic Load Balancing  (Random Agent)")
    agent = RandomAgent(seed=42)

    problem = AirportLoadBalancing(red_cap=10)
    print(f"\n  Variables : {len(problem.variables)} US states")
    print(f"  Red cap   : {problem.red_cap}")
    print(f"  Time limit: {TIME_LIMIT}s")
    state, metrics = agent.solve(problem, time_limit=TIME_LIMIT)
    print_metrics(metrics)

    if metrics["is_goal"]:
        soft = problem.soft_score(state)
        print(f"  Soft score (label-traffic correlation): {soft:.4f}")

    # sample
    print("\n  Sample assignments (top 10 states by traffic):")
    top_states = sorted(
        problem.variables,
        key=lambda s: problem.TRAFFIC_SCORES[s],
        reverse=True
    )[:10]
    for s in top_states:
        score = problem.TRAFFIC_SCORES[s]
        label = state[s]
        print(f"    {s:>4}  traffic={score:.2f}  assigned={label}")


if __name__ == "__main__":
    print("\nCSP Algorithm Comparison — Progress Report Demo")
    print(f"Random Agent | Time limit per problem: {TIME_LIMIT}s")

    demo_nqueens()
    demo_jobshop()
    demo_airport()

    print("\n" + "=" * 60)
    print("  Demo complete.")
    print("=" * 60 + "\n")