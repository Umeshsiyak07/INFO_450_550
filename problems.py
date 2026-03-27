"""
problems.py
===========
CSP problem definitions for:
  1. N-Queens
  2. Job Shop Scheduling
  3. Airport Traffic Load Balancing

Each problem exposes a shared interface:
    problem.variables         -> list of variable names
    problem.domains           -> dict {var: [values]}
    problem.get_random_state()-> dict {var: value}  (complete random assignment)
    problem.is_goal(state)    -> bool
    problem.cost(state)       -> float (lower is better; 0 = solved)
    problem.constraint_violations(state) -> int
    problem.description       -> str
"""

import random


# ---------------------------------------------------------------------------
# 1. N-Queens
# ---------------------------------------------------------------------------

class NQueens:
    """
    Place N queens on an N×N board so no two share a row, column, or diagonal.
    Variables : columns 0..N-1
    Domain    : row positions 0..N-1
    """

    def __init__(self, n=8):
        self.n = n
        self.description = f"{n}-Queens"
        self.variables = list(range(n))
        self.domains = {col: list(range(n)) for col in range(n)}

    # --- shared interface ---------------------------------------------------

    def get_random_state(self):
        return {col: random.choice(self.domains[col]) for col in self.variables}

    def constraint_violations(self, state):
        violations = 0
        cols = self.variables
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                ri, rj = state[cols[i]], state[cols[j]]
                # same row
                if ri == rj:
                    violations += 1
                # same diagonal
                if abs(ri - rj) == abs(cols[i] - cols[j]):
                    violations += 1
        return violations

    def cost(self, state):
        return float(self.constraint_violations(state))

    def is_goal(self, state):
        return self.constraint_violations(state) == 0

    # --- helpers ------------------------------------------------------------

    def display(self, state):
        board = [["." for _ in range(self.n)] for _ in range(self.n)]
        for col, row in state.items():
            board[row][col] = "Q"
        for row in board:
            print(" ".join(row))


# ---------------------------------------------------------------------------
# 2. Job Shop Scheduling
# ---------------------------------------------------------------------------

class JobShopScheduling:
    """
    J jobs × M machines.  Each job has a fixed routing through machines with
    fixed processing times.  Assign a start time to every operation.

    Variables : (job, machine_step) tuples
    Domain    : discrete time slots 0..T_max-1
    Constraints (hard):
        - Precedence : operations within a job must follow order
        - No-overlap : each machine processes ≤1 job at a time
    Soft objective: minimize makespan (handled via cost())

    Default instance matches Fisher & Thompson ft06 (6 jobs × 6 machines).
    """

    # ft06 benchmark: jobs[i] = [(machine, duration), ...]
    FT06 = [
        [(2,1),(0,3),(1,6),(3,7),(5,3),(4,6)],
        [(1,8),(2,5),(4,10),(5,10),(0,10),(3,4)],
        [(2,5),(3,4),(5,8),(0,9),(1,1),(4,7)],
        [(1,5),(0,5),(2,5),(3,3),(4,8),(5,9)],
        [(2,9),(1,3),(4,5),(5,4),(0,3),(3,1)],
        [(1,3),(3,3),(5,9),(0,10),(4,4),(2,1)],
    ]

    def __init__(self, jobs=None, t_max=None):
        self.jobs = jobs if jobs is not None else self.FT06
        self.n_jobs = len(self.jobs)
        self.n_machines = len(self.jobs[0])
        # generous upper bound for domain
        self.t_max = t_max if t_max is not None else sum(
            d for job in self.jobs for _, d in job
        )
        self.description = f"Job Shop Scheduling ({self.n_jobs}×{self.n_machines})"

        # variables: one per operation
        self.variables = [
            (j, s) for j in range(self.n_jobs) for s in range(self.n_machines)
        ]
        self.domains = {v: list(range(self.t_max)) for v in self.variables}

    # --- shared interface ---------------------------------------------------

    def get_random_state(self):
        return {v: random.choice(self.domains[v]) for v in self.variables}

    def constraint_violations(self, state):
        violations = 0

        # Precedence constraints
        for j, job in enumerate(self.jobs):
            for s in range(len(job) - 1):
                _, dur = job[s]
                start_s = state[(j, s)]
                start_next = state[(j, s + 1)]
                if start_s + dur > start_next:
                    violations += 1

        # No-overlap constraints (machine capacity)
        # Group operations by machine
        machine_ops = {}
        for j, job in enumerate(self.jobs):
            for s, (m, dur) in enumerate(job):
                machine_ops.setdefault(m, []).append((j, s, dur))

        for m, ops in machine_ops.items():
            for i in range(len(ops)):
                j1, s1, d1 = ops[i]
                st1 = state[(j1, s1)]
                for k in range(i + 1, len(ops)):
                    j2, s2, d2 = ops[k]
                    st2 = state[(j2, s2)]
                    # overlap if intervals intersect
                    if st1 < st2 + d2 and st2 < st1 + d1:
                        violations += 1

        return violations

    def makespan(self, state):
        end_times = []
        for j, job in enumerate(self.jobs):
            for s, (_, dur) in enumerate(job):
                end_times.append(state[(j, s)] + dur)
        return max(end_times)

    def cost(self, state):
        # Penalise violations heavily; also consider makespan
        violations = self.constraint_violations(state)
        if violations > 0:
            return float(violations * 1000 + self.makespan(state))
        return float(self.makespan(state))

    def is_goal(self, state):
        return self.constraint_violations(state) == 0


# ---------------------------------------------------------------------------
# 3. Airport Traffic Load Balancing
# ---------------------------------------------------------------------------

class AirportLoadBalancing:
    """
    Assign each US state a congestion label {Green, Yellow, Orange, Red}
    based on airport traffic intensity.

    Constraints (hard):
        - Adjacency  : neighboring high-traffic states cannot both be Red
        - Cardinality: total Red assignments ≤ K

    Soft objective: label should correlate with actual traffic score.

    Traffic scores are normalised [0,1] placeholders (replace with real
    FAA Bureau of Transportation Statistics data before final report).
    """

    LABELS = ["Green", "Yellow", "Orange", "Red"]
    LABEL_RANK = {"Green": 0, "Yellow": 1, "Orange": 2, "Red": 3}

    # Placeholder traffic intensity scores (0=low, 1=high).
    # Source: to be replaced with FAA BTS data for final report.
    TRAFFIC_SCORES = {
        "CA": 1.00, "TX": 0.95, "FL": 0.90, "NY": 0.88, "IL": 0.80,
        "GA": 0.75, "CO": 0.70, "WA": 0.68, "NV": 0.65, "AZ": 0.63,
        "NC": 0.60, "VA": 0.58, "MA": 0.57, "PA": 0.55, "MN": 0.52,
        "MI": 0.50, "OR": 0.48, "MD": 0.46, "TN": 0.44, "MO": 0.42,
        "OH": 0.40, "NJ": 0.38, "UT": 0.36, "HI": 0.34, "SC": 0.32,
        "WI": 0.30, "KY": 0.28, "IN": 0.26, "LA": 0.25, "OK": 0.24,
        "AL": 0.22, "NM": 0.20, "KS": 0.19, "NE": 0.18, "CT": 0.17,
        "AR": 0.16, "MS": 0.15, "IA": 0.14, "ID": 0.13, "WV": 0.12,
        "NH": 0.11, "ME": 0.10, "RI": 0.09, "ND": 0.08, "SD": 0.07,
        "MT": 0.06, "DE": 0.05, "AK": 0.04, "VT": 0.03, "WY": 0.02,
    }

    # Adjacency list (contiguous US states)
    ADJACENCY = {
        "AL": ["FL","GA","MS","TN"],
        "AK": [],
        "AZ": ["CA","CO","NM","NV","UT"],
        "AR": ["LA","MO","MS","OK","TN","TX"],
        "CA": ["AZ","NV","OR"],
        "CO": ["AZ","KS","NE","NM","OK","UT","WY"],
        "CT": ["MA","NY","RI"],
        "DE": ["MD","NJ","PA"],
        "FL": ["AL","GA"],
        "GA": ["AL","FL","NC","SC","TN"],
        "HI": [],
        "ID": ["MT","NV","OR","UT","WA","WY"],
        "IL": ["IN","IA","KY","MO","WI"],
        "IN": ["IL","KY","MI","OH"],
        "IA": ["IL","MN","MO","NE","SD","WI"],
        "KS": ["CO","MO","NE","OK"],
        "KY": ["IL","IN","MO","OH","TN","VA","WV"],
        "LA": ["AR","MS","TX"],
        "ME": ["NH"],
        "MD": ["DE","PA","VA","WV"],
        "MA": ["CT","NH","NY","RI","VT"],
        "MI": ["IN","OH","WI"],
        "MN": ["IA","ND","SD","WI"],
        "MS": ["AL","AR","LA","TN"],
        "MO": ["AR","IL","IA","KS","KY","NE","OK","TN"],
        "MT": ["ID","ND","SD","WY"],
        "NE": ["CO","IA","KS","MO","SD","WY"],
        "NV": ["AZ","CA","ID","OR","UT"],
        "NH": ["MA","ME","VT"],
        "NJ": ["DE","NY","PA"],
        "NM": ["AZ","CO","OK","TX"],
        "NY": ["CT","MA","NJ","PA","VT"],
        "NC": ["GA","SC","TN","VA"],
        "ND": ["MN","MT","SD"],
        "OH": ["IN","KY","MI","PA","WV"],
        "OK": ["AR","CO","KS","MO","NM","TX"],
        "OR": ["CA","ID","NV","WA"],
        "PA": ["DE","MD","NJ","NY","OH","WV"],
        "RI": ["CT","MA"],
        "SC": ["GA","NC"],
        "SD": ["IA","MN","MT","ND","NE","WY"],
        "TN": ["AL","AR","GA","KY","MS","MO","NC","VA"],
        "TX": ["AR","LA","NM","OK"],
        "UT": ["AZ","CO","ID","NV","NM","WY"],
        "VT": ["MA","NH","NY"],
        "VA": ["KY","MD","NC","TN","WV"],
        "WA": ["ID","OR"],
        "WV": ["KY","MD","OH","PA","VA"],
        "WI": ["IL","IA","MI","MN"],
        "WY": ["CO","ID","MT","NE","SD","UT"],
    }

    # High-traffic threshold: states above this score trigger adjacency constraints
    HIGH_TRAFFIC_THRESHOLD = 0.40

    def __init__(self, red_cap=10):
        """
        red_cap : maximum number of states that may be assigned Red.
        """
        self.red_cap = red_cap
        self.description = f"Airport Load Balancing (Red cap={red_cap})"
        self.variables = sorted(self.TRAFFIC_SCORES.keys())
        self.domains = {s: list(self.LABELS) for s in self.variables}

        # Pre-compute high-traffic states for adjacency constraint
        self.high_traffic = {
            s for s, score in self.TRAFFIC_SCORES.items()
            if score >= self.HIGH_TRAFFIC_THRESHOLD
        }

    # --- shared interface ---------------------------------------------------

    def get_random_state(self):
        return {s: random.choice(self.LABELS) for s in self.variables}

    def constraint_violations(self, state):
        violations = 0

        # Adjacency: neighboring high-traffic states cannot both be Red
        checked = set()
        for s in self.high_traffic:
            for nb in self.ADJACENCY.get(s, []):
                if nb in self.high_traffic and (nb, s) not in checked:
                    checked.add((s, nb))
                    if state[s] == "Red" and state[nb] == "Red":
                        violations += 1

        # Global cardinality: total Red ≤ red_cap
        red_count = sum(1 for v in state.values() if v == "Red")
        if red_count > self.red_cap:
            violations += red_count - self.red_cap

        return violations

    def soft_score(self, state):
        """
        Soft objective: reward assignments where label rank correlates
        with traffic intensity.  Returns value in [0, 1] (higher = better).
        """
        total = 0.0
        for s, label in state.items():
            rank = self.LABEL_RANK[label] / 3.0          # normalise to [0,1]
            traffic = self.TRAFFIC_SCORES[s]
            # penalise mismatch between rank and traffic score
            total += 1.0 - abs(rank - traffic)
        return total / len(self.variables)

    def cost(self, state):
        violations = self.constraint_violations(state)
        soft = self.soft_score(state)
        # violations dominate; soft score breaks ties
        return float(violations * 100) + (1.0 - soft)

    def is_goal(self, state):
        return self.constraint_violations(state) == 0