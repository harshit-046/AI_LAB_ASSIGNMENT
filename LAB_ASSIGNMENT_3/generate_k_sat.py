import random

def generate_k_sat(k, m, n, seed=None):
    # generate a uniform random k-SAT instance.
    # k: number of literals per clause
    # m: number of clauses
    # n: number of vars
    # seed: optional random seed
    # returns: list of clauses

    if seed is not None:
        random.seed(seed)

    clauses = []
    for _ in range(m):
        vars = random.sample(range(1, n + 1), k)
        clause = []
        for var in vars:
            literal = var if random.choice([True, False]) else -var
            clause.append(literal)
        clauses.append(clause)
    return clauses
