import random

def evaluate(solution, clauses, heuristic):
    # evaluate solution using given heuristic function
    return heuristic(solution, clauses)

def rand_sols(n):
    return [random.choice([True, False]) for _ in range(n)]

def neighbors(current):
    n = len(current)
    nbrs = []
    for i in range(n):
        neighbor = current.copy()
        neighbor[i] = not neighbor[i]
        nbrs.append(neighbor)
    return nbrs
