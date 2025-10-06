from evaluate import rand_sols, neighbors, evaluate

def hill_climbing(clauses, heuristic, max_iterations=1000):
    n = max(abs(lit) for clause in clauses for lit in clause)
    curr = rand_sols(n)
    curr_sco = evaluate(curr, clauses, heuristic)

    for _ in range(max_iterations):
        nbrs = neighbors(curr)
        scores = [evaluate(nb, clauses, heuristic) for nb in nbrs]
        best_sco = max(scores)
        if best_sco <= curr_sco:
            break
        best_neighbor = nbrs[scores.index(best_sco)]
        curr, curr_sco = best_neighbor, best_sco
        if curr_sco == len(clauses):
            break
    return curr, curr_sco
