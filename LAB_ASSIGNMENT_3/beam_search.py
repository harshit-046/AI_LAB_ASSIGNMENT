from evaluate import rand_sols, neighbors, evaluate

def beam_search(clauses, heuristic, beam_width=3, max_iterations=100):
    n = max(abs(lit) for clause in clauses for lit in clause)
    beam = [rand_sols(n) for _ in range(beam_width)]

    for _ in range(max_iterations):
        candidates = []
        for assignment in beam:
            candidates.extend(neighbors(assignment))
        candidates = sorted(candidates, key=lambda x: evaluate(x, clauses, heuristic), reverse=True)
        beam = candidates[:beam_width]

        for assignment in beam:
            if evaluate(assignment, clauses, heuristic) == len(clauses):
                return assignment, len(clauses)
    best_assignment = max(beam, key=lambda x: evaluate(x, clauses, heuristic))
    return best_assignment, evaluate(best_assignment, clauses, heuristic)
