def h2(solution, clauses):
    # number of satisfied clauses
    return sum(any((lit > 0 and solution[abs(lit) - 1]) or 
                   (lit < 0 and not solution[abs(lit) - 1]) 
                   for lit in clause) for clause in clauses)

def h1(solution, clauses):
    # weighted heuristic: satisfied clauses + penalty for unsatisfied
    satisfied = 0
    penalty = 0
    for clause in clauses:
        if any((lit > 0 and solution[abs(lit) - 1]) or 
               (lit < 0 and not solution[abs(lit) - 1]) for lit in clause):
            satisfied += 1
        else:
            penalty += 1
    return satisfied - penalty * 0.5
