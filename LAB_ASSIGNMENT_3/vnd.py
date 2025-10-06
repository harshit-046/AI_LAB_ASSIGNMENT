from evaluate import rand_sols, evaluate

def vnd(clauses, heuristic, max_iterations=1000):
    n = max(abs(lit) for clause in clauses for lit in clause)
    curr = rand_sols(n)
    curr_score = evaluate(curr, clauses, heuristic)

    for _ in range(max_iterations):
        improved = False
        for k in [1, 2, 3]:
            nbrs = []
            idxs = list(range(n))
            if k == 1:
                for i in idxs:
                    neighbor = curr.copy()
                    neighbor[i] = not neighbor[i]
                    nbrs.append(neighbor)
            elif k == 2:
                for i in idxs:
                    for j in idxs:
                        if i < j:
                            neighbor = curr.copy()
                            neighbor[i] = not neighbor[i]
                            neighbor[j] = not neighbor[j]
                            nbrs.append(neighbor)
            else:
                for i in idxs:
                    for j in idxs:
                        for l in idxs:
                            if i < j < l:
                                neighbor = curr.copy()
                                neighbor[i] = not neighbor[i]
                                neighbor[j] = not neighbor[j]
                                neighbor[l] = not neighbor[l]
                                nbrs.append(neighbor)

            best_score = curr_score
            best_neighbor = curr
            for nb in nbrs:
                score = evaluate(nb, clauses, heuristic)
                if score > best_score:
                    best_score = score
                    best_neighbor = nb
                    improved = True
            curr, curr_score = best_neighbor, best_score
            if curr_score == len(clauses):
                return curr, curr_score
        if not improved:
            break
    return curr, curr_score
