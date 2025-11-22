import numpy as np
from math import exp, factorial

DISCOUNT = 0.9
MAX_BIKES = 20
MAX_MOVE = 5

rent_rate = [3, 4]
return_rate = [3, 2]
MAX_REQ = 7

def poisson_prob(n, lam):
    return (lam**n * exp(-lam)) / factorial(n)

def expected_modified(state, action, V):
    b1, b2 = state

    action = max(-min(MAX_MOVE, b2), min(action, min(MAX_MOVE, b1)))

    nb1 = b1 - action
    nb2 = b2 + action

    if action > 0:
        move_cost = -2 * (action - 1)   # 1 bike free
    else:
        move_cost = -2 * abs(action)

    if nb1 > 10:
        move_cost -= 4
    if nb2 > 10:
        move_cost -= 4

    expected = move_cost

    for r1 in range(MAX_REQ):
        pr1 = poisson_prob(r1, rent_rate[0])
        rent1 = min(nb1, r1)

        for r2 in range(MAX_REQ):
            pr2 = poisson_prob(r2, rent_rate[1])
            rent2 = min(nb2, r2)

            prob_rent = pr1 * pr2
            reward = (rent1 + rent2) * 10

            b1_after, b2_after = nb1 - rent1, nb2 - rent2

            for ret1 in range(MAX_REQ):
                for ret2 in range(MAX_REQ):
                    prob = prob_rent * poisson_prob(ret1, return_rate[0]) * poisson_prob(ret2, return_rate[1])
                    new_state = (
                        min(MAX_BIKES, b1_after + ret1),
                        min(MAX_BIKES, b2_after + ret2),
                    )
                    expected += prob * (reward + DISCOUNT * V[new_state])

    return expected

def policy_iteration_modified():
    V = np.zeros((MAX_BIKES + 1, MAX_BIKES + 1))
    policy = np.zeros((MAX_BIKES + 1, MAX_BIKES + 1), dtype=int)
    stable = False

    while not stable:
        for _ in range(5):
            newV = V.copy()
            for i in range(MAX_BIKES + 1):
                for j in range(MAX_BIKES + 1):
                    newV[i, j] = expected_modified((i, j), policy[i, j], V)
            V = newV

        stable = True
        for i in range(MAX_BIKES + 1):
            for j in range(MAX_BIKES + 1):
                bestA, bestV = 0, -np.inf
                for a in range(-MAX_MOVE, MAX_MOVE + 1):
                    v = expected_modified((i, j), a, V)
                    if v > bestV:
                        bestV, bestA = v, a
                if bestA != policy[i, j]:
                    stable = False
                policy[i, j] = bestA

    return V, policy

if __name__ == "__main__":
    V, P = policy_iteration_modified()
    print("\nOptimal Modified Policy:")
    print(P)
