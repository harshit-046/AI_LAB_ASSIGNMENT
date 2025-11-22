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

def expected_value(state, action, V):
    b1, b2 = state
    action = max(-min(MAX_MOVE, b2), min(MAX_MOVE, action, b1))  # enforce bounds

    nb1 = b1 - action
    nb2 = b2 + action

    expected = -2 * abs(action)

    for r1 in range(MAX_REQ):
        pr1 = poisson_prob(r1, rent_rate[0])
        rented1 = min(nb1, r1)

        for r2 in range(MAX_REQ):
            pr2 = poisson_prob(r2, rent_rate[1])
            rented2 = min(nb2, r2)

            prob_rent = pr1 * pr2
            income = (rented1 + rented2) * 10

            b1_after = nb1 - rented1
            b2_after = nb2 - rented2

            for ret1 in range(MAX_REQ):
                p1 = poisson_prob(ret1, return_rate[0])
                for ret2 in range(MAX_REQ):
                    p2 = poisson_prob(ret2, return_rate[1])

                    final_state = (
                        min(MAX_BIKES, b1_after + ret1),
                        min(MAX_BIKES, b2_after + ret2),
                    )
                    prob = prob_rent * p1 * p2
                    expected += prob * (income + DISCOUNT * V[final_state])

    return expected

def policy_iteration():
    V = np.zeros((MAX_BIKES + 1, MAX_BIKES + 1))
    policy = np.zeros((MAX_BIKES + 1, MAX_BIKES + 1), dtype=int)

    stable = False
    itr = 0

    while not stable:
        itr += 1
        print("\n--- Policy Iteration Round", itr, "---")

        # Evaluate policy
        for _ in range(5):
            newV = V.copy()
            for i in range(MAX_BIKES + 1):
                for j in range(MAX_BIKES + 1):
                    newV[i, j] = expected_value((i, j), policy[i, j], V)
            V = newV

        # Improve policy
        stable = True
        for i in range(MAX_BIKES + 1):
            for j in range(MAX_BIKES + 1):
                best_a = 0
                best_val = -np.inf
                for a in range(-MAX_MOVE, MAX_MOVE + 1):
                    val = expected_value((i, j), a, V)
                    if val > best_val:
                        best_val = val
                        best_a = a
                if best_a != policy[i, j]:
                    stable = False
                policy[i, j] = best_a

    return V, policy

if __name__ == "__main__":
    V, P = policy_iteration()
    print("\nOptimal Bike Movement Policy:")
    print(P)
