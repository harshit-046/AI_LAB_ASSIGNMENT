import numpy as np

# Grid definitions
ROWS, COLS = 3, 4
GOAL_STATE = (0, 3)
FAIL_STATE = (1, 3)
BLOCK = (1, 1)
DISCOUNT = 0.99
STEP_REWARD = 0.02   # change according to question: -2, 0.1, 0.02, 1

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

def inside(r, c):
    return 0 <= r < ROWS and 0 <= c < COLS and (r, c) != BLOCK

def transition(state, action):
    r, c = state
    moves = {"UP":(-1,0),"DOWN":(1,0),"LEFT":(0,-1),"RIGHT":(0,1)}
    nr, nc = r + moves[action][0], c + moves[action][1]
    return (nr, nc) if inside(nr, nc) else state

# slips
left_turn = {"UP":"LEFT","DOWN":"RIGHT","LEFT":"DOWN","RIGHT":"UP"}
right_turn = {"UP":"RIGHT","DOWN":"LEFT","LEFT":"UP","RIGHT":"DOWN"}

def value_iteration():
    V = np.zeros((ROWS, COLS))
    iteration = 0

    while True:
        diff = 0
        newV = V.copy()

        for r in range(ROWS):
            for c in range(COLS):
                if (r, c) in [GOAL_STATE, FAIL_STATE, BLOCK]:
                    continue

                outcomes = []
                for act in ACTIONS:
                    val = 0.8 * V[transition((r, c), act)]
                    val += 0.1 * V[transition((r, c), left_turn[act])]
                    val += 0.1 * V[transition((r, c), right_turn[act])]
                    outcomes.append(val)

                newV[r, c] = STEP_REWARD + DISCOUNT * max(outcomes)
                diff = max(diff, abs(newV[r, c] - V[r, c]))

        V = newV
        iteration += 1
        if diff < 1e-4:
            break

    return V, iteration

if __name__ == "__main__":
    values, k = value_iteration()
    print("\nFinal Value Table after", k, "iterations:")
    print(np.round(values, 3))
