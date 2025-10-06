from generate_k_sat import generate_k_sat
from heuristics import h1, h2
from hill_climbing import hill_climbing
from beam_search import beam_search
from vnd import vnd

def main_run(k=3, m=5, n=4):
    clauses = generate_k_sat(k, m, n, seed=42)
    print("generated k-SAT clauses:")
    print(clauses)

    for heuristic, name in [(h1, "h1"), (h2, "h2")]:
        print(f"\nheuristic: {name}")

        print("\nhill climbing")
        solution, score = hill_climbing(clauses, heuristic)
        print("satisfied clauses:", score, "/", len(clauses))
        print("solution:", solution)

        print("\nbeam search (width=3)")
        solution, score = beam_search(clauses, heuristic, beam_width=3)
        print("satisfied clauses:", score, "/", len(clauses))
        print("solution:", solution)

        print("\nbeam search (width=4)")
        solution, score = beam_search(clauses, heuristic, beam_width=4)
        print("satisfied clauses:", score, "/", len(clauses))
        print("solution:", solution)

        print("\nvariable neighborhood descent")
        solution, score = vnd(clauses, heuristic)
        print("Satisfied clauses:", score, "/", len(clauses))
        print("Solution:", solution)

if __name__ == "__main__":
    main_run(k=3, m=5, n=4)
