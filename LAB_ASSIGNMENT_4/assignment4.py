import numpy as np
import matplotlib.pyplot as plt
import random
import math

def load_octave_ascii_mat(path):
    """Load Octave ASCII .mat file (skip comments, first two numbers = dims)."""
    numbers = []
    with open(path, "r") as f:
        for line in f:
            if line.strip() == "" or line.startswith("#"):
                continue
            numbers.extend([int(x) for x in line.strip().split()])
    h, w = numbers[0], numbers[1]
    data = np.array(numbers[2:], dtype=np.uint8)
    return data.reshape((h, w))

class JigsawPuzzleSolver:
    def __init__(self, image, grid_size=4, rng_seed=42):
        random.seed(rng_seed)
        np.random.seed(rng_seed)

        self.original_image = image
        self.grid_size = grid_size
        self.num_pieces = grid_size * grid_size
        self.height, self.width = image.shape

        self.piece_height = self.height // grid_size
        self.piece_width = self.width // grid_size
        self.pieces = self._create_pieces()

        # Precompute edges & compatibility costs
        self.right_edges = [p[:, -1].astype(np.float32) for p in self.pieces]
        self.left_edges  = [p[:, 0].astype(np.float32)  for p in self.pieces]
        self.bottom_edges= [p[-1, :].astype(np.float32) for p in self.pieces]
        self.top_edges   = [p[0, :].astype(np.float32)  for p in self.pieces]

        n = self.num_pieces
        self.right_cost = np.full((n, n), np.inf, dtype=np.float32)
        self.bottom_cost= np.full((n, n), np.inf, dtype=np.float32)
        self._compute_compatibility()

    def _create_pieces(self):
        pieces = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                piece = self.original_image[
                    i*self.piece_height:(i+1)*self.piece_height,
                    j*self.piece_width:(j+1)*self.piece_width
                ]
                pieces.append(piece)
        return pieces

    def _edge_diff(self, e1, e2):
        return float(np.mean(np.abs(e1 - e2)) / 255.0)

    def _compute_compatibility(self):
        n = self.num_pieces
        for i in range(n):
            for j in range(n):
                self.right_cost[i, j]  = self._edge_diff(self.right_edges[i], self.left_edges[j])
                self.bottom_cost[i, j] = self._edge_diff(self.bottom_edges[i], self.top_edges[j])

    def _energy(self, state):
        g, total = self.grid_size, 0.0
        for r in range(g):
            for c in range(g-1):
                total += self.right_cost[state[r*g+c], state[r*g+c+1]]
        for r in range(g-1):
            for c in range(g):
                total += self.bottom_cost[state[r*g+c], state[(r+1)*g+c]]
        return total

    def _greedy_build(self, first_piece=None):
        g, available, placed = self.grid_size, set(range(self.num_pieces)), []
        for pos in range(self.num_pieces):
            r, c = divmod(pos, g)
            best_p, best_cost = None, float('inf')
            for p in available:
                cost = 0.0
                if c > 0: cost += self.right_cost[placed[-1], p]
                if r > 0: cost += self.bottom_cost[placed[pos-g], p]
                if cost < best_cost:
                    best_p, best_cost = p, cost
            if pos == 0 and first_piece is not None:
                best_p = first_piece
            placed.append(best_p)
            available.remove(best_p)
        return placed

    def _hill_climb(self, state):
        n, current = self.num_pieces, list(state)
        current_energy = self._energy(current)
        improved = True
        while improved:
            improved = False
            for i in range(n-1):
                for j in range(i+1, n):
                    new_state = current.copy()
                    new_state[i], new_state[j] = new_state[j], new_state[i]
                    new_energy = self._energy(new_state)
                    if new_energy < current_energy:
                        current, current_energy, improved = new_state, new_energy, True
                        break
                if improved: break
        return current, current_energy

    def _simulated_annealing(self, start, initial_temp=1.0, cooling_rate=0.9995, max_iter=12000):
        current, best = start.copy(), start.copy()
        current_e, best_e, temp = self._energy(current), self._energy(start), initial_temp
        n = self.num_pieces
        for _ in range(max_iter):
            i, j = random.sample(range(n), 2)
            cand = current.copy()
            cand[i], cand[j] = cand[j], cand[i]
            delta = self._energy(cand) - current_e
            if delta < 0 or random.random() < math.exp(-delta/max(temp,1e-12)):
                current, current_e = cand, self._energy(cand)
                if current_e < best_e:
                    best, best_e = current.copy(), current_e
            temp *= cooling_rate
        return best, best_e

    def solve(self):
        best_state, best_energy = None, float('inf')

        # Try greedy with all seeds
        for seed in range(self.num_pieces):
            state = self._greedy_build(first_piece=seed)
            hc_state, hc_energy = self._hill_climb(state)
            if hc_energy < best_energy:
                best_state, best_energy = hc_state, hc_energy

        # Try random restarts
        for _ in range(50):
            perm = list(range(self.num_pieces))
            random.shuffle(perm)
            hc_state, hc_energy = self._hill_climb(perm)
            if hc_energy < best_energy:
                best_state, best_energy = hc_state, hc_energy

        # Refine with simulated annealing
        sa_state, sa_energy = self._simulated_annealing(best_state)
        if sa_energy < best_energy:
            best_state, best_energy = sa_state, sa_energy

        return best_state

    def reconstruct(self, state):
        g = self.grid_size
        img = np.zeros_like(self.original_image)
        for idx, pid in enumerate(state):
            r, c = divmod(idx, g)
            piece = self.pieces[pid]
            img[r*self.piece_height:(r+1)*self.piece_height,
                c*self.piece_width:(c+1)*self.piece_width] = piece
        return img

    def visualize(self, best_state):
        fig = plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(self.original_image, cmap='gray')
        plt.title("Original"); plt.axis("off")

        plt.subplot(1, 3, 2)
        scrambled = self.reconstruct(random.sample(range(self.num_pieces), self.num_pieces))
        plt.imshow(scrambled, cmap='gray')
        plt.title("Example Scramble"); plt.axis("off")

        plt.subplot(1, 3, 3)
        solved = self.reconstruct(best_state)
        plt.imshow(solved, cmap='gray')
        plt.title("Solved"); plt.axis("off")

        plt.tight_layout()
        plt.savefig("jigsaw_solution.png", dpi=200)
        plt.show()

def main():
    print("Loading scrambled_lena.mat...")
    image = load_octave_ascii_mat("scrambled_lena.mat")
    print(f"Image shape: {image.shape}, dtype={image.dtype}")

    solver = JigsawPuzzleSolver(image, grid_size=4, rng_seed=123)

    print("Solving...")
    best_state = solver.solve()

    solver.visualize(best_state)
    print("Image corrected")

if __name__ == "__main__":
    main()

