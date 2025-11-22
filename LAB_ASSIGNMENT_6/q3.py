import numpy as np
import matplotlib.pyplot as plt

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        """Hebbian Learning Rule"""
        num_patterns = len(patterns)
        for p in patterns:
            self.weights += np.outer(p, p)
        np.fill_diagonal(self.weights, 0)
        self.weights /= self.size

    def predict(self, pattern, steps=5):
        """Asynchronous update"""
        state = pattern.copy()
        indices = np.arange(self.size)
        
        for _ in range(steps * self.size):
            i = np.random.choice(indices)
            # Update rule: sign(sum(Wij * sj))
            activation = np.dot(self.weights[i], state)
            state[i] = 1 if activation >= 0 else -1
        return state

# --- Simulation ---
# 1. Define a 10x10 pattern (e.g., a Cross 'X')
N = 100 # 10x10
pattern_X = -1 * np.ones((10, 10))
for i in range(10):
    pattern_X[i, i] = 1
    pattern_X[i, 9-i] = 1

flat_X = pattern_X.flatten()

# 2. Train Network
hn = HopfieldNetwork(N)
hn.train([flat_X])

# 3. Add Noise (Flip 20% of bits)
noisy_X = flat_X.copy()
noise_indices = np.random.choice(N, size=20, replace=False)
noisy_X[noise_indices] *= -1

# 4. Recover
recovered_X = hn.predict(noisy_X)

# 5. Visualization
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(flat_X.reshape(10, 10), cmap='gray')
ax[0].set_title("Original Pattern")
ax[1].imshow(noisy_X.reshape(10, 10), cmap='gray')
ax[1].set_title("Corrupted Input (20% Noise)")
ax[2].imshow(recovered_X.reshape(10, 10), cmap='gray')
ax[2].set_title("Recovered Pattern")
plt.show()
