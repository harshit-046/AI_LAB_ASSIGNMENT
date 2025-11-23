import numpy as np
import matplotlib.pyplot as plt

def initialize_board(size=8):
    """Initialize the board with rooks placed randomly."""
    board = np.zeros((size, size), dtype=int)
    positions = np.random.choice(size * size, size, replace=False)
    for pos in positions:
        board[pos // size, pos % size] = 1
    return board

def calculate_energy(board):
    """Calculate the energy of the board configuration."""
    row_conflicts = sum((np.sum(board, axis=1) - 1) ** 2)
    col_conflicts = sum((np.sum(board, axis=0) - 1) ** 2)
    return row_conflicts + col_conflicts

def optimize_board(board, iterations=1000):
    """Optimize the board configuration to minimize energy."""
    current_energy = calculate_energy(board)
    size = board.shape[0]
    
    for _ in range(iterations):
        pos1, pos2 = np.random.choice(size * size, 2, replace=False)
        r1, c1 = divmod(pos1, size)
        r2, c2 = divmod(pos2, size)
        
        if board[r1, c1] == 1 and board[r2, c2] == 0:
            board[r1, c1], board[r2, c2] = board[r2, c2], board[r1, c1]
            new_energy = calculate_energy(board)
            if new_energy < current_energy:
                current_energy = new_energy
            else:
                board[r1, c1], board[r2, c2] = board[r2, c2], board[r1, c1]
    
    return board, current_energy

# Initialize and optimize the board
initial_board = initialize_board()
plt.imshow(initial_board, cmap='binary', interpolation='nearest')
plt.title("Initial Board")
plt.show()

optimized_board, final_energy = optimize_board(initial_board)
print("Final Energy of the optimized board:", final_energy)

plt.imshow(optimized_board, cmap='binary', interpolation='nearest')
plt.title("Optimized Board")
plt.show()
