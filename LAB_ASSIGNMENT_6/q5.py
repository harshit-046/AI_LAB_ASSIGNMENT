import numpy as np
import matplotlib.pyplot as plt

# Define city names and generate random coordinates
city_names = [str(i) for i in range(1, 11)]
np.random.seed(0)  # Ensure reproducibility
city_coordinates = {city: np.random.rand(2) * 50 for city in city_names}

# Function to compute the distance matrix between cities
def compute_distance_matrix(city_names):
    num_cities = len(city_names)
    distance_matrix = np.zeros((num_cities, num_cities))
    for i, city_i in enumerate(city_names):
        for j, city_j in enumerate(city_names):
            if i != j:
                distance_matrix[i, j] = np.linalg.norm(city_coordinates[city_i] - city_coordinates[city_j])
    return distance_matrix

# Class to solve the Traveling Salesman Problem
class TravelingSalesmanSolver:
    def __init__(self, city_names, distance_matrix):
        self.city_names = city_names
        self.num_cities = len(city_names)
        self.distance_matrix = distance_matrix
        self.weights = -distance_matrix

    def calculate_tour_cost(self, tour):
        cost = sum(self.distance_matrix[tour[i], tour[i + 1]] for i in range(self.num_cities - 1))
        cost += self.distance_matrix[tour[-1], tour[0]]  # Return to start
        return cost

    def find_optimal_tour(self, max_iterations=100000):
        best_cost = float('inf')
        best_tour = None
        current_tour = np.random.permutation(self.num_cities)
        for _ in range(max_iterations):
            i, j = np.random.choice(self.num_cities, 2, replace=False)
            current_tour[i], current_tour[j] = current_tour[j], current_tour[i]
            current_cost = self.calculate_tour_cost(current_tour)
            if current_cost < best_cost:
                best_cost = current_cost
                best_tour = current_tour.copy()
        return best_tour, best_cost

# Calculate the distance matrix
distance_matrix = compute_distance_matrix(city_names)

# Create an instance of the TSP solver
tsp_solver = TravelingSalesmanSolver(city_names, distance_matrix)

# Solve the TSP
optimal_tour, minimum_cost = tsp_solver.find_optimal_tour()

# Display the optimal tour and its cost
print("Optimal Tour:")
for i, city_index in enumerate(optimal_tour):
    if i == 0:
        print(f"Start from {city_names[city_index]}")
    else:
        print(f"Go to {city_names[city_index]}")
print(f"Return to {city_names[optimal_tour[0]]}")
print("Minimum Path Cost:", minimum_cost)

# Visualize the cities and the optimal tour
plt.figure(figsize=(10, 8))
for city in city_names:
    plt.scatter(*city_coordinates[city], color='green')
    plt.text(*city_coordinates[city], city, ha='center', va='center', fontsize=12)
for i in range(len(optimal_tour) - 1):
    plt.plot([city_coordinates[city_names[optimal_tour[i]]][0], city_coordinates[city_names[optimal_tour[i + 1]]][0]],
             [city_coordinates[city_names[optimal_tour[i]]][1], city_coordinates[city_names[optimal_tour[i + 1]]][1]],
             color='blue')
plt.plot([city_coordinates[city_names[optimal_tour[-1]]][0], city_coordinates[city_names[optimal_tour[0]]][0]],
         [city_coordinates[city_names[optimal_tour[-1]]][1], city_coordinates[city_names[optimal_tour[0]]][1]],
         color='blue')

# Annotate the path costs
for i in range(len(optimal_tour) - 1):
    mid_x = (city_coordinates[city_names[optimal_tour[i]]][0] + city_coordinates[city_names[optimal_tour[i + 1]]][0]) / 2
    mid_y = (city_coordinates[city_names[optimal_tour[i]]][1] + city_coordinates[city_names[optimal_tour[i + 1]]][1]) / 2
    plt.text(mid_x, mid_y, f'{distance_matrix[optimal_tour[i], optimal_tour[i + 1]]:.2f}', color='red')

plt.title("Traveling Salesman Problem (TSP)")
plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")
plt.grid(True)
plt.show()
