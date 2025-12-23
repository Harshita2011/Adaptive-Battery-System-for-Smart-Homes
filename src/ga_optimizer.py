import pygad
import numpy as np

# Research Data: Typical Smart Home Appliance Loads (kW)
# These represent the "genes" the GA will schedule
appliance_loads = [7.0, 3.5, 2.0] # EV Charger, AC, Dishwasher

# Simulated Hourly Grid Prices ($/kWh) - Peak at 18:00 (6 PM)
grid_prices = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.3, 0.2, 0.2, 0.2, 
               0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.7, 0.4, 0.2, 0.1, 0.1]

def fitness_func(ga_instance, solution, solution_idx):
    """
    Advanced Fitness Function:
    Minimizes Cost + Penalizes Fatigue (Exceeding Fuzzy Limit)
    """
    total_cost = 0
    penalty = 0
    # This limit comes from your Fuzzy Governor (39.04% of a 10kW system = 3.9kW)
    safe_limit = 3.9 
    
    for i, start_time in enumerate(solution):
        hour = int(start_time) % 24
        load = appliance_loads[i]
        
        # 1. Calculate Monetary Cost
        total_cost += load * grid_prices[hour]
        
        # 2. Apply Fatigue Penalty if appliance exceeds Governor's ceiling
        if load > safe_limit:
            # Quadratic penalty for "stressing" the battery
            penalty += (load - safe_limit) ** 2 * 10 
            
    # GA maximizes fitness, so we return the inverse of (Cost + Penalty)
    return 1.0 / (total_cost + penalty + 0.001)

# Configure the Evolutionary Strategy
ga_instance = pygad.GA(num_generations=100,
                       num_parents_mating=5,
                       fitness_func=fitness_func,
                       sol_per_pop=20,
                       num_genes=len(appliance_loads),
                       init_range_low=0,
                       init_range_high=23,
                       mutation_percent_genes=10)

if __name__ == "__main__":
    print("🧬 GA: Searching for optimal low-fatigue schedule...")
    ga_instance.run()
    
    solution, fitness, idx = ga_instance.best_solution()
    print(f"✅ Optimized Schedule (Hours of Day): {np.round(solution, 1)}")
    print(f"💰 Predicted Energy Cost Index: {1/fitness:.2f}")