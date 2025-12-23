import numpy as np
import pandas as pd
import os

class BatterySimulator:
    def __init__(self, capacity_ah=100, nominal_voltage=3.7):
        self.capacity = capacity_ah  # Amp-hours
        self.voltage = nominal_voltage
        self.soc = 1.0  # 100% charge
        self.soh = 1.0  # 100% health
        self.temp = 25.0 # Ambient starting temp

    def simulate_step(self, current_draw, ambient_temp=25.0):
        # 1. Thermal Model: Joule heating (I^2 * R)
        internal_resistance = 0.01 * (2 - self.soh) # Resistance increases as health drops
        heat_generated = (current_draw ** 2) * internal_resistance
        self.temp += heat_generated - (self.temp - ambient_temp) * 0.1
        
        # 2. Discharge Model
        self.soc -= (current_draw / self.capacity) / 60 # per minute step
        
        # 3. Fatigue Model: Accelerated aging if temp > 45C
        if self.temp > 45:
            self.soh -= 0.00001 * (self.temp - 45)
            
        return {"soc": max(0, self.soc), "temp": self.temp, "soh": self.soh}

def generate_training_data(rows=2000):
    sim = BatterySimulator()
    data = []
    for i in range(rows):
        # Simulate varying loads (0 to 20 Amps)
        load = np.random.uniform(0, 20)
        state = sim.simulate_step(load)
        data.append([i, load, state['temp'], state['soc'], state['soh']])
    
    df = pd.DataFrame(data, columns=['step', 'current', 'temp', 'soc', 'soh'])
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/battery_history.csv', index=False)
    print("✅ Created data/battery_history.csv with simulation data.")

if __name__ == "__main__":
    generate_training_data()