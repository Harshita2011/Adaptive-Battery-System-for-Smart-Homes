import time
import pandas as pd
import numpy as np
from battery_sim import BatterySimulator
from fuzzy_governor import run_governor_test
from ga_optimizer import ga_instance

# A list to store automated decisions for the UI to display later
action_log = []

def autonomous_rolling_horizon_loop():
    sim = BatterySimulator()
    last_safe_limit = 100.0  # Initial state

    while True:
        # 1. Get Live Fluctuating Data (t)
        load = 15.0 + np.random.normal(0, 2)
        state = sim.simulate_step(load)
        
        # 2. Perception & Governance (Automated)
        # Assuming RNN SOH prediction logic is here
        soh_pred = 0.999 # Placeholder for brevity
        current_safe_limit = run_governor_test(soh_pred, state['temp'])
        
        # 3. ROLLING HORIZON TRIGGER:
        # Re-run GA only if the safe limit has changed significantly (>5%)
        # This prevents the system from wasting CPU on tiny fluctuations
        if abs(current_safe_limit - last_safe_limit) > 5.0:
            ga_instance.run()
            best_schedule, _, _ = ga_instance.best_solution()
            
            # 4. Log the Automated Action
            action_entry = {
                "Timestamp": time.strftime('%H:%M:%S'),
                "Event": "Safety Throttling / Re-optimization",
                "Temp": f"{state['temp']:.1f}°C",
                "New_Limit": f"{current_safe_limit:.1f}%",
                "Action": f"Rescheduled EV to {best_schedule[0]:.1f}h"
            }
            action_log.append(action_entry)
            last_safe_limit = current_safe_limit
            
            print(f"🤖 AUTO-ACTION: {action_entry['Action']} due to {action_entry['Temp']}")

        time.sleep(5) # The system 'breathes' and monitors every 5 seconds