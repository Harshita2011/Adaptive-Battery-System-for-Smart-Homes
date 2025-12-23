import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def create_governor():
    # 1. Define Antecedents (Inputs) and Consequents (Outputs)
    # Predicted SoH from RNN (0 to 1)
    soh = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'soh')
    # Current Temperature (20 to 60 C)
    temp = ctrl.Antecedent(np.arange(20, 61, 1), 'temperature')
    # Output: Discharge Power Limit (0% to 100%)
    p_limit = ctrl.Consequent(np.arange(0, 101, 1), 'p_limit')

    # 2. Membership Functions (Research-standard Gaussian/Triangular)
    soh['poor'] = fuzz.trimf(soh.universe, [0, 0, 0.6])
    soh['fair'] = fuzz.trimf(soh.universe, [0.4, 0.7, 0.9])
    soh['good'] = fuzz.trimf(soh.universe, [0.8, 1, 1])

    temp['cool'] = fuzz.trimf(temp.universe, [20, 20, 35])
    temp['warm'] = fuzz.trimf(temp.universe, [30, 45, 55])
    temp['hot'] = fuzz.trimf(temp.universe, [50, 60, 60])

    p_limit['low'] = fuzz.trimf(p_limit.universe, [0, 0, 40])
    p_limit['med'] = fuzz.trimf(p_limit.universe, [30, 60, 80])
    p_limit['high'] = fuzz.trimf(p_limit.universe, [70, 100, 100])

    # 3. Research Rule Base: Prioritizing Longevity
    rule1 = ctrl.Rule(soh['poor'] | temp['hot'], p_limit['low'])
    rule2 = ctrl.Rule(soh['fair'] & temp['warm'], p_limit['med'])
    rule3 = ctrl.Rule(soh['good'] & temp['cool'], p_limit['high'])
    
    # Adaptive Rule: Even if SoH is good, if it's hot, throttle it
    rule4 = ctrl.Rule(soh['good'] & temp['hot'], p_limit['med'])

    # 4. Control System Creation
    governor_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
    return ctrl.ControlSystemSimulation(governor_ctrl)

def run_governor_test(current_soh, current_temp):
    gov = create_governor()
    
    # Advanced Research Fix: Clip inputs to ensure they stay within the fuzzy universe
    # This prevents KeyError if RNN predicts 1.0001 or temp is 60.1
    safe_soh = np.clip(current_soh, 0, 1.0)
    safe_temp = np.clip(current_temp, 20, 60)
    
    gov.input['soh'] = safe_soh
    gov.input['temperature'] = safe_temp
    
    try:
        gov.compute()
        return gov.output['p_limit']
    except Exception as e:
        # Fallback for Research Stability: If fuzzy logic fails, return a safe minimum
        print(f"⚠️ Fuzzy Inference Warning: {e}. Using safety fallback.")
        return 20.0  # 20% power limit as a safety default

if __name__ == "__main__":
    # Test Scenario: Battery is healthy (0.95) but getting hot (52C)
    limit = run_governor_test(0.95, 52)
    print(f"🛠️ Governor Decision: Discharge limit set to {limit:.2f}%")