import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_results():
    # Load history
    df = pd.read_csv('data/battery_history.csv').head(1440) # One full day
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Temperature
    color = 'tab:red'
    ax1.set_xlabel('Time (Minutes)')
    ax1.set_ylabel('Temperature (°C)', color=color)
    ax1.plot(df['step'], df['temp'], color=color, alpha=0.6, label='Battery Temp')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axhline(y=45, color='black', linestyle='--', label='Fatigue Threshold')

    # Plot SOH on secondary axis
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('State of Health (SOH)', color=color)
    ax2.plot(df['step'], df['soh'], color=color, linewidth=2, label='SOH (Health)')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Advanced Battery Fatigue & Health Tracking')
    fig.tight_layout()
    plt.savefig('notebooks/final_performance_plot.png')
    print("📈 Performance plot saved to notebooks/final_performance_plot.png")
    plt.show()

if __name__ == "__main__":
    plot_results()