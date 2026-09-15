import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

def plot_evaluation_dashboard(json_path):
    # 1. Load and parse the JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    df = pd.DataFrame(data['per_day'])
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter for data starting only from January 2025
    df = df[df['date'] >= '2025-01-01']
    df = df.sort_values('date')
    
    # 2. Setup the figure
    fig, axes = plt.subplots(1, 3, figsize=(20, 5), constrained_layout=True)
    
    # --- PLOT 1: Error Time Series ---
    ax1 = axes[0]
    ax1.plot(df['date'], df['crps'], marker='o', markersize=5, linestyle='-', color='tab:red', label='CRPS')
    ax1.plot(df['date'], df['rmse_mean'], marker='s', markersize=4, linestyle='--', color='tab:blue', alpha=0.7, label='Ensemble RMSE')
    
    # Apply log scale to handle the massive spikes on extreme weather days
    ax1.set_yscale('log')
    ax1.set_title('Daily Error Metrics', fontweight='bold')
    ax1.set_ylabel('Error (mm/day) [Log Scale]')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend()
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    # --- PLOT 2: Spread-Skill Reliability ---
    ax2 = axes[1]
    scatter = ax2.scatter(df['spread_skill'], df['crps'], 
                          c=df['coverage_90'], cmap='viridis', 
                          edgecolor='k', alpha=0.8, s=60)
    
    ax2.set_title('Spread/Skill vs. CRPS', fontweight='bold')
    ax2.set_xlabel('Spread / Skill Ratio')
    ax2.set_ylabel('CRPS (mm/day)')
    ax2.axvline(1.0, color='k', linestyle='--', alpha=0.5, label='Perfect Spread Calibration')
    
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('90% Interval Coverage')
    ax2.legend()
    ax2.grid(True, alpha=0.5)

    # --- PLOT 3: Calibration / Coverage Stability ---
    ax3 = axes[2]
    ax3.plot(df['date'], df['coverage_50'], label='50% Interval (Target: 0.5)', marker='.', color='mediumseagreen')
    ax3.plot(df['date'], df['coverage_80'], label='80% Interval (Target: 0.8)', marker='.', color='dodgerblue')
    ax3.plot(df['date'], df['coverage_90'], label='90% Interval (Target: 0.9)', marker='.', color='darkviolet')
    
    ax3.set_title('Ensemble Coverage Calibration', fontweight='bold')
    ax3.set_ylabel('Proportion of Truth Captured')
    
    # Add target reference lines
    ax3.axhline(0.5, color='mediumseagreen', linestyle=':', alpha=0.8)
    ax3.axhline(0.8, color='dodgerblue', linestyle=':', alpha=0.8)
    ax3.axhline(0.9, color='darkviolet', linestyle=':', alpha=0.8)
    
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.5)

    # 3. Save the dashboard
    out_path = Path(json_path).parent / 'cfm_evaluation_dashboard.png'
    fig.suptitle('Continuous Flow Matching (CFM) ODE Evaluation', fontsize=14, fontweight='bold')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Dashboard successfully saved to: {out_path}")

if __name__ == "__main__":
    json_path = "sequential_plots_cfm_ode_v3_no_flow/results.json"
    plot_evaluation_dashboard(json_path)