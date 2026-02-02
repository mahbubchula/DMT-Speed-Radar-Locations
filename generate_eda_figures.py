"""
Generate EDA Figures for Q1 Journal Publication
===============================================
Publication-quality Exploratory Data Analysis figures.
300 DPI, Times New Roman font, NO titles (only axis labels).

Author: Mahbub Hassan
        Department of Civil Engineering, Chulalongkorn University
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from pathlib import Path
from scipy import stats

from config import (
    PROCESSED_DATA_DIR, FIGURES_DIR, REPORTS_DIR,
    COLORS, COLOR_PALETTE, FIGURE_SETTINGS, MPL_STYLE,
    SPEED_LIMIT_DEFAULT, VEHICLE_CLASSES
)

# Publication-quality settings
matplotlib.rcParams.update(MPL_STYLE)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# Custom color palette
PALETTE = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D', '#C73E1D']

print("=" * 60)
print("GENERATING EDA FIGURES FOR PUBLICATION")
print("=" * 60)

# Load data
print("\nLoading data...")
data = pd.read_csv(PROCESSED_DATA_DIR / 'processed_data.csv')
data['timestamp'] = pd.to_datetime(data['timestamp'], format='mixed')
print(f"Loaded {len(data):,} records")

# Create EDA figures directory
EDA_DIR = FIGURES_DIR / 'eda'
EDA_DIR.mkdir(exist_ok=True)

def save_fig(fig, name):
    """Save figure in PNG and PDF formats."""
    fig.savefig(EDA_DIR / f'{name}.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(EDA_DIR / f'{name}.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {name}.png")

# =============================================================================
# FIGURE 1: Speed Distribution Overall
# =============================================================================
print("\n[1/15] Speed Distribution...")

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(data['speed'], bins=80, color=PALETTE[0], edgecolor='white', alpha=0.8, density=True)

# Add KDE
from scipy.stats import gaussian_kde
kde = gaussian_kde(data['speed'].dropna())
x_range = np.linspace(data['speed'].min(), data['speed'].max(), 200)
ax.plot(x_range, kde(x_range), color=PALETTE[1], linewidth=2, label='KDE')

# Speed limit line
ax.axvline(x=SPEED_LIMIT_DEFAULT, color=PALETTE[4], linestyle='--', linewidth=2, label=f'Speed Limit ({SPEED_LIMIT_DEFAULT} km/h)')

ax.set_xlabel('Speed (km/h)')
ax.set_ylabel('Density')
ax.legend(frameon=False)
ax.grid(alpha=0.3)
save_fig(fig, 'eda_01_speed_distribution')

# =============================================================================
# FIGURE 2: Speed Distribution by Vehicle Type
# =============================================================================
print("[2/15] Speed by Vehicle Type...")

fig, ax = plt.subplots(figsize=(10, 6))
vehicle_order = data.groupby('vehicle_name')['speed'].median().sort_values().index

sns.boxplot(data=data.sample(min(100000, len(data))),
            x='vehicle_name', y='speed',
            order=vehicle_order,
            palette=PALETTE[:len(vehicle_order)], ax=ax)

ax.axhline(y=SPEED_LIMIT_DEFAULT, color=PALETTE[4], linestyle='--', linewidth=2, label=f'Speed Limit')
ax.set_xlabel('Vehicle Type')
ax.set_ylabel('Speed (km/h)')
ax.legend(frameon=False)
ax.grid(axis='y', alpha=0.3)
save_fig(fig, 'eda_02_speed_by_vehicle')

# =============================================================================
# FIGURE 3: Violin Plot - Speed by Vehicle Type
# =============================================================================
print("[3/15] Violin Plot...")

fig, ax = plt.subplots(figsize=(10, 6))
sns.violinplot(data=data.sample(min(50000, len(data))),
               x='vehicle_name', y='speed',
               order=vehicle_order,
               palette=PALETTE[:len(vehicle_order)], ax=ax)

ax.axhline(y=SPEED_LIMIT_DEFAULT, color=PALETTE[4], linestyle='--', linewidth=2)
ax.set_xlabel('Vehicle Type')
ax.set_ylabel('Speed (km/h)')
ax.grid(axis='y', alpha=0.3)
save_fig(fig, 'eda_03_violin_speed_vehicle')

# =============================================================================
# FIGURE 4: Hourly Speed Pattern
# =============================================================================
print("[4/15] Hourly Speed Pattern...")

hourly_stats = data.groupby('hour').agg({
    'speed': ['mean', 'std', 'median'],
    'is_violation': 'mean'
}).reset_index()
hourly_stats.columns = ['hour', 'mean_speed', 'std_speed', 'median_speed', 'violation_rate']

fig, ax1 = plt.subplots(figsize=(10, 6))

# Speed line with confidence interval
ax1.fill_between(hourly_stats['hour'],
                 hourly_stats['mean_speed'] - hourly_stats['std_speed'],
                 hourly_stats['mean_speed'] + hourly_stats['std_speed'],
                 alpha=0.2, color=PALETTE[0])
ax1.plot(hourly_stats['hour'], hourly_stats['mean_speed'],
         color=PALETTE[0], linewidth=2.5, marker='o', markersize=6, label='Mean Speed')
ax1.plot(hourly_stats['hour'], hourly_stats['median_speed'],
         color=PALETTE[1], linewidth=2, linestyle='--', marker='s', markersize=5, label='Median Speed')

ax1.axhline(y=SPEED_LIMIT_DEFAULT, color=PALETTE[4], linestyle=':', linewidth=2, label='Speed Limit')
ax1.set_xlabel('Hour of Day')
ax1.set_ylabel('Speed (km/h)')
ax1.set_xticks(range(0, 24, 2))
ax1.legend(loc='upper left', frameon=False)
ax1.grid(alpha=0.3)

# Secondary axis for violation rate
ax2 = ax1.twinx()
ax2.bar(hourly_stats['hour'], hourly_stats['violation_rate'] * 100,
        alpha=0.3, color=PALETTE[2], width=0.8, label='Violation Rate')
ax2.set_ylabel('Violation Rate (%)', color=PALETTE[2])
ax2.tick_params(axis='y', labelcolor=PALETTE[2])

save_fig(fig, 'eda_04_hourly_pattern')

# =============================================================================
# FIGURE 5: Daily Pattern (Day of Week)
# =============================================================================
print("[5/15] Daily Pattern...")

daily_stats = data.groupby('day_of_week').agg({
    'speed': ['mean', 'std', 'count'],
    'is_violation': 'mean'
}).reset_index()
daily_stats.columns = ['day', 'mean_speed', 'std_speed', 'count', 'violation_rate']
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Speed by day
axes[0].bar(daily_stats['day'], daily_stats['mean_speed'],
            yerr=daily_stats['std_speed']/10, capsize=5,
            color=PALETTE[:7], edgecolor='black', linewidth=0.5)
axes[0].axhline(y=SPEED_LIMIT_DEFAULT, color=PALETTE[4], linestyle='--', linewidth=2)
axes[0].set_xticks(range(7))
axes[0].set_xticklabels([d[:3] for d in days])
axes[0].set_xlabel('Day of Week')
axes[0].set_ylabel('Mean Speed (km/h)')
axes[0].grid(axis='y', alpha=0.3)

# Violation rate by day
colors = [PALETTE[4] if d >= 5 else PALETTE[0] for d in daily_stats['day']]
axes[1].bar(daily_stats['day'], daily_stats['violation_rate'] * 100,
            color=colors, edgecolor='black', linewidth=0.5)
axes[1].set_xticks(range(7))
axes[1].set_xticklabels([d[:3] for d in days])
axes[1].set_xlabel('Day of Week')
axes[1].set_ylabel('Violation Rate (%)')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
save_fig(fig, 'eda_05_daily_pattern')

# =============================================================================
# FIGURE 6: Heatmap - Hour vs Day
# =============================================================================
print("[6/15] Heatmap Hour vs Day...")

pivot_speed = data.pivot_table(values='speed', index='day_of_week',
                                columns='hour', aggfunc='mean')

fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(pivot_speed, cmap='YlOrRd', ax=ax,
            cbar_kws={'label': 'Mean Speed (km/h)'},
            annot=False, fmt='.0f')
ax.set_yticklabels([d[:3] for d in days], rotation=0)
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Day of Week')
save_fig(fig, 'eda_06_heatmap_hour_day')

# =============================================================================
# FIGURE 7: Violation Heatmap
# =============================================================================
print("[7/15] Violation Heatmap...")

pivot_violation = data.pivot_table(values='is_violation', index='day_of_week',
                                    columns='hour', aggfunc='mean') * 100

fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(pivot_violation, cmap='RdYlGn_r', ax=ax,
            cbar_kws={'label': 'Violation Rate (%)'},
            annot=False, vmin=90, vmax=100)
ax.set_yticklabels([d[:3] for d in days], rotation=0)
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Day of Week')
save_fig(fig, 'eda_07_violation_heatmap')

# =============================================================================
# FIGURE 8: Vehicle Type Composition
# =============================================================================
print("[8/15] Vehicle Composition...")

vehicle_counts = data['vehicle_name'].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pie chart
axes[0].pie(vehicle_counts.values, labels=vehicle_counts.index,
            autopct='%1.1f%%', colors=PALETTE[:len(vehicle_counts)],
            explode=[0.02]*len(vehicle_counts), startangle=90)

# Bar chart
axes[1].barh(vehicle_counts.index, vehicle_counts.values, color=PALETTE[:len(vehicle_counts)])
axes[1].set_xlabel('Number of Vehicles')
axes[1].set_ylabel('Vehicle Type')
for i, v in enumerate(vehicle_counts.values):
    axes[1].text(v + vehicle_counts.max()*0.01, i, f'{v:,}', va='center', fontsize=10)

plt.tight_layout()
save_fig(fig, 'eda_08_vehicle_composition')

# =============================================================================
# FIGURE 9: Speed by Location
# =============================================================================
print("[9/15] Speed by Location...")

location_stats = data.groupby('location_name').agg({
    'speed': ['mean', 'std', 'count'],
    'is_violation': 'mean'
}).reset_index()
location_stats.columns = ['location', 'mean_speed', 'std_speed', 'count', 'violation_rate']
location_stats = location_stats.sort_values('mean_speed', ascending=True)

fig, ax = plt.subplots(figsize=(10, 8))
colors = [PALETTE[4] if v > 0.99 else PALETTE[0] for v in location_stats['violation_rate']]
ax.barh(location_stats['location'], location_stats['mean_speed'],
        xerr=location_stats['std_speed']/5, capsize=3,
        color=colors, edgecolor='black', linewidth=0.5)
ax.axvline(x=SPEED_LIMIT_DEFAULT, color=PALETTE[4], linestyle='--', linewidth=2, label='Speed Limit')
ax.set_xlabel('Mean Speed (km/h)')
ax.set_ylabel('Location')
ax.legend(frameon=False)
ax.grid(axis='x', alpha=0.3)
save_fig(fig, 'eda_09_speed_by_location')

# =============================================================================
# FIGURE 10: Lane Analysis
# =============================================================================
print("[10/15] Lane Analysis...")

lane_stats = data.groupby('lane').agg({
    'speed': ['mean', 'std', 'count'],
    'is_violation': 'mean'
}).reset_index()
lane_stats.columns = ['lane', 'mean_speed', 'std_speed', 'count', 'violation_rate']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Speed by lane
axes[0].bar(lane_stats['lane'].astype(str), lane_stats['mean_speed'],
            yerr=lane_stats['std_speed']/10, capsize=5,
            color=PALETTE[:len(lane_stats)], edgecolor='black')
axes[0].axhline(y=SPEED_LIMIT_DEFAULT, color=PALETTE[4], linestyle='--', linewidth=2)
axes[0].set_xlabel('Lane')
axes[0].set_ylabel('Mean Speed (km/h)')
axes[0].grid(axis='y', alpha=0.3)

# Traffic volume by lane
axes[1].bar(lane_stats['lane'].astype(str), lane_stats['count'],
            color=PALETTE[:len(lane_stats)], edgecolor='black')
axes[1].set_xlabel('Lane')
axes[1].set_ylabel('Number of Vehicles')
axes[1].grid(axis='y', alpha=0.3)
for i, v in enumerate(lane_stats['count']):
    axes[1].text(i, v + lane_stats['count'].max()*0.01, f'{v:,}', ha='center', fontsize=9)

plt.tight_layout()
save_fig(fig, 'eda_10_lane_analysis')

# =============================================================================
# FIGURE 11: Monthly Trend
# =============================================================================
print("[11/15] Monthly Trend...")

monthly_stats = data.groupby('month').agg({
    'speed': ['mean', 'count'],
    'is_violation': 'mean'
}).reset_index()
monthly_stats.columns = ['month', 'mean_speed', 'count', 'violation_rate']
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.bar(monthly_stats['month'], monthly_stats['count'],
        color=PALETTE[0], alpha=0.7, label='Traffic Volume')
ax1.set_xlabel('Month')
ax1.set_ylabel('Number of Vehicles', color=PALETTE[0])
ax1.tick_params(axis='y', labelcolor=PALETTE[0])
ax1.set_xticks(monthly_stats['month'])
ax1.set_xticklabels([months[m-1] for m in monthly_stats['month']])

ax2 = ax1.twinx()
ax2.plot(monthly_stats['month'], monthly_stats['mean_speed'],
         color=PALETTE[1], linewidth=2.5, marker='o', markersize=8, label='Mean Speed')
ax2.axhline(y=SPEED_LIMIT_DEFAULT, color=PALETTE[4], linestyle='--', linewidth=2)
ax2.set_ylabel('Mean Speed (km/h)', color=PALETTE[1])
ax2.tick_params(axis='y', labelcolor=PALETTE[1])

fig.legend(loc='upper right', bbox_to_anchor=(0.88, 0.88), frameon=False)
save_fig(fig, 'eda_11_monthly_trend')

# =============================================================================
# FIGURE 12: Correlation Heatmap
# =============================================================================
print("[12/15] Correlation Heatmap...")

corr_cols = ['speed', 'hour', 'day_of_week', 'is_weekend', 'is_rush_hour',
             'vehicle_class', 'lane', 'is_violation', 'congestion_index']
available_cols = [c for c in corr_cols if c in data.columns]
corr_matrix = data[available_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, cmap='RdBu_r', center=0,
            annot=True, fmt='.2f', ax=ax, square=True,
            cbar_kws={'label': 'Correlation Coefficient'})
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
save_fig(fig, 'eda_12_correlation_heatmap')

# =============================================================================
# FIGURE 13: Speed Percentiles by Hour
# =============================================================================
print("[13/15] Speed Percentiles...")

percentiles = data.groupby('hour')['speed'].describe(percentiles=[.05, .25, .5, .75, .95])

fig, ax = plt.subplots(figsize=(12, 6))

ax.fill_between(percentiles.index, percentiles['5%'], percentiles['95%'],
                alpha=0.2, color=PALETTE[0], label='5th-95th percentile')
ax.fill_between(percentiles.index, percentiles['25%'], percentiles['75%'],
                alpha=0.4, color=PALETTE[0], label='25th-75th percentile')
ax.plot(percentiles.index, percentiles['50%'], color=PALETTE[1],
        linewidth=2.5, marker='o', label='Median')
ax.axhline(y=SPEED_LIMIT_DEFAULT, color=PALETTE[4], linestyle='--', linewidth=2, label='Speed Limit')

ax.set_xlabel('Hour of Day')
ax.set_ylabel('Speed (km/h)')
ax.set_xticks(range(0, 24, 2))
ax.legend(frameon=False, loc='upper right')
ax.grid(alpha=0.3)
save_fig(fig, 'eda_13_speed_percentiles')

# =============================================================================
# FIGURE 14: Rush Hour Comparison
# =============================================================================
print("[14/15] Rush Hour Comparison...")

data['period'] = 'Off-Peak'
data.loc[(data['hour'] >= 7) & (data['hour'] <= 9), 'period'] = 'Morning Rush'
data.loc[(data['hour'] >= 17) & (data['hour'] <= 19), 'period'] = 'Evening Rush'
data.loc[(data['hour'] >= 22) | (data['hour'] <= 5), 'period'] = 'Night'

period_order = ['Morning Rush', 'Off-Peak', 'Evening Rush', 'Night']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Box plot
sns.boxplot(data=data.sample(min(50000, len(data))),
            x='period', y='speed', order=period_order,
            palette=PALETTE[:4], ax=axes[0])
axes[0].axhline(y=SPEED_LIMIT_DEFAULT, color=PALETTE[4], linestyle='--', linewidth=2)
axes[0].set_xlabel('Time Period')
axes[0].set_ylabel('Speed (km/h)')
axes[0].grid(axis='y', alpha=0.3)

# Violation rate
period_violation = data.groupby('period')['is_violation'].mean().reindex(period_order) * 100
axes[1].bar(period_order, period_violation.values, color=PALETTE[:4], edgecolor='black')
axes[1].set_xlabel('Time Period')
axes[1].set_ylabel('Violation Rate (%)')
axes[1].grid(axis='y', alpha=0.3)
for i, v in enumerate(period_violation.values):
    axes[1].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
save_fig(fig, 'eda_14_rush_hour_comparison')

# =============================================================================
# FIGURE 15: CDF (Cumulative Distribution)
# =============================================================================
print("[15/15] Cumulative Distribution...")

fig, ax = plt.subplots(figsize=(10, 6))

# Overall CDF
sorted_speeds = np.sort(data['speed'])
cdf = np.arange(1, len(sorted_speeds) + 1) / len(sorted_speeds)

# Sample for plotting
sample_idx = np.linspace(0, len(sorted_speeds)-1, 1000).astype(int)
ax.plot(sorted_speeds[sample_idx], cdf[sample_idx],
        color=PALETTE[0], linewidth=2.5, label='All Vehicles')

# By vehicle type
for i, vtype in enumerate(data['vehicle_name'].unique()):
    vdata = data[data['vehicle_name'] == vtype]['speed']
    sorted_v = np.sort(vdata)
    cdf_v = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
    sample_idx_v = np.linspace(0, len(sorted_v)-1, min(500, len(sorted_v))).astype(int)
    ax.plot(sorted_v[sample_idx_v], cdf_v[sample_idx_v],
            linewidth=1.5, linestyle='--', alpha=0.7, label=vtype)

ax.axvline(x=SPEED_LIMIT_DEFAULT, color=PALETTE[4], linestyle=':', linewidth=2, label='Speed Limit')
ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)

ax.set_xlabel('Speed (km/h)')
ax.set_ylabel('Cumulative Probability')
ax.legend(frameon=False, loc='lower right')
ax.grid(alpha=0.3)
ax.set_xlim(0, 180)
save_fig(fig, 'eda_15_cumulative_distribution')

# =============================================================================
# GENERATE STATISTICAL SUMMARY TABLE
# =============================================================================
print("\n[BONUS] Generating statistical summary...")

# Overall statistics
overall_stats = {
    'Metric': [
        'Total Records',
        'Date Range',
        'Number of Locations',
        'Number of Vehicle Types',
        'Mean Speed (km/h)',
        'Median Speed (km/h)',
        'Std Dev Speed (km/h)',
        'Min Speed (km/h)',
        'Max Speed (km/h)',
        'Speed Limit (km/h)',
        'Overall Violation Rate (%)',
        'Peak Hour Violation Rate (%)',
        'Weekend Violation Rate (%)',
        'Weekday Violation Rate (%)'
    ],
    'Value': [
        f"{len(data):,}",
        f"{data['timestamp'].min().date()} to {data['timestamp'].max().date()}",
        f"{data['device_id'].nunique()}",
        f"{data['vehicle_name'].nunique()}",
        f"{data['speed'].mean():.2f}",
        f"{data['speed'].median():.2f}",
        f"{data['speed'].std():.2f}",
        f"{data['speed'].min():.2f}",
        f"{data['speed'].max():.2f}",
        f"{SPEED_LIMIT_DEFAULT}",
        f"{data['is_violation'].mean()*100:.2f}",
        f"{data[data['is_rush_hour']==1]['is_violation'].mean()*100:.2f}",
        f"{data[data['is_weekend']==1]['is_violation'].mean()*100:.2f}",
        f"{data[data['is_weekend']==0]['is_violation'].mean()*100:.2f}"
    ]
}
stats_df = pd.DataFrame(overall_stats)
stats_df.to_csv(REPORTS_DIR / 'eda_statistics_summary.csv', index=False)
print("  Saved: eda_statistics_summary.csv")

# Vehicle statistics
vehicle_stats = data.groupby('vehicle_name').agg({
    'speed': ['count', 'mean', 'std', 'median', 'min', 'max'],
    'is_violation': 'mean'
}).round(2)
vehicle_stats.columns = ['Count', 'Mean Speed', 'Std Speed', 'Median Speed', 'Min Speed', 'Max Speed', 'Violation Rate']
vehicle_stats['Violation Rate'] = (vehicle_stats['Violation Rate'] * 100).round(2)
vehicle_stats.to_csv(REPORTS_DIR / 'eda_vehicle_statistics.csv')
print("  Saved: eda_vehicle_statistics.csv")

# Hourly statistics
hourly_export = hourly_stats.copy()
hourly_export['violation_rate'] = (hourly_export['violation_rate'] * 100).round(2)
hourly_export.to_csv(REPORTS_DIR / 'eda_hourly_statistics.csv', index=False)
print("  Saved: eda_hourly_statistics.csv")

# Location statistics
location_export = location_stats.copy()
location_export['violation_rate'] = (location_export['violation_rate'] * 100).round(2)
location_export.to_csv(REPORTS_DIR / 'eda_location_statistics.csv', index=False)
print("  Saved: eda_location_statistics.csv")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("EDA FIGURE GENERATION COMPLETE")
print("=" * 60)

import os
eda_figures = sorted([f for f in os.listdir(EDA_DIR) if f.endswith('.png')])
print(f"\nGenerated {len(eda_figures)} EDA figures:")
for f in eda_figures:
    print(f"  - {f}")

print(f"\nFigures saved to: {EDA_DIR}")
print(f"Statistics saved to: {REPORTS_DIR}")
print("\nAll figures are publication-ready:")
print("  - 300 DPI resolution")
print("  - Times New Roman font")
print("  - No titles (axis labels only)")
print("  - PNG + PDF formats")
