#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import os

# Create directory to save figures
os.makedirs('figures', exist_ok=True)

# Performance data
# Extracted from RESULTS.md
mscclpp_run1 = {
    'avg': 9.79,
    'min': 3.31,
    'max': 23.56,
    'stddev': 7.34
}

fast_bootstrap_run1 = {
    'avg': 1.29,
    'min': 1.26,
    'max': 1.47,
    'stddev': 0.07
}

mscclpp_run2 = {
    'avg': 6.48,
    'min': 3.24,
    'max': 13.97,
    'stddev': 4.04
}

fast_bootstrap_run2 = {
    'avg': 1.30,
    'min': 1.27,
    'max': 1.69,
    'stddev': 0.16
}

# Calculate averages
mscclpp_avg = {
    'avg': (mscclpp_run1['avg'] + mscclpp_run2['avg']) / 2,
    'min': (mscclpp_run1['min'] + mscclpp_run2['min']) / 2,
    'max': (mscclpp_run1['max'] + mscclpp_run2['max']) / 2,
    'stddev': (mscclpp_run1['stddev'] + mscclpp_run2['stddev']) / 2
}

fast_bootstrap_avg = {
    'avg': (fast_bootstrap_run1['avg'] + fast_bootstrap_run2['avg']) / 2,
    'min': (fast_bootstrap_run1['min'] + fast_bootstrap_run2['min']) / 2,
    'max': (fast_bootstrap_run1['max'] + fast_bootstrap_run2['max']) / 2,
    'stddev': (fast_bootstrap_run1['stddev'] + fast_bootstrap_run2['stddev']) / 2
}

# Calculate speedup
speedup = {
    'avg': mscclpp_avg['avg'] / fast_bootstrap_avg['avg'],
    'min': mscclpp_avg['min'] / fast_bootstrap_avg['min'],
    'max': mscclpp_avg['max'] / fast_bootstrap_avg['max'],
    'stddev': mscclpp_avg['stddev'] / fast_bootstrap_avg['stddev']
}

# Figure 1: Average Bootstrap Time Comparison
plt.figure(figsize=(10, 6))
labels = ['MSCCLPP Bootstrap', 'Fast Bootstrap']
avg_times = [mscclpp_avg['avg'], fast_bootstrap_avg['avg']]
colors = ['#3498db', '#e74c3c']

bars = plt.bar(labels, avg_times, color=colors, width=0.6)
plt.ylabel('Bootstrap Time (ms)', fontsize=14)
plt.title('Average Bootstrap Time Comparison', fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{height:.2f} ms', ha='center', va='bottom', fontsize=12)

# Add speedup annotation
plt.text(0.5, max(avg_times) * 0.5, 
         f'Speedup: {speedup["avg"]:.2f}x', 
         ha='center', va='center', 
         bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'),
         fontsize=14)

plt.tight_layout()
plt.savefig('figures/avg_time_comparison_en.png', dpi=300)

# Figure 2: Min, Avg, Max Bootstrap Time Comparison
plt.figure(figsize=(12, 7))
x = np.arange(3)
width = 0.35

metrics = ['Min Bootstrap Time', 'Avg Bootstrap Time', 'Max Bootstrap Time']
mscclpp_values = [mscclpp_avg['min'], mscclpp_avg['avg'], mscclpp_avg['max']]
fast_values = [fast_bootstrap_avg['min'], fast_bootstrap_avg['avg'], fast_bootstrap_avg['max']]

bars1 = plt.bar(x - width/2, mscclpp_values, width, label='MSCCLPP Bootstrap', color='#3498db')
bars2 = plt.bar(x + width/2, fast_values, width, label='Fast Bootstrap', color='#e74c3c')

plt.ylabel('Time (ms)', fontsize=14)
plt.title('Bootstrap Performance Metrics Comparison', fontsize=16)
plt.xticks(x, metrics, fontsize=12)
plt.legend(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f} ms', ha='center', va='bottom', fontsize=10)

add_labels(bars1)
add_labels(bars2)

# Add speedup
for i, (m, f) in enumerate(zip(mscclpp_values, fast_values)):
    speedup_val = m / f
    plt.text(i, max(m, f) * 0.5, 
             f'Speedup: {speedup_val:.2f}x', 
             ha='center', va='center', 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'),
             fontsize=10)

plt.tight_layout()
plt.savefig('figures/min_avg_max_comparison_en.png', dpi=300)

# Figure 3: Stability Comparison (Standard Deviation)
plt.figure(figsize=(10, 6))
labels = ['MSCCLPP Bootstrap', 'Fast Bootstrap']
stddev_values = [mscclpp_avg['stddev'], fast_bootstrap_avg['stddev']]

bars = plt.bar(labels, stddev_values, color=colors, width=0.6)
plt.ylabel('Standard Deviation (ms)', fontsize=14)
plt.title('Bootstrap Stability Comparison (Standard Deviation)', fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{height:.2f} ms', ha='center', va='bottom', fontsize=12)

# Add improvement ratio annotation
plt.text(0.5, max(stddev_values) * 0.5, 
         f'Improvement Ratio: {speedup["stddev"]:.2f}x', 
         ha='center', va='center', 
         bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'),
         fontsize=14)

plt.tight_layout()
plt.savefig('figures/stability_comparison_en.png', dpi=300)

# Figure 4: Radar Chart for Comprehensive Performance Comparison
plt.figure(figsize=(10, 8))
categories = ['Avg Bootstrap Time\n(lower is better)', 'Min Bootstrap Time\n(lower is better)', 
              'Max Bootstrap Time\n(lower is better)', 'Standard Deviation\n(lower is better)']
N = len(categories)

# Calculate angles
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # Close the loop

# Prepare data
# To make smaller values show as better performance, we take the inverse and normalize
max_avg = max(mscclpp_avg['avg'], fast_bootstrap_avg['avg'])
max_min = max(mscclpp_avg['min'], fast_bootstrap_avg['min'])
max_max = max(mscclpp_avg['max'], fast_bootstrap_avg['max'])
max_stddev = max(mscclpp_avg['stddev'], fast_bootstrap_avg['stddev'])

# Normalize data (values closer to 1 are better)
mscclpp_normalized = [
    1 - (mscclpp_avg['avg'] / max_avg),
    1 - (mscclpp_avg['min'] / max_min),
    1 - (mscclpp_avg['max'] / max_max),
    1 - (mscclpp_avg['stddev'] / max_stddev)
]
mscclpp_normalized += mscclpp_normalized[:1]  # Close the loop

fast_normalized = [
    1 - (fast_bootstrap_avg['avg'] / max_avg),
    1 - (fast_bootstrap_avg['min'] / max_min),
    1 - (fast_bootstrap_avg['max'] / max_max),
    1 - (fast_bootstrap_avg['stddev'] / max_stddev)
]
fast_normalized += fast_normalized[:1]  # Close the loop

# Draw radar chart
ax = plt.subplot(111, polar=True)
ax.plot(angles, mscclpp_normalized, 'o-', linewidth=2, label='MSCCLPP Bootstrap', color='#3498db')
ax.fill(angles, mscclpp_normalized, alpha=0.25, color='#3498db')
ax.plot(angles, fast_normalized, 'o-', linewidth=2, label='Fast Bootstrap', color='#e74c3c')
ax.fill(angles, fast_normalized, alpha=0.25, color='#e74c3c')

# Set radar chart properties
ax.set_thetagrids(np.degrees(angles[:-1]), categories)
ax.set_ylim(0, 1)
ax.grid(True)
ax.set_title('Comprehensive Bootstrap Performance Comparison', fontsize=16, y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

plt.tight_layout()
plt.savefig('figures/radar_comparison_en.png', dpi=300)

# Figure 5: Speedup Comparison
plt.figure(figsize=(10, 6))
metrics = ['Avg Bootstrap Time', 'Min Bootstrap Time', 'Max Bootstrap Time', 'Standard Deviation\n(Stability)']
speedup_values = [speedup['avg'], speedup['min'], speedup['max'], speedup['stddev']]

bars = plt.bar(metrics, speedup_values, color='#2ecc71', width=0.6)
plt.ylabel('Speedup Factor (x)', fontsize=14)
plt.title('Fast Bootstrap Performance Improvement over MSCCLPP', fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=0)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.3,
             f'{height:.2f}x', ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.savefig('figures/speedup_comparison_en.png', dpi=300)

print("Charts have been generated and saved to the figures directory")
