#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import os

# 测试数据
iterations = list(range(20))
times = [
    8.65744, 3.31813, 4.44795, 4.3248, 3.39773, 4.42127, 36.5938, 3.29367, 
    3.211, 8.69981, 5.68893, 12.2961, 4.29252, 12.2084, 3.31169, 3.21841, 
    4.34098, 4.24085, 7.74172, 7.31898
]

# 计算统计数据
min_time = min(times)
max_time = max(times)
mean_time = np.mean(times)
median_time = np.median(times)
stddev = np.std(times)

# 排除第一次运行的统计数据
times_without_first = times[1:]
iterations_without_first = iterations[1:]
mean_without_first = np.mean(times_without_first)

# 创建图表
plt.figure(figsize=(12, 8))

# 绘制主图表
plt.subplot(2, 1, 1)
plt.plot(iterations, times, 'o-', color='blue', label='Bootstrap Time')
plt.axhline(y=mean_time, color='r', linestyle='--', label=f'Mean: {mean_time:.2f} ms')
plt.axhline(y=median_time, color='g', linestyle='--', label=f'Median: {median_time:.2f} ms')
plt.fill_between(iterations, mean_time - stddev, mean_time + stddev, alpha=0.2, color='r', label=f'Std Dev: {stddev:.2f} ms')

plt.title('mscclpp Bootstrap Performance (8 GPUs, Single Node)')
plt.xlabel('Iteration')
plt.ylabel('Time (ms)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# 绘制箱线图
plt.subplot(2, 1, 2)
plt.boxplot([times, times_without_first], labels=['All Runs', 'Excluding First Run'])
plt.title('Bootstrap Time Distribution')
plt.ylabel('Time (ms)')
plt.grid(True, linestyle='--', alpha=0.7)

# 添加统计信息文本框
stats_text = f"""
Statistics:
  Min Time: {min_time:.2f} ms
  Max Time: {max_time:.2f} ms
  Mean Time (All): {mean_time:.2f} ms
  Mean Time (Excl. First): {mean_without_first:.2f} ms
  Median Time: {median_time:.2f} ms
  Std Dev: {stddev:.2f} ms
"""
plt.figtext(0.15, 0.01, stats_text, fontsize=10, 
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig('bootstrap_performance.png', dpi=300, bbox_inches='tight')
print(f"图表已保存为 {os.path.abspath('bootstrap_performance.png')}")

# 创建直方图
plt.figure(figsize=(10, 6))
plt.hist(times, bins=10, alpha=0.7, color='blue', edgecolor='black')
plt.axvline(x=mean_time, color='r', linestyle='--', label=f'Mean: {mean_time:.2f} ms')
plt.axvline(x=median_time, color='g', linestyle='--', label=f'Median: {median_time:.2f} ms')

plt.title('Distribution of Bootstrap Times')
plt.xlabel('Time (ms)')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.tight_layout()
plt.savefig('bootstrap_histogram.png', dpi=300)
print(f"直方图已保存为 {os.path.abspath('bootstrap_histogram.png')}")
