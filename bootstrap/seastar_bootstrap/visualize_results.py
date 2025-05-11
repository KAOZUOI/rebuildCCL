#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import os

# 创建保存图表的目录
os.makedirs('figures', exist_ok=True)

# 性能数据
# 从RESULTS.md中提取的数据
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

# 计算平均值
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

# 计算加速比
speedup = {
    'avg': mscclpp_avg['avg'] / fast_bootstrap_avg['avg'],
    'min': mscclpp_avg['min'] / fast_bootstrap_avg['min'],
    'max': mscclpp_avg['max'] / fast_bootstrap_avg['max'],
    'stddev': mscclpp_avg['stddev'] / fast_bootstrap_avg['stddev']
}

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 图1：平均启动时间对比
plt.figure(figsize=(10, 6))
labels = ['MSCCLPP Bootstrap', 'Fast Bootstrap']
avg_times = [mscclpp_avg['avg'], fast_bootstrap_avg['avg']]
colors = ['#3498db', '#e74c3c']

bars = plt.bar(labels, avg_times, color=colors, width=0.6)
plt.ylabel('启动时间 (毫秒)', fontsize=14)
plt.title('Bootstrap平均启动时间对比', fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 添加数值标签
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{height:.2f} ms', ha='center', va='bottom', fontsize=12)

# 添加加速比标注
plt.text(0.5, max(avg_times) * 0.5, 
         f'加速比: {speedup["avg"]:.2f}x', 
         ha='center', va='center', 
         bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'),
         fontsize=14)

plt.tight_layout()
plt.savefig('figures/avg_time_comparison.png', dpi=300)

# 图2：最小、平均和最大启动时间对比
plt.figure(figsize=(12, 7))
x = np.arange(3)
width = 0.35

metrics = ['最小启动时间', '平均启动时间', '最大启动时间']
mscclpp_values = [mscclpp_avg['min'], mscclpp_avg['avg'], mscclpp_avg['max']]
fast_values = [fast_bootstrap_avg['min'], fast_bootstrap_avg['avg'], fast_bootstrap_avg['max']]

bars1 = plt.bar(x - width/2, mscclpp_values, width, label='MSCCLPP Bootstrap', color='#3498db')
bars2 = plt.bar(x + width/2, fast_values, width, label='Fast Bootstrap', color='#e74c3c')

plt.ylabel('时间 (毫秒)', fontsize=14)
plt.title('Bootstrap性能指标对比', fontsize=16)
plt.xticks(x, metrics, fontsize=12)
plt.legend(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 添加数值标签
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f} ms', ha='center', va='bottom', fontsize=10)

add_labels(bars1)
add_labels(bars2)

# 添加加速比
for i, (m, f) in enumerate(zip(mscclpp_values, fast_values)):
    speedup_val = m / f
    plt.text(i, max(m, f) * 0.5, 
             f'加速比: {speedup_val:.2f}x', 
             ha='center', va='center', 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'),
             fontsize=10)

plt.tight_layout()
plt.savefig('figures/min_avg_max_comparison.png', dpi=300)

# 图3：标准差对比（稳定性）
plt.figure(figsize=(10, 6))
labels = ['MSCCLPP Bootstrap', 'Fast Bootstrap']
stddev_values = [mscclpp_avg['stddev'], fast_bootstrap_avg['stddev']]

bars = plt.bar(labels, stddev_values, color=colors, width=0.6)
plt.ylabel('标准差 (毫秒)', fontsize=14)
plt.title('Bootstrap稳定性对比 (标准差)', fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 添加数值标签
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{height:.2f} ms', ha='center', va='bottom', fontsize=12)

# 添加改进比标注
plt.text(0.5, max(stddev_values) * 0.5, 
         f'改进比: {speedup["stddev"]:.2f}x', 
         ha='center', va='center', 
         bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'),
         fontsize=14)

plt.tight_layout()
plt.savefig('figures/stability_comparison.png', dpi=300)

# 图4：综合性能雷达图
plt.figure(figsize=(10, 8))
categories = ['平均启动时间\n(越低越好)', '最小启动时间\n(越低越好)', 
              '最大启动时间\n(越低越好)', '标准差\n(越低越好)']
N = len(categories)

# 计算角度
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # 闭合雷达图

# 准备数据
# 为了使较小的值显示为较好的性能，我们取倒数并归一化
max_avg = max(mscclpp_avg['avg'], fast_bootstrap_avg['avg'])
max_min = max(mscclpp_avg['min'], fast_bootstrap_avg['min'])
max_max = max(mscclpp_avg['max'], fast_bootstrap_avg['max'])
max_stddev = max(mscclpp_avg['stddev'], fast_bootstrap_avg['stddev'])

# 归一化数据（值越大越好）
mscclpp_normalized = [
    1 - (mscclpp_avg['avg'] / max_avg),
    1 - (mscclpp_avg['min'] / max_min),
    1 - (mscclpp_avg['max'] / max_max),
    1 - (mscclpp_avg['stddev'] / max_stddev)
]
mscclpp_normalized += mscclpp_normalized[:1]  # 闭合雷达图

fast_normalized = [
    1 - (fast_bootstrap_avg['avg'] / max_avg),
    1 - (fast_bootstrap_avg['min'] / max_min),
    1 - (fast_bootstrap_avg['max'] / max_max),
    1 - (fast_bootstrap_avg['stddev'] / max_stddev)
]
fast_normalized += fast_normalized[:1]  # 闭合雷达图

# 绘制雷达图
ax = plt.subplot(111, polar=True)
ax.plot(angles, mscclpp_normalized, 'o-', linewidth=2, label='MSCCLPP Bootstrap', color='#3498db')
ax.fill(angles, mscclpp_normalized, alpha=0.25, color='#3498db')
ax.plot(angles, fast_normalized, 'o-', linewidth=2, label='Fast Bootstrap', color='#e74c3c')
ax.fill(angles, fast_normalized, alpha=0.25, color='#e74c3c')

# 设置雷达图属性
ax.set_thetagrids(np.degrees(angles[:-1]), categories)
ax.set_ylim(0, 1)
ax.grid(True)
ax.set_title('Bootstrap性能综合对比', fontsize=16, y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

plt.tight_layout()
plt.savefig('figures/radar_comparison.png', dpi=300)

# 图5：加速比对比
plt.figure(figsize=(10, 6))
metrics = ['平均启动时间', '最小启动时间', '最大启动时间', '标准差(稳定性)']
speedup_values = [speedup['avg'], speedup['min'], speedup['max'], speedup['stddev']]

bars = plt.bar(metrics, speedup_values, color='#2ecc71', width=0.6)
plt.ylabel('加速比 (倍)', fontsize=14)
plt.title('Fast Bootstrap相对于MSCCLPP的性能提升', fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=0)

# 添加数值标签
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.3,
             f'{height:.2f}x', ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.savefig('figures/speedup_comparison.png', dpi=300)

print("图表已生成并保存到figures目录")
