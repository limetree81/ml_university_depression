import numpy as np
import matplotlib.pyplot as plt
from minepy import MINE
from scipy.stats import pearsonr

# 1. Generate Non-Linear Data (Sine Wave)
np.random.seed(42)
x = np.linspace(0, 2 * np.pi, 500)
y = np.sin(x) + np.random.normal(0, 0.1, 500)

# theta = np.linspace(0, 2 * np.pi, 500)
# x = np.cos(theta) + np.random.normal(0, 0.1, 500)
# y = np.sin(theta) + np.random.normal(0, 0.1, 500)

# 2. Calculate Scores
# Pearson
pearson_r, _ = pearsonr(x, y)

# MIC
mine = MINE(alpha=0.6, c=15)
mine.compute_score(x, y)
mic_score = mine.mic()

# 3. Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Plot 1: How Pearson Sees the Data ---
axes[0].scatter(x, y, alpha=0.4, color='gray', label='Data Points')
# Draw a linear regression line
m, b = np.polyfit(x, y, 1)
axes[0].plot(x, m*x + b, color='red', linewidth=3, label=f'Best Fit Line (r={pearson_r:.2f})')
#axes[0].set_title(f"Pearson Correlation: {pearson_r:.2f}\n(Fails to see the relationship)", fontsize=14)
axes[0].set_title(f"Pearson Correlation: {pearson_r:.2f}", fontsize=14)
axes[0].legend()
axes[0].set_xlabel("X")
axes[0].set_ylabel("Y")

# --- Plot 2: How MIC Sees the Data (Grid Optimization) ---
axes[1].scatter(x, y, alpha=0.4, color='gray')

# Simulate a "Maximized Grid" that MIC might find
# MIC searches for a grid where cells are either full or empty (low entropy)
x_ticks = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
y_ticks = [-1.2, 0, 1.2]
# x_ticks = [-1.2, -0.4, 0.4 ,1.2]
# y_ticks = [-1.2, -0.4, 0.4 ,1.2]

# Draw the grid lines
for xt in x_ticks:
    axes[1].axvline(xt, color='blue', linestyle='--', alpha=0.6)
for yt in y_ticks:
    axes[1].axhline(yt, color='blue', linestyle='--', alpha=0.6)

# Highlight a "Success" cell
rect = plt.Rectangle((0, 0), np.pi/2, 1.2, color='green', alpha=0.2, label='High Information Cell')
axes[1].add_patch(rect)

# for i in [(-1.2, -1.2), (-1.2, -0.4), (-1.2, 0.4), (-0.4, 0.4), (0.4, 0.4), (0.4, -0.4), (0.4, -1.2), (-0.4, -1.2)]:
#     rect = plt.Rectangle(i, 0.8, 0.8, color='green', alpha=0.2)
#     axes[1].add_patch(rect)

rect = plt.Rectangle((0,0), 0,0, color='green', alpha=0.2, label='High Information Cell')
axes[1].add_patch(rect)
#axes[1].set_title(f"MIC: {mic_score:.2f}\n(Captures pattern via Grid Partitioning)", fontsize=14)
axes[1].set_title(f"MIC: {mic_score:.2f}", fontsize=14)
axes[1].set_xlabel("X (Partitioned into bins)")
axes[1].set_ylabel("Y (Partitioned into bins)")
axes[1].legend()

plt.tight_layout()
plt.show()

print(f"Pearson Correlation: {pearson_r:.4f}")
print(f"Maximal Information Coefficient (MIC): {mic_score:.4f}")