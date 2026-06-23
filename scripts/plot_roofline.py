import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# =============================================================================
# 0. PATH RESOLUTION (Adapting to your project structure)
# =============================================================================
# Get the absolute path to the directory where this script lives (.../tensor_engine/scripts)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up one level to the project root (.../tensor_engine)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Define exact paths for the CSV input and PNG output
csv_path = os.path.join(PROJECT_ROOT, 'benchmarks', 'gemm_roofline.csv')
png_path = os.path.join(PROJECT_ROOT, 'benchmarks', 'roofline.png')

# =============================================================================
# 1. HARDWARE SPECIFICATIONS (Intel Core i7-1165G7)
# =============================================================================
PEAK_GFLOPS = 150.4  
BANDWIDTH = 51.2     
CACHE_LINE_BYTES = 64 

# =============================================================================
# 2. DATA LOADING & CLEANING
# =============================================================================
if not os.path.exists(csv_path):
    print(f"Error: Could not find {csv_path}")
    print(f"Make sure your CSV is located at: {csv_path}")
    exit(1)

df = pd.read_csv(csv_path, thousands=',')


# =============================================================================
# 3. MATHEMATICAL CALCULATIONS (Operational Intensity)
# =============================================================================
# Operational Intensity = Total FLOPs / Total Bytes fetched from RAM
# For Matrix Multiplication (C = A * B), total FLOPs = 2 * N^3
df['total_flops'] = 2 * (df['N'] ** 3)

# Total Bytes from RAM = LLC Misses * 64 bytes (size of one cache line)
df['total_bytes_from_ram'] = df['LLC_misses'] * CACHE_LINE_BYTES

# Operational Intensity (FLOPs per byte)
df['intensity'] = df['total_flops'] / df['total_bytes_from_ram']

# Ensure GFLOPS is numeric
df['gflops'] = pd.to_numeric(df['gflops'])


# =============================================================================
# 4. DRAWING THE ROOFLINE MODEL
# =============================================================================
plt.figure(figsize=(10, 7))

# Create an X-axis (Operational Intensity) ranging from 1 to 10,000 using log scale
I_range = np.logspace(0, 4, 500)

# The Memory Roof: Performance is limited by how fast RAM can feed the CPU
mem_roof = BANDWIDTH * I_range

# The Compute Roof: Performance is limited by maximum CPU Math speed
comp_roof = PEAK_GFLOPS * np.ones_like(I_range)

# The actual roofline is the bottleneck (the minimum of memory OR compute limit)
actual_roof = np.minimum(mem_roof, comp_roof)

# Plot the lines
plt.loglog(I_range, mem_roof, 'b--', alpha=0.5, label=f'Memory Bandwidth Limit ({BANDWIDTH} GB/s)')
plt.loglog(I_range, comp_roof, 'r--', alpha=0.5, label=f'Compute Limit ({PEAK_GFLOPS} GFLOPS)')
plt.loglog(I_range, actual_roof, 'k-', linewidth=2.5, label='Hardware Roofline')


# =============================================================================
# 5. PLOTTING YOUR BENCHMARK DATA
# =============================================================================
# We will use different colors/markers for Naive vs Tiled
implementations = df['impl'].unique()
colors = {'naive': 'red', 'tiled': 'green'}
markers = {'naive': 'o', 'tiled': 's'} # 'o' = circle, 's' = square

for impl in implementations:
    subset = df[df['impl'] == impl]
    plt.scatter(
        subset['intensity'], 
        subset['gflops'], 
        color=colors.get(impl, 'blue'), 
        marker=markers.get(impl, '^'),
        s=100, # Size of the dots
        zorder=5, # Ensure dots render on top of the lines
        label=f'{impl.capitalize()} GEMM'
    )
    
    # Add text labels next to the dots so we know which N size is which
    for _, row in subset.iterrows():
        plt.text(
            row['intensity'] * 1.15,  # Shift text slightly to the right
            row['gflops'] * 0.9,      # Shift text slightly down
            f"N={int(row['N'])}", 
            fontsize=9
        )

# =============================================================================
# 6. GRAPH FORMATTING & SAVING
# =============================================================================
plt.xlabel('Operational Intensity (FLOPs / Byte fetched from RAM)', fontsize=12)
plt.ylabel('Performance (GFLOPS)', fontsize=12)
plt.title('Roofline Model: Intel Core i7-1165G7 (Single-Thread)', fontsize=14, fontweight='bold')

# The Ridge Point is where the system shifts from Memory-Bound to Compute-Bound
ridge_point_intensity = PEAK_GFLOPS / BANDWIDTH
plt.axvline(x=ridge_point_intensity, color='gray', linestyle=':', label=f'Ridge Point ({ridge_point_intensity:.2f})')

plt.legend(loc='lower right', framealpha=0.9)
plt.grid(True, which='both', linestyle='--', alpha=0.4)

# Set axis limits so the graph focuses nicely on the data
plt.xlim(1, 3000)
plt.ylim(0.1, max(PEAK_GFLOPS * 2, df['gflops'].max() * 2))

plt.tight_layout()
plt.savefig(png_path, dpi=200)
print("Successfully generated and saved 'benchmarks/roofline.png'!")