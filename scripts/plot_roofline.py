import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# =============================================================================
# 0. PATH RESOLUTION 
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
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
    exit(1)

df = pd.read_csv(csv_path, thousands=',')
df['LLC_misses'] = pd.to_numeric(df['LLC_misses'], errors='coerce')
df = df.dropna(subset=['LLC_misses']).copy()

# =============================================================================
# 3. MATHEMATICAL CALCULATIONS (Operational Intensity)
# =============================================================================
df['total_flops'] = 2 * (df['N'] ** 3)
df['total_bytes_from_ram'] = df['LLC_misses'] * CACHE_LINE_BYTES
df['intensity'] = df['total_flops'] / df['total_bytes_from_ram']
df['gflops'] = pd.to_numeric(df['gflops'])

# =============================================================================
# 4. DYNAMIC GRAPH LIMITS (Fixes the missing N=2048 point)
# =============================================================================
# Find the min/max of our actual data to set boundaries
min_intensity = df['intensity'].min()
max_intensity = df['intensity'].max()

# Pad the edges so points don't touch the borders of the image
axis_min_x = min_intensity * 0.2  # Go slightly lower than the lowest point
axis_max_x = max_intensity * 5.0  # Go slightly higher than the highest point

# Create X-axis points dynamically based on our new limits
I_range = np.logspace(np.log10(axis_min_x), np.log10(axis_max_x), 500)

# =============================================================================
# 5. DRAWING THE ROOFLINE MODEL
# =============================================================================
plt.figure(figsize=(10, 7))

mem_roof = BANDWIDTH * I_range
comp_roof = PEAK_GFLOPS * np.ones_like(I_range)
actual_roof = np.minimum(mem_roof, comp_roof)

plt.loglog(I_range, mem_roof, 'b--', alpha=0.5, label=f'Memory Bandwidth Limit ({BANDWIDTH} GB/s)')
plt.loglog(I_range, comp_roof, 'r--', alpha=0.5, label=f'Compute Limit ({PEAK_GFLOPS} GFLOPS)')
plt.loglog(I_range, actual_roof, 'k-', linewidth=2.5, label='Hardware Roofline')

# =============================================================================
# 6. PLOTTING THE POINTS & SMART LABELS
# =============================================================================
colors = {'naive': 'red', 'tiled': 'green'}
markers = {'naive': 'o', 'tiled': 's'}

for impl in df['impl'].unique():
    subset = df[df['impl'] == impl]
    
    # Plot the dots
    plt.scatter(
        subset['intensity'], subset['gflops'], 
        color=colors.get(impl, 'blue'), marker=markers.get(impl, 'o'),
        s=120, zorder=5, label=f'{impl.capitalize()} GEMM'
    )
    
    # Add smart labels
    for _, row in subset.iterrows():
        n_val = int(row['N'])
        
        # Default offset: 10 pixels right, 0 pixels up
        x_offset = 10
        y_offset = 0
        
        # Custom rules to prevent overlaps on specific clusters
        if impl == 'tiled':
            if n_val == 2048:
                x_offset = -15  # Push left
                y_offset = -15  # Push down
            elif n_val == 256:
                x_offset = 15   # Push right
                y_offset = 10   # Push up
            else:
                x_offset = 15   # Standard right shift for other tiled points
        elif impl == 'naive':
            if n_val == 256 or n_val == 512:
                y_offset = -15  # Push down so it doesn't crowd the dot
            
        # plt.annotate lets us use exact pixel offsets ('offset points')
        plt.annotate(
            f"N={n_val}",
            xy=(row['intensity'], row['gflops']),  # The exact coordinate of the dot
            xytext=(x_offset, y_offset),           # The pixel offset for the text
            textcoords='offset points',            # Tell matplotlib these are pixel offsets
            fontsize=9,
            ha='center' if x_offset < 0 else 'left', # Align text properly
            va='center'
        )# =============================================================================
# 7. GRAPH FORMATTING & SAVING
# =============================================================================
plt.xlabel('Operational Intensity (FLOPs / Byte fetched from RAM)', fontsize=12)
plt.ylabel('Performance (GFLOPS)', fontsize=12)
plt.title('Roofline Model: Intel Core i7-1165G7 (Single-Thread)', fontsize=14, fontweight='bold')

ridge_point_intensity = PEAK_GFLOPS / BANDWIDTH
plt.axvline(x=ridge_point_intensity, color='gray', linestyle=':', label=f'Ridge Point ({ridge_point_intensity:.2f})')

plt.legend(loc='lower right', framealpha=0.9)
plt.grid(True, which='both', linestyle='--', alpha=0.4)

# Apply our dynamic limits
plt.xlim(axis_min_x, axis_max_x)
plt.ylim(df['gflops'].min() * 0.3, PEAK_GFLOPS * 2) # Adjust Y-axis dynamically too

plt.tight_layout()
plt.savefig(png_path, dpi=200)
print(f"Successfully generated and saved '{png_path}'!")
