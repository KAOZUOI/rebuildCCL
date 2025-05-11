#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os

# Create sample sizes (in bytes)
sizes = [8192, 49152, 262144, 1048576, 8388608, 67108864]

# Create sample data for AllReduce
allreduce_data = []
for size in sizes:
    # Calculate realistic bandwidth and latency values
    # Bandwidth increases with size but plateaus
    bandwidth = min(2000, 10 * np.log10(size))
    # Latency increases with size
    latency = 0.05 + size / (1024 * 1024 * 1024) * 10
    
    allreduce_data.append({
        'operation': 'allreduce',
        'size': size,
        'latency_ms': latency,
        'bandwidth_GBps': bandwidth
    })

# Create sample data for NCCL AlltoAll
nccl_alltoall_data = []
for size in sizes:
    # Calculate realistic bandwidth and latency values
    # AlltoAll typically has lower bandwidth than AllReduce
    bandwidth = min(1800, 9 * np.log10(size))
    # AlltoAll typically has higher latency than AllReduce
    latency = 0.07 + size / (1024 * 1024 * 1024) * 12
    
    nccl_alltoall_data.append({
        'implementation': 'nccl',
        'size': size,
        'latency_ms': latency,
        'bandwidth_GBps': bandwidth
    })

# Create sample data for custom AlltoAll
custom_alltoall_data = []
for size in sizes:
    # Calculate realistic bandwidth and latency values
    # Custom implementation has higher raw bandwidth but doesn't scale as well
    bandwidth = min(2200, 11 * np.log10(size))
    # Custom implementation has lower latency for small sizes but higher for large sizes
    latency = 0.03 + size / (1024 * 1024 * 1024) * 15
    
    custom_alltoall_data.append({
        'implementation': 'custom',
        'size': size,
        'latency_ms': latency,
        'bandwidth_GBps': bandwidth
    })

# Create DataFrames
allreduce_df = pd.DataFrame(allreduce_data)
alltoall_df = pd.DataFrame(nccl_alltoall_data + custom_alltoall_data)

# Save to CSV files
allreduce_df.to_csv('sample_allreduce_data.csv', index=False)
alltoall_df.to_csv('sample_alltoall_data.csv', index=False)

print(f"Sample data generated:")
print(f"AllReduce data saved to sample_allreduce_data.csv with {len(allreduce_df)} rows")
print(f"AlltoAll data saved to sample_alltoall_data.csv with {len(alltoall_df)} rows")
