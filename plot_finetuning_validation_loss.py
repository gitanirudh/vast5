#!/usr/bin/env python3
"""
plot_finetuning_validation_loss.py
Generate validation loss evolution plot for fine-tuning (like Figure 3 in the paper)
Uses per-epoch logs from multiple runs to show mean ± std
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
RESULTS_DIR = Path("results_5epoch_test")
OUTPUT_FILE = RESULTS_DIR / "finetuning_validation_loss_evolution.png"

MODELS = ["roberta-base", "microsoft__deberta-v3-large", "answerdotai__ModernBERT-large"]

print("="*80)
print("CREATING FINE-TUNING VALIDATION LOSS EVOLUTION PLOT")
print("="*80)

# Load per-epoch logs
model_epoch_data = {}

for model_name in MODELS:
    epoch_log_file = RESULTS_DIR / f"{model_name}_per_epoch_logs.csv"
    
    if epoch_log_file.exists():
        print(f"\nReading: {epoch_log_file}")
        df = pd.read_csv(epoch_log_file)
        
        # Group by epoch and calculate mean ± std
        epoch_stats = df.groupby('epoch').agg({
            'val_loss': ['mean', 'std'],
            'train_loss': ['mean', 'std'],
            'f1_score': ['mean', 'std']
        }).reset_index()
        
        model_epoch_data[model_name] = epoch_stats
        print(f"  Loaded {len(df)} epoch records from {df['run'].nunique()} runs")
        print(f"  Epochs: {epoch_stats['epoch'].min()} to {epoch_stats['epoch'].max()}")
    else:
        print(f"\nWARNING: {epoch_log_file} not found!")

if not model_epoch_data:
    print("\nERROR: No per-epoch log files found!")
    print("Make sure you've run fine-tuning with the modified script.")
    exit(1)

# Create plot matching paper's Figure 3
fig, ax = plt.subplots(figsize=(12, 8))

colors = {
    "roberta-base": "#E74C3C",
    "microsoft__deberta-v3-large": "#3498DB", 
    "answerdotai__ModernBERT-large": "#2ECC71"
}

labels_clean = {
    "roberta-base": "RoBERTa-base",
    "microsoft__deberta-v3-large": "DeBERTa-v3-large",
    "answerdotai__ModernBERT-large": "ModernBERT-large"
}

markers = {
    "roberta-base": 'o',
    "microsoft__deberta-v3-large": 's',
    "answerdotai__ModernBERT-large": '^'
}

for model_name, epoch_stats in model_epoch_data.items():
    epochs = epoch_stats['epoch'].values
    val_loss_mean = epoch_stats['val_loss']['mean'].values
    val_loss_std = epoch_stats['val_loss']['std'].values
    
    # Plot with error bars (like the paper)
    ax.errorbar(
        epochs, 
        val_loss_mean, 
        yerr=val_loss_std,
        label=labels_clean[model_name],
        color=colors[model_name],
        marker=markers[model_name],
        markersize=8,
        linewidth=2.5,
        capsize=5,
        capthick=2,
        alpha=0.85,
        markeredgewidth=2,
        markeredgecolor='white'
    )

# Formatting to match paper style
ax.set_xlabel('Epoch', fontsize=16, fontweight='bold')
ax.set_ylabel('Validation Loss', fontsize=16, fontweight='bold')
ax.set_title('Evolution of the Validation Loss in Function of the\nNumber of Fine-tuning Epochs', 
             fontsize=14, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8, color='gray')
ax.legend(fontsize=13, loc='best', framealpha=0.95, edgecolor='black',
          fancybox=True, shadow=True)
ax.tick_params(axis='both', labelsize=12)

# Set integer x-ticks
ax.set_xticks(range(1, int(epochs.max()) + 1))

# Add subtle background
ax.set_facecolor('#F8F9FA')

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
print(f"\n{'='*80}")
print(f"Validation loss evolution plot saved to: {OUTPUT_FILE}")
print(f"{'='*80}\n")

plt.close()

# Print summary statistics
print("\nValidation Loss Summary (across all runs):")
print("="*80)
print(f"{'Model':<25} {'Epoch 1':>12} {'Final Epoch':>15} {'Min Loss':>12}")
print("-"*80)

for model_name, epoch_stats in model_epoch_data.items():
    first_loss = epoch_stats['val_loss']['mean'].iloc[0]
    final_loss = epoch_stats['val_loss']['mean'].iloc[-1]
    min_loss = epoch_stats['val_loss']['mean'].min()
    min_epoch = epoch_stats.loc[epoch_stats['val_loss']['mean'].idxmin(), 'epoch']
    
    label = labels_clean[model_name]
    print(f"{label:<25} {first_loss:>12.4f} {final_loss:>15.4f} {min_loss:>12.4f} (epoch {int(min_epoch)})")

print("\n" + "="*80)
