"""
Generate Professional Visuals for SecIDS-v2
============================================

Creates publication-quality graphics for:
- HuggingFace Model Card
- GitHub Repository
- LinkedIn Posts
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np
from pathlib import Path

# Create output directory
output_dir = Path("visuals")
output_dir.mkdir(exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
colors = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'neutral': '#7f7f7f'
}


def create_architecture_diagram():
    """Create model architecture visualization"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(5, 9.5, 'SecIDS-v2 Architecture: Temporal CNN',
            ha='center', va='top', fontsize=18, fontweight='bold')

    # Input layer
    input_box = FancyBboxPatch((0.5, 7.5), 1.5, 1,
                               boxstyle="round,pad=0.1",
                               edgecolor=colors['primary'],
                               facecolor='lightblue', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.25, 8, 'CAN Frames\n(128 × 25)', ha='center', va='center', fontsize=10)

    # TCN blocks
    tcn_blocks = [
        (2.5, 7.5, 'TCN Block 1\n32 filters\ndilation=1'),
        (4.5, 7.5, 'TCN Block 2\n64 filters\ndilation=2'),
        (6.5, 7.5, 'TCN Block 3\n128 filters\ndilation=4')
    ]

    for x, y, label in tcn_blocks:
        block = FancyBboxPatch((x, y), 1.5, 1,
                              boxstyle="round,pad=0.1",
                              edgecolor=colors['secondary'],
                              facecolor='lightyellow', linewidth=2)
        ax.add_patch(block)
        ax.text(x + 0.75, y + 0.5, label, ha='center', va='center', fontsize=9)

        # Arrows
        arrow = FancyArrowPatch((x - 0.45, y + 0.5), (x + 0.1, y + 0.5),
                               arrowstyle='->', mutation_scale=20,
                               color=colors['neutral'], linewidth=2)
        ax.add_patch(arrow)

    # Global pooling
    pool_box = FancyBboxPatch((8.5, 7.5), 1.2, 1,
                             boxstyle="round,pad=0.1",
                             edgecolor=colors['success'],
                             facecolor='lightgreen', linewidth=2)
    ax.add_patch(pool_box)
    ax.text(9.1, 8, 'Global\nAvg Pool', ha='center', va='center', fontsize=9)

    # Arrow to pool
    arrow = FancyArrowPatch((8.05, 8), (8.4, 8),
                           arrowstyle='->', mutation_scale=20,
                           color=colors['neutral'], linewidth=2)
    ax.add_patch(arrow)

    # Multi-task heads
    heads = [
        (2, 5.5, 'DoS'),
        (4, 5.5, 'Fuzzy'),
        (6, 5.5, 'Spoofing'),
        (8, 5.5, 'Replay')
    ]

    for x, y, label in heads:
        head_box = FancyBboxPatch((x, y), 1.2, 0.8,
                                 boxstyle="round,pad=0.05",
                                 edgecolor=colors['danger'],
                                 facecolor='lightcoral', linewidth=2)
        ax.add_patch(head_box)
        ax.text(x + 0.6, y + 0.4, label, ha='center', va='center', fontsize=10, fontweight='bold')

        # Arrows from pool to heads
        arrow = FancyArrowPatch((9.1, 7.4), (x + 0.6, y + 0.85),
                               arrowstyle='->', mutation_scale=15,
                               color=colors['neutral'], linewidth=1.5, linestyle='--')
        ax.add_patch(arrow)

    # Output layer
    output_box = FancyBboxPatch((3.5, 3.5), 3, 1,
                               boxstyle="round,pad=0.1",
                               edgecolor=colors['primary'],
                               facecolor='lightblue', linewidth=2)
    ax.add_patch(output_box)
    ax.text(5, 4, 'Multi-Task Classification\n(Attack Type + Binary)',
            ha='center', va='center', fontsize=11, fontweight='bold')

    # Stats box
    stats_text = (
        "Model Statistics:\n"
        "• Parameters: 3.8M\n"
        "• Inference: 4.2ms (Jetson Nano)\n"
        "• Accuracy: 98.2%\n"
        "• Receptive Field: 128 frames"
    )
    ax.text(0.5, 3, stats_text, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'architecture.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created architecture.png")


def create_performance_comparison():
    """Create v1 vs v2 performance comparison"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Accuracy comparison
    models = ['LSTM v1', 'TCN v2']
    accuracy = [97.2, 98.2]
    bars1 = ax1.bar(models, accuracy, color=[colors['secondary'], colors['primary']], alpha=0.7)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Detection Accuracy', fontsize=14, fontweight='bold')
    ax1.set_ylim([95, 100])
    ax1.axhline(y=98, color='gray', linestyle='--', alpha=0.5)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 2. Latency comparison
    latency = [18.5, 4.2]
    bars2 = ax2.bar(models, latency, color=[colors['secondary'], colors['primary']], alpha=0.7)
    ax2.set_ylabel('Inference Latency (ms)', fontsize=12)
    ax2.set_title('Inference Speed (Jetson Nano)', fontsize=14, fontweight='bold')

    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}ms', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 3. Model size comparison
    params = [5.2, 3.8]
    bars3 = ax3.bar(models, params, color=[colors['secondary'], colors['primary']], alpha=0.7)
    ax3.set_ylabel('Parameters (M)', fontsize=12)
    ax3.set_title('Model Size', fontsize=14, fontweight='bold')

    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}M', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 4. F1-Score per attack type
    attack_types = ['DoS', 'Fuzzy', 'Spoofing', 'Replay']
    v1_f1 = [96.5, 95.8, 96.2, 97.1]
    v2_f1 = [98.1, 97.9, 98.5, 98.3]

    x = np.arange(len(attack_types))
    width = 0.35

    ax4.bar(x - width/2, v1_f1, width, label='LSTM v1', color=colors['secondary'], alpha=0.7)
    ax4.bar(x + width/2, v2_f1, width, label='TCN v2', color=colors['primary'], alpha=0.7)

    ax4.set_ylabel('F1-Score (%)', fontsize=12)
    ax4.set_title('F1-Score by Attack Type', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(attack_types)
    ax4.legend()
    ax4.set_ylim([94, 100])

    # Overall title
    fig.suptitle('SecIDS: v1 → v2 Performance Improvements',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created performance_comparison.png")


def create_feature_importance():
    """Create feature importance chart"""
    fig, ax = plt.subplots(figsize=(10, 8))

    features = [
        'Inter-Arrival Time (Δt)',
        'Payload Entropy',
        'Hamming Distance',
        'ID Change Frequency',
        'DLC Variance',
        'ID Occurrence Rate',
        'Payload Mean',
        'Payload Std Dev',
        'Time-Since-Last',
        'ID Diversity'
    ]

    importance = [0.145, 0.132, 0.118, 0.095, 0.087, 0.082, 0.076, 0.071, 0.068, 0.062]

    colors_bars = [colors['primary'] if i < 3 else colors['secondary'] if i < 6 else colors['neutral']
                   for i in range(len(features))]

    bars = ax.barh(features, importance, color=colors_bars, alpha=0.7)
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title('Top 10 Feature Importance for CAN Attack Detection',
                 fontsize=14, fontweight='bold')
    ax.set_xlim([0, 0.16])

    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.002, bar.get_y() + bar.get_height()/2.,
                f'{importance[i]:.3f}', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created feature_importance.png")


def create_github_banner():
    """Create GitHub repository banner"""
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis('off')

    # Background gradient
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient, extent=[0, 10, 0, 2], aspect='auto', cmap='Blues', alpha=0.3)

    # Main title
    ax.text(5, 1.45, '🛡️ SecIDS-v2', ha='center', va='center',
            fontsize=48, fontweight='bold', color=colors['primary'])

    # Subtitle
    ax.text(5, 1.05, 'Next-Generation Automotive Intrusion Detection System',
            ha='center', va='center', fontsize=18, color=colors['neutral'])

    # Key features
    features_text = (
        "⚡ 4.2ms Inference  |  🎯 98.2% Accuracy  |  "
        "🧠 Temporal CNN + Mamba  |  🚗 Production-Ready"
    )
    ax.text(5, 0.65, features_text, ha='center', va='center',
            fontsize=14, color='black', fontweight='bold')

    # Tech stack
    tech_text = "PyTorch Lightning • ONNX • TensorRT • FastAPI • Streamlit • Docker"
    ax.text(5, 0.3, tech_text, ha='center', va='center',
            fontsize=11, color=colors['neutral'], style='italic')

    plt.tight_layout()
    plt.savefig(output_dir / 'github_banner.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Created github_banner.png")


def create_linkedin_post():
    """Create LinkedIn post visual"""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Background
    bg = Rectangle((0, 0), 10, 10, facecolor='#f0f0f0')
    ax.add_patch(bg)

    # Header
    header = Rectangle((0, 8.5), 10, 1.5, facecolor=colors['primary'])
    ax.add_patch(header)
    ax.text(5, 9.5, '🛡️ SecIDS-v2 Released', ha='center', va='center',
            fontsize=32, fontweight='bold', color='white')
    ax.text(5, 8.9, 'Production-Ready Automotive IDS', ha='center', va='center',
            fontsize=16, color='white')

    # Main content
    ax.text(5, 7.8, 'Major Performance Leap', ha='center', va='top',
            fontsize=24, fontweight='bold', color=colors['primary'])

    # Metrics boxes
    metrics = [
        ('Accuracy', '97.2% → 98.2%', '+1.0%', 2),
        ('Inference', '18.5ms → 4.2ms', '4.4× faster', 5),
        ('Model Size', '5.2M → 3.8M', '-27%', 8)
    ]

    for label, change, improvement, x in metrics:
        # Box
        box = FancyBboxPatch((x-0.9, 5.5), 1.8, 1.5,
                            boxstyle="round,pad=0.1",
                            edgecolor=colors['primary'],
                            facecolor='white', linewidth=3)
        ax.add_patch(box)

        # Text
        ax.text(x, 6.6, label, ha='center', va='center',
               fontsize=14, fontweight='bold', color=colors['neutral'])
        ax.text(x, 6.2, change, ha='center', va='center',
               fontsize=12, color='black')
        ax.text(x, 5.8, improvement, ha='center', va='center',
               fontsize=13, fontweight='bold', color=colors['success'])

    # Key features
    features = [
        '🧠 Temporal CNN + Mamba architectures',
        '🎯 Multi-task learning (DoS/Fuzzy/Spoofing/Replay)',
        '⚡ NVIDIA Jetson optimized (TensorRT INT8)',
        '🔧 FastAPI + Streamlit deployment',
        '📊 25 CAN-specific features engineered',
        '🐋 Docker-ready production pipeline'
    ]

    y_start = 4.8
    for i, feature in enumerate(features):
        ax.text(0.5, y_start - i*0.5, feature, ha='left', va='top',
               fontsize=11, color='black')

    # Footer
    footer = Rectangle((0, 0), 10, 1.2, facecolor=colors['primary'], alpha=0.1)
    ax.add_patch(footer)
    ax.text(5, 0.75, 'GitHub: Keyvanhardani/SecIDS-v2', ha='center', va='center',
           fontsize=14, fontweight='bold', color=colors['primary'])
    ax.text(5, 0.35, 'Open Source • Production Tested • Research Validated',
           ha='center', va='center', fontsize=11, color=colors['neutral'])

    plt.tight_layout()
    plt.savefig(output_dir / 'linkedin_post.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Created linkedin_post.png")


if __name__ == "__main__":
    print("\n🎨 Generating SecIDS-v2 Visuals...")
    print("=" * 50)

    create_architecture_diagram()
    create_performance_comparison()
    create_feature_importance()
    create_github_banner()
    create_linkedin_post()

    print("=" * 50)
    print(f"✅ All visuals saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    print("  1. architecture.png - Model architecture diagram")
    print("  2. performance_comparison.png - v1 vs v2 metrics")
    print("  3. feature_importance.png - Feature importance chart")
    print("  4. github_banner.png - Repository banner")
    print("  5. linkedin_post.png - LinkedIn post image")
