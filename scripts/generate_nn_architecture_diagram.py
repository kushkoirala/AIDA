#!/usr/bin/env python3
"""
Generate Neural Network Architecture Diagram for AIDA PPO Policy

Creates a publication-quality diagram showing the Actor-Critic architecture
used for autonomous flight control.

Author: AIDA Project
Date: December 2024
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

def draw_layer(ax, x, y, n_neurons, layer_width, layer_height, color, label=None, neuron_radius=0.15):
    """Draw a neural network layer as a rectangle with neurons"""
    # Draw the layer box
    box = FancyBboxPatch(
        (x - layer_width/2, y - layer_height/2),
        layer_width, layer_height,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=color,
        edgecolor='black',
        linewidth=1.5,
        alpha=0.9
    )
    ax.add_patch(box)

    # Add label inside
    if label:
        ax.text(x, y, label, ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    return x, y

def draw_neurons(ax, x, y, n_neurons, max_show=6, radius=0.12, color='white'):
    """Draw individual neurons for a layer"""
    if n_neurons <= max_show:
        neurons_to_draw = n_neurons
        show_dots = False
    else:
        neurons_to_draw = max_show - 1
        show_dots = True

    spacing = 0.35
    total_height = (neurons_to_draw - 1) * spacing
    start_y = y + total_height / 2

    positions = []
    for i in range(neurons_to_draw):
        ny = start_y - i * spacing
        circle = Circle((x, ny), radius, facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(circle)
        positions.append((x, ny))

    if show_dots:
        # Add dots to indicate more neurons
        ax.text(x, y - total_height/2 - 0.25, '⋮', ha='center', va='center', fontsize=14, color='gray')

    return positions

def create_architecture_diagram():
    """Create the full architecture diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(-1, 15)
    ax.set_ylim(-2, 8)
    ax.set_aspect('equal')
    ax.axis('off')

    # Colors
    input_color = '#4CAF50'      # Green
    shared_color = '#2196F3'     # Blue
    policy_color = '#9C27B0'     # Purple
    value_color = '#FF9800'      # Orange
    output_color = '#F44336'     # Red

    # Title
    ax.text(7, 7.5, 'PPO Actor-Critic Neural Network Architecture',
            ha='center', va='center', fontsize=16, fontweight='bold')
    ax.text(7, 7.0, 'for Autonomous Flight Control',
            ha='center', va='center', fontsize=12, style='italic', color='gray')

    # ==================== INPUT LAYER ====================
    input_x = 0.5
    input_labels = [
        ('x, y, z', 'Position'),
        ('u, v, w', 'Velocity'),
        ('φ, θ, ψ', 'Attitude'),
        ('p, q, r', 'Angular Rate'),
    ]

    # Input box
    draw_layer(ax, input_x, 3, 12, 1.2, 4.5, input_color, None)
    ax.text(input_x, 5.8, 'Input\n(12)', ha='center', va='center', fontsize=10, fontweight='bold')

    # Draw input labels
    y_positions = [4.5, 3.5, 2.5, 1.5]
    for (symbol, name), y_pos in zip(input_labels, y_positions):
        ax.text(input_x, y_pos, symbol, ha='center', va='center', fontsize=9, color='white', fontweight='bold')

    # Input description box
    ax.text(input_x, -0.5, 'State Vector\n(12-dim)', ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    # ==================== SHARED FEATURE EXTRACTOR ====================
    # This is implicit - PPO uses separate networks for policy and value

    # ==================== POLICY NETWORK (Actor) ====================
    policy_y = 4.5

    # Layer 1: 256 neurons
    draw_layer(ax, 3, policy_y, 256, 1.0, 2.0, policy_color)
    ax.text(3, policy_y, '256', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    ax.text(3, policy_y + 1.3, 'Dense + ReLU', ha='center', va='center', fontsize=8)

    # Layer 2: 256 neurons
    draw_layer(ax, 5.5, policy_y, 256, 1.0, 2.0, policy_color)
    ax.text(5.5, policy_y, '256', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    ax.text(5.5, policy_y + 1.3, 'Dense + ReLU', ha='center', va='center', fontsize=8)

    # Layer 3: 128 neurons
    draw_layer(ax, 8, policy_y, 128, 1.0, 1.6, policy_color)
    ax.text(8, policy_y, '128', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    ax.text(8, policy_y + 1.1, 'Dense + ReLU', ha='center', va='center', fontsize=8)

    # Policy output: Mean (4) + Log Std (4)
    draw_layer(ax, 10.5, policy_y + 0.7, 4, 0.8, 1.2, output_color)
    ax.text(10.5, policy_y + 0.7, 'μ', ha='center', va='center', fontsize=14, fontweight='bold', color='white')

    draw_layer(ax, 10.5, policy_y - 0.7, 4, 0.8, 1.2, '#E91E63')
    ax.text(10.5, policy_y - 0.7, 'σ', ha='center', va='center', fontsize=14, fontweight='bold', color='white')

    # Policy label
    ax.text(6.5, policy_y + 2.3, 'Policy Network (Actor) π(a|s)', ha='center', va='center',
            fontsize=11, fontweight='bold', color=policy_color,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=policy_color, alpha=0.9))

    # ==================== VALUE NETWORK (Critic) ====================
    value_y = 1.5

    # Layer 1: 256 neurons
    draw_layer(ax, 3, value_y, 256, 1.0, 2.0, value_color)
    ax.text(3, value_y, '256', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    ax.text(3, value_y - 1.3, 'Dense + ReLU', ha='center', va='center', fontsize=8)

    # Layer 2: 256 neurons
    draw_layer(ax, 5.5, value_y, 256, 1.0, 2.0, value_color)
    ax.text(5.5, value_y, '256', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    ax.text(5.5, value_y - 1.3, 'Dense + ReLU', ha='center', va='center', fontsize=8)

    # Layer 3: 128 neurons
    draw_layer(ax, 8, value_y, 128, 1.0, 1.6, value_color)
    ax.text(8, value_y, '128', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    ax.text(8, value_y - 1.1, 'Dense + ReLU', ha='center', va='center', fontsize=8)

    # Value output: 1 neuron
    draw_layer(ax, 10.5, value_y, 1, 0.8, 0.8, '#795548')
    ax.text(10.5, value_y, 'V', ha='center', va='center', fontsize=12, fontweight='bold', color='white')

    # Value label
    ax.text(6.5, value_y - 2.0, 'Value Network (Critic) V(s)', ha='center', va='center',
            fontsize=11, fontweight='bold', color=value_color,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=value_color, alpha=0.9))

    # ==================== ARROWS ====================
    arrow_style = dict(arrowstyle='->', color='gray', lw=1.5, mutation_scale=15)

    # Input to Policy network
    ax.annotate('', xy=(2.4, policy_y), xytext=(1.2, 3.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    # Input to Value network
    ax.annotate('', xy=(2.4, value_y), xytext=(1.2, 2.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    # Policy network arrows
    ax.annotate('', xy=(4.9, policy_y), xytext=(3.6, policy_y),
                arrowprops=dict(arrowstyle='->', color=policy_color, lw=1.5))
    ax.annotate('', xy=(7.4, policy_y), xytext=(6.1, policy_y),
                arrowprops=dict(arrowstyle='->', color=policy_color, lw=1.5))
    ax.annotate('', xy=(10.0, policy_y + 0.5), xytext=(8.6, policy_y + 0.2),
                arrowprops=dict(arrowstyle='->', color=policy_color, lw=1.5))
    ax.annotate('', xy=(10.0, policy_y - 0.5), xytext=(8.6, policy_y - 0.2),
                arrowprops=dict(arrowstyle='->', color=policy_color, lw=1.5))

    # Value network arrows
    ax.annotate('', xy=(4.9, value_y), xytext=(3.6, value_y),
                arrowprops=dict(arrowstyle='->', color=value_color, lw=1.5))
    ax.annotate('', xy=(7.4, value_y), xytext=(6.1, value_y),
                arrowprops=dict(arrowstyle='->', color=value_color, lw=1.5))
    ax.annotate('', xy=(10.0, value_y), xytext=(8.6, value_y),
                arrowprops=dict(arrowstyle='->', color=value_color, lw=1.5))

    # ==================== OUTPUT LABELS ====================
    # Action outputs
    action_labels = ['δₜ (Throttle)', 'δₐ (Aileron)', 'δₑ (Elevator)', 'δᵣ (Rudder)']
    ax.text(12.5, policy_y + 0.7, 'Action Mean\n(4-dim)', ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=output_color, alpha=0.9))
    ax.text(12.5, policy_y - 0.7, 'Action Std\n(4-dim)', ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='#E91E63', alpha=0.9))

    # Value output
    ax.text(12.5, value_y, 'State Value\n(scalar)', ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='#795548', alpha=0.9))

    # Output arrows
    ax.annotate('', xy=(11.8, policy_y + 0.7), xytext=(11.0, policy_y + 0.7),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.annotate('', xy=(11.8, policy_y - 0.7), xytext=(11.0, policy_y - 0.7),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.annotate('', xy=(11.8, value_y), xytext=(11.0, value_y),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    # ==================== GAUSSIAN SAMPLING BOX ====================
    gauss_x, gauss_y = 13.5, policy_y
    ax.text(gauss_x, gauss_y, 'a ~ N(μ, σ²)', ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange', alpha=0.9, lw=1.5))

    # Arrows to Gaussian
    ax.annotate('', xy=(gauss_x - 0.6, gauss_y + 0.3), xytext=(13.2, policy_y + 0.7),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1, connectionstyle='arc3,rad=-0.2'))
    ax.annotate('', xy=(gauss_x - 0.6, gauss_y - 0.3), xytext=(13.2, policy_y - 0.7),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1, connectionstyle='arc3,rad=0.2'))

    # ==================== LEGEND / INFO BOX ====================
    info_text = """Network Architecture:
• Input: 12-dim state vector (position, velocity, attitude, angular rates)
• Policy: 3 hidden layers [256, 256, 128] with ReLU activation
• Value: 3 hidden layers [256, 256, 128] with ReLU activation
• Output: Gaussian policy (mean + std) for continuous actions
• Total Parameters: ~270k (Policy: ~135k, Value: ~135k)
• Framework: Stable-Baselines3 PPO"""

    ax.text(7, -1.3, info_text, ha='center', va='top', fontsize=9,
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='#f5f5f5', edgecolor='gray', alpha=0.95))

    plt.tight_layout()
    return fig

def main():
    """Generate and save the architecture diagram"""
    print("Generating Neural Network Architecture Diagram...")

    fig = create_architecture_diagram()

    # Save in multiple formats
    output_path = '/home/AIDA/docs/img/ppo_architecture.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved: {output_path}")

    # Also save PDF for papers
    pdf_path = '/home/AIDA/docs/img/ppo_architecture.pdf'
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved: {pdf_path}")

    plt.close()
    print("Done!")

if __name__ == '__main__':
    main()
