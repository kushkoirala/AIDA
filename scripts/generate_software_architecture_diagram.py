#!/usr/bin/env python3
"""
Generate Software Architecture Diagram for AIDA

Creates a UML-style component diagram showing the system architecture
for the autonomous flight control framework.

Author: AIDA Project
Date: December 2024
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon
import numpy as np

def draw_component(ax, x, y, width, height, name, stereotype=None, color='#E3F2FD', border_color='#1976D2'):
    """Draw a UML component box"""
    # Main box
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=color,
        edgecolor=border_color,
        linewidth=2
    )
    ax.add_patch(box)

    # Component tabs (UML style)
    tab_width = 0.3
    tab_height = 0.15
    tab1 = Rectangle((x - 0.05, y + height - 0.4), tab_width, tab_height,
                      facecolor=color, edgecolor=border_color, linewidth=1.5)
    tab2 = Rectangle((x - 0.05, y + height - 0.65), tab_width, tab_height,
                      facecolor=color, edgecolor=border_color, linewidth=1.5)
    ax.add_patch(tab1)
    ax.add_patch(tab2)

    # Stereotype
    if stereotype:
        ax.text(x + width/2, y + height - 0.25, f'«{stereotype}»',
                ha='center', va='center', fontsize=8, style='italic', color='gray')
        ax.text(x + width/2, y + height - 0.55, name,
                ha='center', va='center', fontsize=10, fontweight='bold')
    else:
        ax.text(x + width/2, y + height - 0.4, name,
                ha='center', va='center', fontsize=10, fontweight='bold')

    return x, y, width, height

def draw_package(ax, x, y, width, height, name, color='#FFF8E1', border_color='#FF8F00'):
    """Draw a UML package"""
    # Tab
    tab_width = min(width * 0.4, 2.0)
    tab_height = 0.4
    tab = FancyBboxPatch(
        (x, y + height), tab_width, tab_height,
        boxstyle="round,pad=0.01,rounding_size=0.05",
        facecolor=color,
        edgecolor=border_color,
        linewidth=2
    )
    ax.add_patch(tab)

    # Main box
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=color,
        edgecolor=border_color,
        linewidth=2,
        alpha=0.7
    )
    ax.add_patch(box)

    # Name in tab
    ax.text(x + tab_width/2, y + height + tab_height/2, name,
            ha='center', va='center', fontsize=10, fontweight='bold')

    return x, y, width, height

def draw_interface(ax, x, y, name, color='#4CAF50'):
    """Draw a UML interface (lollipop)"""
    circle = Circle((x, y), 0.15, facecolor='white', edgecolor=color, linewidth=2)
    ax.add_patch(circle)
    ax.text(x, y - 0.35, name, ha='center', va='top', fontsize=8)
    return x, y

def draw_database(ax, x, y, width, height, name, color='#E8F5E9', border_color='#388E3C'):
    """Draw a database cylinder"""
    # Cylinder body
    from matplotlib.patches import Ellipse

    # Bottom ellipse
    bottom = Ellipse((x + width/2, y + 0.15), width, 0.4,
                     facecolor=color, edgecolor=border_color, linewidth=2)
    ax.add_patch(bottom)

    # Rectangle body
    rect = Rectangle((x, y + 0.15), width, height - 0.3,
                     facecolor=color, edgecolor=border_color, linewidth=2)
    ax.add_patch(rect)

    # Top ellipse
    top = Ellipse((x + width/2, y + height - 0.15), width, 0.4,
                  facecolor=color, edgecolor=border_color, linewidth=2)
    ax.add_patch(top)

    # Cover the rectangle edges
    rect_cover = Rectangle((x + 0.02, y + 0.2), width - 0.04, height - 0.4,
                           facecolor=color, edgecolor=color, linewidth=0)
    ax.add_patch(rect_cover)

    ax.text(x + width/2, y + height/2, name,
            ha='center', va='center', fontsize=9, fontweight='bold')

    return x, y, width, height

def draw_arrow(ax, start, end, style='->', color='#333', lw=1.5, label=None, connectionstyle=None):
    """Draw an arrow between components"""
    props = dict(arrowstyle=style, color=color, lw=lw)
    if connectionstyle:
        props['connectionstyle'] = connectionstyle
    ax.annotate('', xy=end, xytext=start, arrowprops=props)

    if label:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(mid_x, mid_y + 0.15, label, ha='center', va='bottom', fontsize=7, color='gray')

def create_software_architecture():
    """Create the full software architecture diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(18, 14))
    ax.set_xlim(-1, 17)
    ax.set_ylim(-1, 13)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(8, 12.5, 'AIDA Software Architecture',
            ha='center', va='center', fontsize=18, fontweight='bold')
    ax.text(8, 12.0, 'Autonomous Intelligent Decision Architecture',
            ha='center', va='center', fontsize=12, style='italic', color='gray')

    # ==================== PACKAGES ====================

    # Training Package
    draw_package(ax, 0.5, 7, 5, 4, 'Training', color='#E8EAF6', border_color='#3F51B5')

    # Simulation Package
    draw_package(ax, 6, 7, 5, 4, 'Simulation', color='#E3F2FD', border_color='#1976D2')

    # Visualization Package
    draw_package(ax, 11.5, 7, 4.5, 4, 'Visualization', color='#F3E5F5', border_color='#7B1FA2')

    # Data Package
    draw_package(ax, 0.5, 1.5, 4, 4.5, 'Data & Storage', color='#E8F5E9', border_color='#388E3C')

    # Aircraft Models Package
    draw_package(ax, 5, 1.5, 5.5, 4.5, 'Aircraft Models', color='#FFF3E0', border_color='#E65100')

    # Configuration Package
    draw_package(ax, 11, 1.5, 5, 4.5, 'Configuration', color='#FFEBEE', border_color='#C62828')

    # ==================== TRAINING COMPONENTS ====================

    draw_component(ax, 0.8, 9.2, 2.0, 1.2, 'PPO\nTrainer', 'algorithm', '#C5CAE9', '#3F51B5')
    draw_component(ax, 3.2, 9.2, 2.0, 1.2, 'Curriculum\nLearning', 'strategy', '#C5CAE9', '#3F51B5')
    draw_component(ax, 0.8, 7.5, 2.0, 1.2, 'Behavior\nCloning', 'algorithm', '#C5CAE9', '#3F51B5')
    draw_component(ax, 3.2, 7.5, 2.0, 1.2, 'Policy\nNetwork', 'neural net', '#C5CAE9', '#3F51B5')

    # ==================== SIMULATION COMPONENTS ====================

    draw_component(ax, 6.3, 9.2, 2.0, 1.2, 'Flight\nEnvironment', 'gymnasium', '#BBDEFB', '#1976D2')
    draw_component(ax, 8.7, 9.2, 2.0, 1.2, 'Reward\nShaping', 'module', '#BBDEFB', '#1976D2')
    draw_component(ax, 6.3, 7.5, 2.0, 1.2, 'GPU Flight\nDynamics', 'CUDA', '#BBDEFB', '#1976D2')
    draw_component(ax, 8.7, 7.5, 2.0, 1.2, 'State\nManager', 'module', '#BBDEFB', '#1976D2')

    # ==================== VISUALIZATION COMPONENTS ====================

    draw_component(ax, 11.8, 9.2, 1.9, 1.2, 'Tensor\nBoard', 'logging', '#E1BEE7', '#7B1FA2')
    draw_component(ax, 13.9, 9.2, 1.8, 1.2, '3D\nViewer', 'web', '#E1BEE7', '#7B1FA2')
    draw_component(ax, 11.8, 7.5, 1.9, 1.2, 'Telemetry\nServer', 'websocket', '#E1BEE7', '#7B1FA2')
    draw_component(ax, 13.9, 7.5, 1.8, 1.2, 'HTTP\nServer', 'web', '#E1BEE7', '#7B1FA2')

    # ==================== DATA COMPONENTS ====================

    draw_component(ax, 0.8, 4.2, 1.6, 1.2, 'Check\npoints', 'storage', '#C8E6C9', '#388E3C')
    draw_component(ax, 2.6, 4.2, 1.6, 1.2, 'BC\nDataset', 'storage', '#C8E6C9', '#388E3C')
    draw_component(ax, 0.8, 2.0, 1.6, 1.2, 'Training\nLogs', 'storage', '#C8E6C9', '#388E3C')
    draw_component(ax, 2.6, 2.0, 1.6, 1.2, 'Eval\nResults', 'storage', '#C8E6C9', '#388E3C')

    # ==================== AIRCRAFT MODEL COMPONENTS ====================

    draw_component(ax, 5.3, 4.2, 2.2, 1.2, 'Cessna 172', 'aircraft', '#FFE0B2', '#E65100')
    draw_component(ax, 7.8, 4.2, 2.2, 1.2, 'Udaan\nUAV', 'aircraft', '#FFE0B2', '#E65100')
    draw_component(ax, 5.3, 2.0, 2.2, 1.2, 'Aero\nCoefficients', 'data', '#FFE0B2', '#E65100')
    draw_component(ax, 7.8, 2.0, 2.2, 1.2, '3D Models\n(GLTF)', 'asset', '#FFE0B2', '#E65100')

    # ==================== CONFIGURATION COMPONENTS ====================

    draw_component(ax, 11.3, 4.2, 2.0, 1.2, 'Runway\nConfig', 'config', '#FFCDD2', '#C62828')
    draw_component(ax, 13.6, 4.2, 2.0, 1.2, 'Training\nParams', 'config', '#FFCDD2', '#C62828')
    draw_component(ax, 11.3, 2.0, 2.0, 1.2, 'Aircraft\nParams', 'config', '#FFCDD2', '#C62828')
    draw_component(ax, 13.6, 2.0, 2.0, 1.2, 'Environment\nParams', 'config', '#FFCDD2', '#C62828')

    # ==================== CONNECTIONS ====================

    # Training -> Simulation
    draw_arrow(ax, (5.5, 9.8), (6.3, 9.8), label='step()')
    draw_arrow(ax, (6.3, 8.1), (5.5, 8.1), '->', label='obs, reward')

    # Simulation -> Visualization
    draw_arrow(ax, (11, 9.8), (11.8, 9.8), label='metrics')
    draw_arrow(ax, (11, 8.1), (11.8, 8.1), label='telemetry')

    # Training -> Data
    draw_arrow(ax, (1.8, 7.5), (1.8, 5.9), label='save')
    draw_arrow(ax, (3.4, 7.5), (3.4, 5.9), label='load')

    # Simulation -> Aircraft Models
    draw_arrow(ax, (7.3, 7.5), (7.3, 5.9), label='params')
    draw_arrow(ax, (9.2, 7.5), (9.2, 5.9), label='dynamics')

    # Aircraft -> Configuration
    draw_arrow(ax, (10.3, 3.5), (11.3, 3.5), label='config')
    draw_arrow(ax, (10.3, 2.6), (11.3, 2.6))

    # Internal Training connections
    draw_arrow(ax, (2.8, 10.3), (3.2, 10.3), '->', '#3F51B5', 1)
    draw_arrow(ax, (4.2, 9.2), (4.2, 8.7), '->', '#3F51B5', 1)

    # Internal Simulation connections
    draw_arrow(ax, (8.3, 10.3), (8.7, 10.3), '->', '#1976D2', 1)
    draw_arrow(ax, (7.3, 9.2), (7.3, 8.7), '->', '#1976D2', 1)
    draw_arrow(ax, (9.7, 9.2), (9.7, 8.7), '->', '#1976D2', 1)

    # Visualization internal
    draw_arrow(ax, (12.75, 9.2), (12.75, 8.7), '->', '#7B1FA2', 1)
    draw_arrow(ax, (14.8, 9.2), (14.8, 8.7), '->', '#7B1FA2', 1)

    # ==================== EXTERNAL INTERFACES ====================

    # User interface
    ax.text(14.8, 11.3, 'User', ha='center', va='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    draw_arrow(ax, (14.8, 11.0), (14.8, 10.4), '->')

    # GPU
    ax.text(7.3, 6.5, 'GPU (CUDA)', ha='center', va='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#76FF03', alpha=0.6))
    draw_arrow(ax, (7.3, 6.8), (7.3, 7.5), '->')

    # ==================== LEGEND ====================
    legend_x = 0.5
    legend_y = -0.3

    ax.text(legend_x, legend_y, 'Legend:', fontsize=10, fontweight='bold')

    # Package example
    pkg = FancyBboxPatch((legend_x + 1.5, legend_y - 0.2), 0.8, 0.4,
                         boxstyle="round,pad=0.02", facecolor='#FFF8E1',
                         edgecolor='#FF8F00', linewidth=1.5)
    ax.add_patch(pkg)
    ax.text(legend_x + 2.5, legend_y, 'Package', fontsize=8, va='center')

    # Component example
    comp = FancyBboxPatch((legend_x + 4, legend_y - 0.2), 0.8, 0.4,
                          boxstyle="round,pad=0.02", facecolor='#E3F2FD',
                          edgecolor='#1976D2', linewidth=1.5)
    ax.add_patch(comp)
    ax.text(legend_x + 5, legend_y, 'Component', fontsize=8, va='center')

    # Data flow
    draw_arrow(ax, (legend_x + 6.5, legend_y), (legend_x + 7.3, legend_y))
    ax.text(legend_x + 7.5, legend_y, 'Data Flow', fontsize=8, va='center')

    # Technologies
    ax.text(legend_x + 9.5, legend_y, 'Technologies: Python, PyTorch, CUDA/CuPy, Stable-Baselines3, Gymnasium, WebSocket',
            fontsize=8, va='center', color='gray')

    plt.tight_layout()
    return fig

def main():
    """Generate and save the architecture diagram"""
    print("Generating Software Architecture Diagram...")

    fig = create_software_architecture()

    # Save in multiple formats
    output_path = '/home/AIDA/docs/img/software_architecture.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved: {output_path}")

    # Also save PDF for papers
    pdf_path = '/home/AIDA/docs/img/software_architecture.pdf'
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved: {pdf_path}")

    plt.close()
    print("Done!")

if __name__ == '__main__':
    main()
