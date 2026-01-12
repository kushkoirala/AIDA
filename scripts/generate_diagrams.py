#!/usr/bin/env python3
"""
Generate high-quality architecture diagrams for AIDA README
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon
from matplotlib.collections import PatchCollection
import numpy as np

# Set high DPI for crisp images
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 250
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12

# Color palette
COLORS = {
    'primary': '#2563EB',      # Blue
    'secondary': '#7C3AED',    # Purple
    'success': '#059669',      # Green
    'warning': '#D97706',      # Orange
    'danger': '#DC2626',       # Red
    'dark': '#1F2937',         # Dark gray
    'light': '#F3F4F6',        # Light gray
    'white': '#FFFFFF',
    'neural': '#EC4899',       # Pink for NN
    'expert': '#10B981',       # Teal for expert
    'sim': '#F59E0B',          # Amber for simulation
}

def create_rounded_box(ax, x, y, width, height, text, color, text_color='white', fontsize=11, alpha=0.9):
    """Create a rounded rectangle with centered text"""
    box = FancyBboxPatch((x, y), width, height,
                          boxstyle="round,pad=0.02,rounding_size=0.08",
                          facecolor=color, edgecolor='white', linewidth=2.5, alpha=alpha)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=text_color, wrap=True)
    return box

def draw_arrow(ax, start, end, color='#374151', style='->', lw=2.5):
    """Draw an arrow between two points"""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))


# =============================================================================
# 1. SYSTEM ARCHITECTURE DIAGRAM
# =============================================================================
def create_system_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(16, 11))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 11)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title - positioned higher with more space
    ax.text(8, 10.3, 'AIDA SYSTEM ARCHITECTURE', ha='center', va='center',
            fontsize=22, fontweight='bold', color=COLORS['dark'])

    # Main container
    main_box = FancyBboxPatch((0.5, 0.5), 15, 9,
                               boxstyle="round,pad=0.02,rounding_size=0.15",
                               facecolor=COLORS['light'], edgecolor=COLORS['dark'],
                               linewidth=3, alpha=0.3)
    ax.add_patch(main_box)

    # Training Layer
    create_rounded_box(ax, 1, 7, 4, 2.2, 'TRAINING LAYER\n\nPPO Algorithm\nResidual RL\nBehavior Cloning\nCurriculum Learning',
                       COLORS['primary'], fontsize=10)

    # Inference Layer
    create_rounded_box(ax, 6, 7, 4, 2.2, 'INFERENCE LAYER\n\nPolicy Network\nExpert Controller\nHybrid Blending\nSafety Monitor',
                       COLORS['secondary'], fontsize=10)

    # Visualization Layer
    create_rounded_box(ax, 11, 7, 4, 2.2, 'VISUALIZATION\n\n3D WebGL View\nTensorBoard\nTelemetry WebSocket\nAudio GPWS',
                       COLORS['success'], fontsize=10)

    # Arrows between top layers
    draw_arrow(ax, (5, 8.1), (6, 8.1))
    draw_arrow(ax, (10, 8.1), (11, 8.1))

    # Simulation Layer (large box at bottom)
    sim_box = FancyBboxPatch((1, 1), 14, 5.2,
                              boxstyle="round,pad=0.02,rounding_size=0.15",
                              facecolor=COLORS['sim'], edgecolor='white',
                              linewidth=2, alpha=0.2)
    ax.add_patch(sim_box)
    ax.text(8, 5.8, 'SIMULATION LAYER', ha='center', va='center',
            fontsize=16, fontweight='bold', color=COLORS['dark'])

    # GPU Flight Sim
    create_rounded_box(ax, 1.5, 1.8, 3.8, 3, 'GPU FLIGHT SIM\n\nCuPy/CUDA\n10k+ instances\nRK4 integration\n6-DOF dynamics',
                       COLORS['warning'], fontsize=10)

    # Gymnasium Env
    create_rounded_box(ax, 6.1, 1.8, 3.8, 3, 'GYMNASIUM ENV\n\nResidualEnvV2\n25-dim observation\n7-dim action\nPhase rewards',
                       COLORS['danger'], fontsize=10)

    # Aircraft Models
    create_rounded_box(ax, 10.7, 1.8, 3.8, 3, 'AIRCRAFT MODELS\n\nCessna 172\nUdaan UAV\nAero coefficients\nMass/inertia',
                       COLORS['neural'], fontsize=10)

    # Vertical arrows from top to simulation
    draw_arrow(ax, (3, 7), (3, 6.3))
    draw_arrow(ax, (8, 7), (8, 6.3))
    draw_arrow(ax, (13, 7), (13, 6.3))

    # Horizontal connections in sim layer
    draw_arrow(ax, (5.3, 3.3), (6.1, 3.3))
    draw_arrow(ax, (9.9, 3.3), (10.7, 3.3))

    plt.tight_layout()
    plt.savefig('/home/AIDA/docs/img/system_architecture.png', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: system_architecture.png")


# =============================================================================
# 2. NEURAL NETWORK ARCHITECTURE DIAGRAM
# =============================================================================
def create_nn_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title - positioned at very top with plenty of space
    ax.text(8, 11.5, 'RESIDUAL RL NEURAL NETWORK ARCHITECTURE', ha='center', va='center',
            fontsize=20, fontweight='bold', color=COLORS['dark'])

    # Aircraft State Input
    create_rounded_box(ax, 0.3, 6.5, 3, 3, 'AIRCRAFT\nSTATE\n(12-dim)\n\nx, y, z\nu, v, w\nφ, θ, ψ\np, q, r',
                       COLORS['primary'], fontsize=10)

    # Observation Builder
    obs_box = FancyBboxPatch((4, 5), 3.5, 5.5,
                              boxstyle="round,pad=0.02,rounding_size=0.15",
                              facecolor=COLORS['light'], edgecolor=COLORS['dark'],
                              linewidth=2, alpha=0.5)
    ax.add_patch(obs_box)
    ax.text(5.75, 10, 'OBSERVATION\nBUILDER', ha='center', va='center',
            fontsize=13, fontweight='bold', color=COLORS['dark'])
    ax.text(5.75, 8.5, '25 dimensions', ha='center', va='center', fontsize=11, color=COLORS['dark'])
    ax.text(5.75, 7.7, 'State (12)', ha='center', va='center', fontsize=10, color=COLORS['dark'])
    ax.text(5.75, 7.0, '+ Expert (7)', ha='center', va='center', fontsize=10, color=COLORS['dark'])
    ax.text(5.75, 6.3, '+ Target (3)', ha='center', va='center', fontsize=10, color=COLORS['dark'])
    ax.text(5.75, 5.6, '+ Phase (3)', ha='center', va='center', fontsize=10, color=COLORS['dark'])

    # Arrow to obs builder
    draw_arrow(ax, (3.3, 8), (4, 8))

    # Expert Controller
    create_rounded_box(ax, 8, 7.5, 3, 3, 'EXPERT\nCONTROLLER\n(Classical)\n\nPhase FSM\nPID Control\nTrajectory Gen',
                       COLORS['expert'], fontsize=10)

    # Policy Network
    create_rounded_box(ax, 8, 3.5, 3, 3, 'POLICY\nNETWORK\n(Neural)\n\n512→512→256\nTanh output\n7-dim action',
                       COLORS['neural'], fontsize=10)

    # Arrows from obs builder
    draw_arrow(ax, (7.5, 8.5), (8, 9))
    draw_arrow(ax, (7.5, 6), (8, 5))

    # Blending Box
    blend_box = FancyBboxPatch((11.5, 4.5), 4, 5,
                                boxstyle="round,pad=0.02,rounding_size=0.15",
                                facecolor=COLORS['warning'], edgecolor='white',
                                linewidth=2.5, alpha=0.9)
    ax.add_patch(blend_box)
    ax.text(13.5, 9, 'RESIDUAL\nBLENDING', ha='center', va='center',
            fontsize=14, fontweight='bold', color='white')
    ax.text(13.5, 7.5, 'u_total =', ha='center', va='center', fontsize=12, color='white')
    ax.text(13.5, 6.8, 'u_expert + λ × δ_nn', ha='center', va='center', fontsize=12, color='white', style='italic')

    # Residual scales
    ax.text(13.5, 5.8, 'Residual Scales (λ):', ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    ax.text(13.5, 5.2, 'Throttle: ±15%', ha='center', va='center', fontsize=9, color='white')
    ax.text(13.5, 4.7, 'Rudder: ±10%', ha='center', va='center', fontsize=9, color='white')

    # Arrows to blending
    draw_arrow(ax, (11, 9), (11.5, 8))
    draw_arrow(ax, (11, 5), (11.5, 6))
    ax.text(11.2, 8.8, 'u_expert', fontsize=10, color=COLORS['expert'], fontweight='bold')
    ax.text(11.2, 5.3, 'δ_nn', fontsize=10, color=COLORS['neural'], fontweight='bold')

    # Output to simulator
    create_rounded_box(ax, 12, 1, 3, 2.5, 'FLIGHT\nSIMULATOR\n\nGPU 6-DOF',
                       COLORS['dark'], fontsize=11)
    draw_arrow(ax, (13.5, 4.5), (13.5, 3.5))

    # Network specs table
    ax.text(1.5, 4.5, 'Network Specifications:', ha='left', va='center', fontsize=12, fontweight='bold', color=COLORS['dark'])
    specs = [
        'Input: 25 dims',
        'Hidden 1: 512 (ReLU)',
        'Hidden 2: 512 (ReLU)',
        'Hidden 3: 256 (ReLU)',
        'Output: 7 dims (Tanh)',
        'Total: ~411k params'
    ]
    for i, spec in enumerate(specs):
        ax.text(1.5, 3.8 - i*0.55, spec, ha='left', va='center', fontsize=10, color=COLORS['dark'], family='monospace')

    plt.tight_layout()
    plt.savefig('/home/AIDA/docs/img/nn_architecture.png', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: nn_architecture.png")


# =============================================================================
# 3. TRAINING PIPELINE DIAGRAM
# =============================================================================
def create_training_pipeline():
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(8, 9.3, 'TRAINING ALGORITHM PIPELINE', ha='center', va='center',
            fontsize=20, fontweight='bold', color=COLORS['dark'])

    # Stage 1: Expert
    create_rounded_box(ax, 0.5, 5, 4.5, 3, 'STAGE 1:\nExpert Demonstration\n\nClassical Controller\n(FSM + PID)\n11 flight phases\n100% success rate',
                       COLORS['expert'], fontsize=10)

    # Stage 2: BC
    create_rounded_box(ax, 5.75, 5, 4.5, 3, 'STAGE 2:\nBehavior Cloning\n\nSupervised Learning\nLoss: MSE(π, a_expert)\n50k transitions\n10-20 epochs',
                       COLORS['primary'], fontsize=10)

    # Stage 3: PPO
    create_rounded_box(ax, 11, 5, 4.5, 3, 'STAGE 3:\nResidual PPO\n\nProximal Policy Opt.\nNN learns corrections\nPhase-specific rewards\n2M timesteps',
                       COLORS['neural'], fontsize=10)

    # Arrows
    draw_arrow(ax, (5, 6.5), (5.75, 6.5), lw=3)
    draw_arrow(ax, (10.25, 6.5), (11, 6.5), lw=3)

    # PPO Hyperparameters box
    hyper_box = FancyBboxPatch((0.5, 0.8), 7, 3.5,
                                boxstyle="round,pad=0.02,rounding_size=0.15",
                                facecolor=COLORS['light'], edgecolor=COLORS['dark'],
                                linewidth=2, alpha=0.5)
    ax.add_patch(hyper_box)
    ax.text(4, 3.9, 'PPO Hyperparameters', ha='center', va='center',
            fontsize=14, fontweight='bold', color=COLORS['dark'])

    params = [
        ('learning_rate', '1e-4'),
        ('n_steps', '4096'),
        ('batch_size', '256'),
        ('gamma', '0.995'),
        ('clip_range', '0.1'),
        ('ent_coef', '0.005'),
    ]
    for i, (name, val) in enumerate(params):
        col = i // 3
        row = i % 3
        ax.text(1.2 + col*3, 3.2 - row*0.7, f'{name}: {val}', ha='left', va='center',
                fontsize=11, color=COLORS['dark'], family='monospace')

    # Reward structure box
    reward_box = FancyBboxPatch((8, 0.8), 7.5, 3.5,
                                 boxstyle="round,pad=0.02,rounding_size=0.15",
                                 facecolor=COLORS['light'], edgecolor=COLORS['dark'],
                                 linewidth=2, alpha=0.5)
    ax.add_patch(reward_box)
    ax.text(11.75, 3.9, 'Phase-Specific Rewards', ha='center', va='center',
            fontsize=14, fontweight='bold', color=COLORS['dark'])

    rewards = [
        'Takeoff: +speed, -centerline_err, +climb',
        'Cruise: +alt_track, +hdg_track, -ctrl_effort',
        'Approach: +glideslope, +localizer, +landing'
    ]
    for i, r in enumerate(rewards):
        ax.text(8.5, 3.1 - i*0.7, r, ha='left', va='center', fontsize=11, color=COLORS['dark'])

    plt.tight_layout()
    plt.savefig('/home/AIDA/docs/img/training_pipeline.png', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: training_pipeline.png")


# =============================================================================
# 4. SOFTWARE ARCHITECTURE DIAGRAM
# =============================================================================
def create_software_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(8, 11.3, 'SOFTWARE PACKAGE ARCHITECTURE', ha='center', va='center',
            fontsize=20, fontweight='bold', color=COLORS['dark'])

    # aida_sim package
    pkg1 = FancyBboxPatch((0.5, 5.5), 3.5, 5,
                           boxstyle="round,pad=0.02,rounding_size=0.15",
                           facecolor=COLORS['primary'], edgecolor='white',
                           linewidth=2.5, alpha=0.9)
    ax.add_patch(pkg1)
    ax.text(2.25, 10, 'aida_sim/', ha='center', va='center',
            fontsize=14, fontweight='bold', color='white')
    modules1 = ['env/', '  flight_env', '  residual_env_v2', 'dynamics/', 'systems/', 'io/telemetry']
    for i, m in enumerate(modules1):
        ax.text(1, 9.2 - i*0.6, m, ha='left', va='center', fontsize=10, color='white', family='monospace')

    # scripts package
    pkg2 = FancyBboxPatch((4.5, 5.5), 3.5, 5,
                           boxstyle="round,pad=0.02,rounding_size=0.15",
                           facecolor=COLORS['secondary'], edgecolor='white',
                           linewidth=2.5, alpha=0.9)
    ax.add_patch(pkg2)
    ax.text(6.25, 10, 'scripts/', ha='center', va='center',
            fontsize=14, fontweight='bold', color='white')
    modules2 = ['train_*.py', 'run_*.py', 'generate_*.py', 'test_*.py', 'triangle_controller']
    for i, m in enumerate(modules2):
        ax.text(5, 9.2 - i*0.6, m, ha='left', va='center', fontsize=10, color='white', family='monospace')

    # viewer package
    pkg3 = FancyBboxPatch((8.5, 5.5), 3.5, 5,
                           boxstyle="round,pad=0.02,rounding_size=0.15",
                           facecolor=COLORS['success'], edgecolor='white',
                           linewidth=2.5, alpha=0.9)
    ax.add_patch(pkg3)
    ax.text(10.25, 10, 'viewer/', ha='center', va='center',
            fontsize=14, fontweight='bold', color='white')
    modules3 = ['public/', '  index.html', '  viewer.js', '  engine_sound.js', 'components/']
    for i, m in enumerate(modules3):
        ax.text(9, 9.2 - i*0.6, m, ha='left', va='center', fontsize=10, color='white', family='monospace')

    # gpu-flight-dynamics
    pkg4 = FancyBboxPatch((12.5, 5.5), 3, 5,
                           boxstyle="round,pad=0.02,rounding_size=0.15",
                           facecolor=COLORS['warning'], edgecolor='white',
                           linewidth=2.5, alpha=0.9)
    ax.add_patch(pkg4)
    ax.text(14, 10, 'gpu-flight-\ndynamics/', ha='center', va='center',
            fontsize=12, fontweight='bold', color='white')
    modules4 = ['python/', '  flight_dynamics', '  aircraft_db']
    for i, m in enumerate(modules4):
        ax.text(12.8, 8.5 - i*0.6, m, ha='left', va='center', fontsize=10, color='white', family='monospace')

    # Data flow arrows
    draw_arrow(ax, (4, 8), (4.5, 8))
    draw_arrow(ax, (8, 8), (8.5, 8))
    draw_arrow(ax, (12, 8), (12.5, 8))

    # Bottom section - models and checkpoints
    create_rounded_box(ax, 0.5, 1, 4.5, 3.5, 'models/\n\nTrained weights (.zip)\nVersion controlled\nresidual_ppo_v2_7ctrl.zip',
                       COLORS['danger'], fontsize=11)

    create_rounded_box(ax, 5.75, 1, 4.5, 3.5, 'checkpoints/\n\nTraining checkpoints\nTensorBoard logs\nNot in git (.gitignore)',
                       COLORS['dark'], fontsize=11)

    create_rounded_box(ax, 11, 1, 4.5, 3.5, 'docs/\n\nArchitecture diagrams\nFlight screenshots\nREADME, guides',
                       COLORS['neural'], fontsize=11)

    plt.tight_layout()
    plt.savefig('/home/AIDA/docs/img/software_architecture.png', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: software_architecture.png")


# =============================================================================
# 5. DEVELOPMENT ENVIRONMENT DIAGRAM
# =============================================================================
def create_dev_environment():
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(7, 9.3, 'DEVELOPMENT ENVIRONMENT', ha='center', va='center',
            fontsize=20, fontweight='bold', color=COLORS['dark'])

    # Hardware box
    hw_box = FancyBboxPatch((0.5, 4.5), 6, 4,
                             boxstyle="round,pad=0.02,rounding_size=0.15",
                             facecolor=COLORS['dark'], edgecolor='white',
                             linewidth=2.5, alpha=0.9)
    ax.add_patch(hw_box)
    ax.text(3.5, 8, 'HARDWARE: Dell 7920', ha='center', va='center',
            fontsize=14, fontweight='bold', color='white')
    hw_specs = [
        'CPU: Intel Xeon (16+ cores)',
        'GPU: NVIDIA RTX 4060 (8GB)',
        'RAM: 32+ GB DDR4',
        'OS: Windows 11 + WSL2'
    ]
    for i, spec in enumerate(hw_specs):
        ax.text(1.2, 7.2 - i*0.65, spec, ha='left', va='center', fontsize=12, color='white')

    # Software stack
    sw_box = FancyBboxPatch((7.5, 4.5), 6, 4,
                             boxstyle="round,pad=0.02,rounding_size=0.15",
                             facecolor=COLORS['primary'], edgecolor='white',
                             linewidth=2.5, alpha=0.9)
    ax.add_patch(sw_box)
    ax.text(10.5, 8, 'SOFTWARE STACK', ha='center', va='center',
            fontsize=14, fontweight='bold', color='white')
    sw_specs = [
        'Python 3.10+',
        'PyTorch 2.x + CUDA 12.x',
        'Stable-Baselines3 2.x',
        'CuPy (GPU NumPy)',
        'Three.js (WebGL)'
    ]
    for i, spec in enumerate(sw_specs):
        ax.text(8.2, 7.2 - i*0.55, spec, ha='left', va='center', fontsize=12, color='white')

    # Performance benchmarks
    perf_box = FancyBboxPatch((0.5, 0.5), 13, 3.3,
                               boxstyle="round,pad=0.02,rounding_size=0.15",
                               facecolor=COLORS['success'], edgecolor='white',
                               linewidth=2.5, alpha=0.9)
    ax.add_patch(perf_box)
    ax.text(7, 3.4, 'PERFORMANCE BENCHMARKS', ha='center', va='center',
            fontsize=14, fontweight='bold', color='white')

    benchmarks = [
        ('GPU Sim (100 inst)', '50k steps/sec', '2,500x RT'),
        ('GPU Sim (1k inst)', '100k steps/sec', '5,000x RT'),
        ('GPU Sim (10k inst)', '569k steps/sec', '28,450x RT'),
        ('Training (16 envs)', '1,300 steps/sec', '-'),
    ]
    for i, (name, rate, rt) in enumerate(benchmarks):
        ax.text(1.2 + i*3.2, 2.5, name, ha='left', va='center', fontsize=10, color='white', fontweight='bold')
        ax.text(1.2 + i*3.2, 1.9, rate, ha='left', va='center', fontsize=10, color='white')
        ax.text(1.2 + i*3.2, 1.4, rt, ha='left', va='center', fontsize=9, color='#D1FAE5')

    plt.tight_layout()
    plt.savefig('/home/AIDA/docs/img/dev_environment.png', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: dev_environment.png")


# =============================================================================
# 6. CROSS-COUNTRY FLIGHT PROFILE
# =============================================================================
def create_flight_profile():
    fig, ax = plt.subplots(1, 1, figsize=(16, 7))

    # Flight data points (distance nm, altitude ft)
    distances = [0, 2, 5, 8, 12, 16, 20, 24, 26, 28, 29.5, 30.5, 31]
    altitudes = [1400, 2000, 3500, 4500, 5500, 5500, 5500, 5500, 4500, 3000, 1800, 1450, 1400]

    # Plot altitude profile
    ax.fill_between(distances, altitudes, 1400, alpha=0.3, color=COLORS['primary'])
    ax.plot(distances, altitudes, 'o-', color=COLORS['primary'], linewidth=3, markersize=10)

    # Phase annotations
    phases = [
        (0.5, 1900, 'TAKEOFF', COLORS['success']),
        (3.5, 3200, 'CLIMB\n500 fpm', COLORS['primary']),
        (14, 5900, 'CRUISE @ 5,500 ft, 110 KTAS', COLORS['secondary']),
        (26, 4200, 'DESCENT', COLORS['warning']),
        (29.5, 2400, 'APPROACH', COLORS['danger']),
        (30.8, 1800, 'LAND', COLORS['dark']),
    ]

    for x, y, text, color in phases:
        ax.annotate(text, xy=(x, y), fontsize=12, fontweight='bold', color=color,
                    ha='center', va='bottom')

    # Airports
    ax.annotate('SN65\nLake Waltanna', xy=(0, 1400), xytext=(0, 600),
                fontsize=12, fontweight='bold', color=COLORS['dark'],
                ha='center', va='top',
                arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))

    ax.annotate('KHUT\nHutchinson', xy=(31, 1400), xytext=(31, 600),
                fontsize=12, fontweight='bold', color=COLORS['dark'],
                ha='center', va='top',
                arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))

    # Styling
    ax.set_xlim(-1, 32)
    ax.set_ylim(0, 6500)
    ax.set_xlabel('Distance (nautical miles)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Altitude (feet MSL)', fontsize=14, fontweight='bold')
    ax.set_title('CROSS-COUNTRY FLIGHT PROFILE: SN65 → KHUT (31 NM)',
                 fontsize=18, fontweight='bold', color=COLORS['dark'], pad=15)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#F8FAFC')
    ax.tick_params(axis='both', labelsize=12)

    # Add ground line
    ax.axhline(y=1400, color=COLORS['success'], linestyle='--', alpha=0.5, linewidth=2, label='Ground Level (~1,400 ft)')
    ax.legend(loc='upper right', fontsize=11)

    plt.tight_layout()
    plt.savefig('/home/AIDA/docs/img/flight_profile.png', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: flight_profile.png")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print("Generating AIDA architecture diagrams...")
    print("=" * 50)

    create_system_architecture()
    create_nn_architecture()
    create_training_pipeline()
    create_software_architecture()
    create_dev_environment()
    create_flight_profile()

    print("=" * 50)
    print("All diagrams generated in /home/AIDA/docs/img/")
