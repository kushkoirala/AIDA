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
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

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

def create_rounded_box(ax, x, y, width, height, text, color, text_color='white', fontsize=9, alpha=0.9):
    """Create a rounded rectangle with centered text"""
    box = FancyBboxPatch((x, y), width, height,
                          boxstyle="round,pad=0.02,rounding_size=0.05",
                          facecolor=color, edgecolor='white', linewidth=2, alpha=alpha)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=text_color, wrap=True)
    return box

def draw_arrow(ax, start, end, color='#374151', style='->'):
    """Draw an arrow between two points"""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=2))


# =============================================================================
# 1. SYSTEM ARCHITECTURE DIAGRAM
# =============================================================================
def create_system_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(7, 9.5, 'AIDA SYSTEM ARCHITECTURE', ha='center', va='center',
            fontsize=16, fontweight='bold', color=COLORS['dark'])

    # Main container
    main_box = FancyBboxPatch((0.5, 0.5), 13, 8.5,
                               boxstyle="round,pad=0.02,rounding_size=0.1",
                               facecolor=COLORS['light'], edgecolor=COLORS['dark'],
                               linewidth=3, alpha=0.3)
    ax.add_patch(main_box)

    # Training Layer
    create_rounded_box(ax, 1, 6.5, 3.5, 2, 'TRAINING LAYER\n\nPPO Algorithm\nResidual RL\nBehavior Cloning\nCurriculum Learning',
                       COLORS['primary'], fontsize=8)

    # Inference Layer
    create_rounded_box(ax, 5.25, 6.5, 3.5, 2, 'INFERENCE LAYER\n\nPolicy Network\nExpert Controller\nHybrid Blending\nSafety Monitor',
                       COLORS['secondary'], fontsize=8)

    # Visualization Layer
    create_rounded_box(ax, 9.5, 6.5, 3.5, 2, 'VISUALIZATION\n\n3D WebGL View\nTensorBoard\nTelemetry WS\nAudio GPWS',
                       COLORS['success'], fontsize=8)

    # Arrows between top layers
    draw_arrow(ax, (4.5, 7.5), (5.25, 7.5))
    draw_arrow(ax, (8.75, 7.5), (9.5, 7.5))

    # Simulation Layer (large box at bottom)
    sim_box = FancyBboxPatch((1, 1), 12, 4.5,
                              boxstyle="round,pad=0.02,rounding_size=0.1",
                              facecolor=COLORS['sim'], edgecolor='white',
                              linewidth=2, alpha=0.2)
    ax.add_patch(sim_box)
    ax.text(7, 5.2, 'SIMULATION LAYER', ha='center', va='center',
            fontsize=12, fontweight='bold', color=COLORS['dark'])

    # GPU Flight Sim
    create_rounded_box(ax, 1.5, 2, 3, 2.5, 'GPU FLIGHT SIM\n\nCuPy/CUDA\n10k+ instances\nRK4 integration\n6-DOF dynamics',
                       COLORS['warning'], fontsize=8)

    # Gymnasium Env
    create_rounded_box(ax, 5.5, 2, 3, 2.5, 'GYMNASIUM ENV\n\nResidualEnvV2\n25-dim observation\n7-dim action\nPhase rewards',
                       COLORS['danger'], fontsize=8)

    # Aircraft Models
    create_rounded_box(ax, 9.5, 2, 3, 2.5, 'AIRCRAFT MODELS\n\nCessna 172\nUdaan UAV\nAero coefficients\nMass/inertia',
                       COLORS['neural'], fontsize=8)

    # Vertical arrows from top to simulation
    draw_arrow(ax, (2.75, 6.5), (2.75, 5.5))
    draw_arrow(ax, (7, 6.5), (7, 5.5))
    draw_arrow(ax, (11.25, 6.5), (11.25, 5.5))

    # Horizontal connections in sim layer
    draw_arrow(ax, (4.5, 3.25), (5.5, 3.25))
    draw_arrow(ax, (8.5, 3.25), (9.5, 3.25))

    plt.tight_layout()
    plt.savefig('/home/AIDA/docs/img/system_architecture.png', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: system_architecture.png")


# =============================================================================
# 2. NEURAL NETWORK ARCHITECTURE DIAGRAM
# =============================================================================
def create_nn_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(7, 9.5, 'RESIDUAL RL NEURAL NETWORK ARCHITECTURE', ha='center', va='center',
            fontsize=14, fontweight='bold', color=COLORS['dark'])

    # Aircraft State Input
    create_rounded_box(ax, 0.3, 6, 2.5, 2.5, 'AIRCRAFT\nSTATE\n(12-dim)\n\nx,y,z\nu,v,w\nφ,θ,ψ\np,q,r',
                       COLORS['primary'], fontsize=7)

    # Observation Builder
    obs_box = FancyBboxPatch((3.5, 4.5), 3, 5,
                              boxstyle="round,pad=0.02,rounding_size=0.1",
                              facecolor=COLORS['light'], edgecolor=COLORS['dark'],
                              linewidth=2, alpha=0.5)
    ax.add_patch(obs_box)
    ax.text(5, 9, 'OBSERVATION\nBUILDER', ha='center', va='center',
            fontsize=10, fontweight='bold', color=COLORS['dark'])
    ax.text(5, 7.5, '25 dimensions', ha='center', va='center', fontsize=8, color=COLORS['dark'])
    ax.text(5, 6.8, 'State (12)', ha='center', va='center', fontsize=7, color=COLORS['dark'])
    ax.text(5, 6.3, '+ Expert (7)', ha='center', va='center', fontsize=7, color=COLORS['dark'])
    ax.text(5, 5.8, '+ Target (3)', ha='center', va='center', fontsize=7, color=COLORS['dark'])
    ax.text(5, 5.3, '+ Phase (3)', ha='center', va='center', fontsize=7, color=COLORS['dark'])

    # Arrow to obs builder
    draw_arrow(ax, (2.8, 7.25), (3.5, 7.25))

    # Expert Controller
    create_rounded_box(ax, 7, 7, 2.5, 2.5, 'EXPERT\nCONTROLLER\n(Classical)\n\nPhase FSM\nPID Control\nTrajectory Gen',
                       COLORS['expert'], fontsize=7)

    # Policy Network
    create_rounded_box(ax, 7, 3.5, 2.5, 2.5, 'POLICY\nNETWORK\n(Neural)\n\n512→512→256\nTanh output\n7-dim action',
                       COLORS['neural'], fontsize=7)

    # Arrows from obs builder
    draw_arrow(ax, (6.5, 7.5), (7, 8))
    draw_arrow(ax, (6.5, 5.5), (7, 5))

    # Blending Box
    blend_box = FancyBboxPatch((10, 4.5), 3.5, 4,
                                boxstyle="round,pad=0.02,rounding_size=0.1",
                                facecolor=COLORS['warning'], edgecolor='white',
                                linewidth=2, alpha=0.8)
    ax.add_patch(blend_box)
    ax.text(11.75, 8, 'RESIDUAL\nBLENDING', ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')
    ax.text(11.75, 6.8, 'u_total =', ha='center', va='center', fontsize=9, color='white')
    ax.text(11.75, 6.2, 'u_expert + λ × δ_nn', ha='center', va='center', fontsize=9, color='white', style='italic')

    # Residual scales
    ax.text(11.75, 5.3, 'Residual Scales (λ):', ha='center', va='center', fontsize=7, color='white')
    ax.text(11.75, 4.9, 'Throttle: ±15%', ha='center', va='center', fontsize=6, color='white')
    ax.text(11.75, 4.6, 'Rudder: ±10%', ha='center', va='center', fontsize=6, color='white')

    # Arrows to blending
    draw_arrow(ax, (9.5, 8), (10, 7))
    draw_arrow(ax, (9.5, 5), (10, 6))
    ax.text(9.7, 7.7, 'u_expert', fontsize=7, color=COLORS['expert'])
    ax.text(9.7, 5.3, 'δ_nn', fontsize=7, color=COLORS['neural'])

    # Output to simulator
    create_rounded_box(ax, 10.5, 1, 2.5, 2, 'FLIGHT\nSIMULATOR\n\nGPU 6-DOF',
                       COLORS['dark'], fontsize=8)
    draw_arrow(ax, (11.75, 4.5), (11.75, 3))

    # Network specs table
    ax.text(2, 3.5, 'Network Specifications:', ha='left', va='center', fontsize=9, fontweight='bold', color=COLORS['dark'])
    specs = [
        'Input: 25 dims',
        'Hidden 1: 512 (ReLU)',
        'Hidden 2: 512 (ReLU)',
        'Hidden 3: 256 (ReLU)',
        'Output: 7 dims (Tanh)',
        'Total: ~411k params'
    ]
    for i, spec in enumerate(specs):
        ax.text(2, 3 - i*0.4, spec, ha='left', va='center', fontsize=7, color=COLORS['dark'])

    plt.tight_layout()
    plt.savefig('/home/AIDA/docs/img/nn_architecture.png', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: nn_architecture.png")


# =============================================================================
# 3. TRAINING PIPELINE DIAGRAM
# =============================================================================
def create_training_pipeline():
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(7, 7.5, 'TRAINING ALGORITHM PIPELINE', ha='center', va='center',
            fontsize=14, fontweight='bold', color=COLORS['dark'])

    # Stage 1: Expert
    create_rounded_box(ax, 0.5, 4, 4, 2.5, 'STAGE 1: Expert Demonstration\n\nClassical Controller (FSM + PID)\n11 flight phases\n100% success on known routes',
                       COLORS['expert'], fontsize=8)

    # Stage 2: BC
    create_rounded_box(ax, 5, 4, 4, 2.5, 'STAGE 2: Behavior Cloning\n\nSupervised Learning\nLoss: MSE(π_θ(s), a_expert)\n50k transitions, 10-20 epochs',
                       COLORS['primary'], fontsize=8)

    # Stage 3: PPO
    create_rounded_box(ax, 9.5, 4, 4, 2.5, 'STAGE 3: Residual PPO\n\nProximal Policy Optimization\nNN learns corrections to expert\nPhase-specific rewards',
                       COLORS['neural'], fontsize=8)

    # Arrows
    draw_arrow(ax, (4.5, 5.25), (5, 5.25))
    draw_arrow(ax, (9, 5.25), (9.5, 5.25))

    # PPO Hyperparameters box
    hyper_box = FancyBboxPatch((0.5, 0.5), 6, 2.8,
                                boxstyle="round,pad=0.02,rounding_size=0.1",
                                facecolor=COLORS['light'], edgecolor=COLORS['dark'],
                                linewidth=2, alpha=0.5)
    ax.add_patch(hyper_box)
    ax.text(3.5, 3, 'PPO Hyperparameters', ha='center', va='center',
            fontsize=10, fontweight='bold', color=COLORS['dark'])

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
        ax.text(1 + col*2.5, 2.4 - row*0.5, f'{name}: {val}', ha='left', va='center',
                fontsize=7, color=COLORS['dark'], family='monospace')

    # Reward structure box
    reward_box = FancyBboxPatch((7, 0.5), 6.5, 2.8,
                                 boxstyle="round,pad=0.02,rounding_size=0.1",
                                 facecolor=COLORS['light'], edgecolor=COLORS['dark'],
                                 linewidth=2, alpha=0.5)
    ax.add_patch(reward_box)
    ax.text(10.25, 3, 'Phase-Specific Rewards', ha='center', va='center',
            fontsize=10, fontweight='bold', color=COLORS['dark'])

    rewards = [
        'Takeoff: +speed, -centerline_err, +climb',
        'Cruise: +alt_track, +hdg_track, -ctrl_effort',
        'Approach: +glideslope, +localizer, +landing'
    ]
    for i, r in enumerate(rewards):
        ax.text(7.3, 2.4 - i*0.6, r, ha='left', va='center', fontsize=7, color=COLORS['dark'])

    plt.tight_layout()
    plt.savefig('/home/AIDA/docs/img/training_pipeline.png', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: training_pipeline.png")


# =============================================================================
# 4. SOFTWARE ARCHITECTURE DIAGRAM
# =============================================================================
def create_software_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(7, 9.5, 'SOFTWARE PACKAGE ARCHITECTURE', ha='center', va='center',
            fontsize=14, fontweight='bold', color=COLORS['dark'])

    # aida_sim package
    pkg1 = FancyBboxPatch((0.5, 5), 3, 4,
                           boxstyle="round,pad=0.02,rounding_size=0.1",
                           facecolor=COLORS['primary'], edgecolor='white',
                           linewidth=2, alpha=0.9)
    ax.add_patch(pkg1)
    ax.text(2, 8.5, 'aida_sim/', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')
    modules1 = ['env/', '  flight_env', '  residual_env_v2', 'dynamics/', 'systems/', 'io/telemetry']
    for i, m in enumerate(modules1):
        ax.text(1, 7.8 - i*0.45, m, ha='left', va='center', fontsize=7, color='white', family='monospace')

    # scripts package
    pkg2 = FancyBboxPatch((4, 5), 3, 4,
                           boxstyle="round,pad=0.02,rounding_size=0.1",
                           facecolor=COLORS['secondary'], edgecolor='white',
                           linewidth=2, alpha=0.9)
    ax.add_patch(pkg2)
    ax.text(5.5, 8.5, 'scripts/', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')
    modules2 = ['train_*.py', 'run_*.py', 'generate_*.py', 'test_*.py', 'triangle_controller.py']
    for i, m in enumerate(modules2):
        ax.text(4.5, 7.8 - i*0.45, m, ha='left', va='center', fontsize=7, color='white', family='monospace')

    # viewer package
    pkg3 = FancyBboxPatch((7.5, 5), 3, 4,
                           boxstyle="round,pad=0.02,rounding_size=0.1",
                           facecolor=COLORS['success'], edgecolor='white',
                           linewidth=2, alpha=0.9)
    ax.add_patch(pkg3)
    ax.text(9, 8.5, 'viewer/', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')
    modules3 = ['public/', '  index.html', '  viewer.js', '  engine_sound.js', 'components/']
    for i, m in enumerate(modules3):
        ax.text(8, 7.8 - i*0.45, m, ha='left', va='center', fontsize=7, color='white', family='monospace')

    # gpu-flight-dynamics
    pkg4 = FancyBboxPatch((11, 5), 2.5, 4,
                           boxstyle="round,pad=0.02,rounding_size=0.1",
                           facecolor=COLORS['warning'], edgecolor='white',
                           linewidth=2, alpha=0.9)
    ax.add_patch(pkg4)
    ax.text(12.25, 8.5, 'gpu-flight-\ndynamics/', ha='center', va='center',
            fontsize=9, fontweight='bold', color='white')
    modules4 = ['python/', '  flight_dynamics', '  aircraft_db']
    for i, m in enumerate(modules4):
        ax.text(11.3, 7.3 - i*0.45, m, ha='left', va='center', fontsize=7, color='white', family='monospace')

    # Data flow arrows
    draw_arrow(ax, (3.5, 7), (4, 7))
    draw_arrow(ax, (7, 7), (7.5, 7))
    draw_arrow(ax, (10.5, 7), (11, 7))

    # Bottom section - models and checkpoints
    create_rounded_box(ax, 0.5, 1, 4, 2.5, 'models/\n\nTrained weights (.zip)\nVersion controlled\nresidual_ppo_v2_7ctrl.zip',
                       COLORS['danger'], fontsize=8)

    create_rounded_box(ax, 5, 1, 4, 2.5, 'checkpoints/\n\nTraining checkpoints\nTensorBoard logs\nNot in git (.gitignore)',
                       COLORS['dark'], fontsize=8)

    create_rounded_box(ax, 9.5, 1, 4, 2.5, 'docs/\n\nArchitecture diagrams\nFlight screenshots\nREADME, guides',
                       COLORS['neural'], fontsize=8)

    plt.tight_layout()
    plt.savefig('/home/AIDA/docs/img/software_architecture.png', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: software_architecture.png")


# =============================================================================
# 5. DEVELOPMENT ENVIRONMENT DIAGRAM
# =============================================================================
def create_dev_environment():
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(6, 7.5, 'DEVELOPMENT ENVIRONMENT', ha='center', va='center',
            fontsize=14, fontweight='bold', color=COLORS['dark'])

    # Hardware box
    hw_box = FancyBboxPatch((0.5, 3.5), 5, 3.5,
                             boxstyle="round,pad=0.02,rounding_size=0.1",
                             facecolor=COLORS['dark'], edgecolor='white',
                             linewidth=2, alpha=0.9)
    ax.add_patch(hw_box)
    ax.text(3, 6.5, 'HARDWARE: Dell 7920', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')
    hw_specs = [
        'CPU: Intel Xeon (16+ cores)',
        'GPU: NVIDIA RTX 4060 (8GB)',
        'RAM: 32+ GB DDR4',
        'OS: Windows 11 + WSL2'
    ]
    for i, spec in enumerate(hw_specs):
        ax.text(1, 5.8 - i*0.5, spec, ha='left', va='center', fontsize=8, color='white')

    # Software stack
    sw_box = FancyBboxPatch((6.5, 3.5), 5, 3.5,
                             boxstyle="round,pad=0.02,rounding_size=0.1",
                             facecolor=COLORS['primary'], edgecolor='white',
                             linewidth=2, alpha=0.9)
    ax.add_patch(sw_box)
    ax.text(9, 6.5, 'SOFTWARE STACK', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')
    sw_specs = [
        'Python 3.10+',
        'PyTorch 2.x + CUDA 12.x',
        'Stable-Baselines3 2.x',
        'CuPy (GPU NumPy)',
        'Three.js (WebGL)'
    ]
    for i, spec in enumerate(sw_specs):
        ax.text(7, 5.8 - i*0.45, spec, ha='left', va='center', fontsize=8, color='white')

    # Performance benchmarks
    perf_box = FancyBboxPatch((0.5, 0.5), 11, 2.5,
                               boxstyle="round,pad=0.02,rounding_size=0.1",
                               facecolor=COLORS['success'], edgecolor='white',
                               linewidth=2, alpha=0.9)
    ax.add_patch(perf_box)
    ax.text(6, 2.7, 'PERFORMANCE BENCHMARKS', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')

    benchmarks = [
        ('GPU Sim (100 inst)', '50k steps/sec', '2,500x RT'),
        ('GPU Sim (1k inst)', '100k steps/sec', '5,000x RT'),
        ('GPU Sim (10k inst)', '569k steps/sec', '28,450x RT'),
        ('Training (16 envs)', '1,300 steps/sec', '-'),
    ]
    for i, (name, rate, rt) in enumerate(benchmarks):
        ax.text(1 + i*2.8, 2, name, ha='left', va='center', fontsize=7, color='white', fontweight='bold')
        ax.text(1 + i*2.8, 1.5, rate, ha='left', va='center', fontsize=7, color='white')
        ax.text(1 + i*2.8, 1.1, rt, ha='left', va='center', fontsize=6, color='#D1FAE5')

    plt.tight_layout()
    plt.savefig('/home/AIDA/docs/img/dev_environment.png', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: dev_environment.png")


# =============================================================================
# 6. CROSS-COUNTRY FLIGHT PROFILE
# =============================================================================
def create_flight_profile():
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    # Flight data points (distance nm, altitude ft)
    distances = [0, 2, 5, 8, 12, 16, 20, 24, 26, 28, 29.5, 30.5, 31]
    altitudes = [1400, 2000, 3500, 4500, 5500, 5500, 5500, 5500, 4500, 3000, 1800, 1450, 1400]

    # Plot altitude profile
    ax.fill_between(distances, altitudes, 1400, alpha=0.3, color=COLORS['primary'])
    ax.plot(distances, altitudes, 'o-', color=COLORS['primary'], linewidth=3, markersize=8)

    # Phase annotations
    phases = [
        (0.5, 1800, 'TAKEOFF', COLORS['success']),
        (3, 3000, 'CLIMB\n500 fpm', COLORS['primary']),
        (14, 5800, 'CRUISE @ 5,500 ft, 110 KTAS', COLORS['secondary']),
        (26, 4000, 'DESCENT', COLORS['warning']),
        (29.5, 2200, 'APPROACH', COLORS['danger']),
        (30.8, 1700, 'LAND', COLORS['dark']),
    ]

    for x, y, text, color in phases:
        ax.annotate(text, xy=(x, y), fontsize=9, fontweight='bold', color=color,
                    ha='center', va='bottom')

    # Airports
    ax.annotate('SN65\nLake Waltanna', xy=(0, 1400), xytext=(0, 800),
                fontsize=10, fontweight='bold', color=COLORS['dark'],
                ha='center', va='top',
                arrowprops=dict(arrowstyle='->', color=COLORS['dark']))

    ax.annotate('KHUT\nHutchinson', xy=(31, 1400), xytext=(31, 800),
                fontsize=10, fontweight='bold', color=COLORS['dark'],
                ha='center', va='top',
                arrowprops=dict(arrowstyle='->', color=COLORS['dark']))

    # Styling
    ax.set_xlim(-1, 32)
    ax.set_ylim(0, 6500)
    ax.set_xlabel('Distance (nautical miles)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Altitude (feet MSL)', fontsize=12, fontweight='bold')
    ax.set_title('CROSS-COUNTRY FLIGHT PROFILE: SN65 → KHUT (31 NM)',
                 fontsize=14, fontweight='bold', color=COLORS['dark'])
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#F8FAFC')

    # Add ground line
    ax.axhline(y=1400, color=COLORS['success'], linestyle='--', alpha=0.5, label='Ground Level (~1,400 ft)')
    ax.legend(loc='upper right')

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
