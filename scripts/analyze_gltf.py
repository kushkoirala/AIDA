#!/usr/bin/env python3
"""Analyze GLTF file for mass properties"""
import trimesh
import numpy as np

print('='*70)
print('  CALCULATE INERTIA FROM CAD MODEL (GLTF)')
print('='*70)

# Load GLTF file
print('\nLoading GLTF: Udaan-Product4.gltf')
scene = trimesh.load('../Udaan-Product4.gltf')

# GLTF files load as Scene objects with multiple meshes
print(f'Loaded as: {type(scene).__name__}')

if isinstance(scene, trimesh.Scene):
    # Combine all geometries into a single mesh
    print(f'Number of geometries: {len(scene.geometry)}')

    # Dump the scene into a single mesh
    mesh = scene.dump(concatenate=True)
    print(f'Combined into single mesh')
else:
    mesh = scene

# Apply scale (mm to m)
scale = 0.001
mesh.apply_scale(scale)
print(f'Applied scale factor: {scale}')

# Check mesh properties
print(f'\nMesh properties:')
print(f'  Watertight: {mesh.is_watertight}')
print(f'  Volume: {mesh.volume:.6f} m³ ({mesh.volume * 1e6:.2f} cm³)')

if not mesh.is_watertight:
    print('  WARNING: Mesh is not watertight (has holes)!')
    print('  Attempting repair...')
    mesh.fill_holes()
    mesh.fix_normals()
    if mesh.volume < 0:
        mesh.invert()
    print(f'  Volume after repair: {mesh.volume:.6f} m³')

# Calculate properties with ABS plastic density
density = 1040.0  # kg/m³
volume = abs(mesh.volume)
mass = volume * density

print('\n' + '='*70)
print('  MASS PROPERTIES')
print('='*70)
print(f'\nVolume: {volume:.6f} m³ ({volume * 1e6:.2f} cm³)')
print(f'Density: {density:.1f} kg/m³ (ABS plastic)')
print(f'Mass: {mass:.3f} kg ({mass * 2.205:.2f} lbs)')

# Center of mass
com = mesh.center_mass
print(f'\nCenter of Mass (m):')
print(f'  X: {com[0]:.4f}')
print(f'  Y: {com[1]:.4f}')
print(f'  Z: {com[2]:.4f}')

# Inertia tensor
I = mesh.moment_inertia * density
print(f'\nInertia Tensor (kg·m²) about origin:')
for i in range(3):
    print(f'  [{I[i,0]:8.4f}  {I[i,1]:8.4f}  {I[i,2]:8.4f}]')

Ixx = abs(I[0,0])
Iyy = abs(I[1,1])
Izz = abs(I[2,2])
Ixy = I[0,1]
Ixz = I[0,2]
Iyz = I[1,2]

print(f'\nPrincipal Moments of Inertia (kg·m²):')
print(f'  Ixx (roll):  {Ixx:.4f}')
print(f'  Iyy (pitch): {Iyy:.4f}')
print(f'  Izz (yaw):   {Izz:.4f}')

print(f'\nProducts of Inertia (kg·m²):')
print(f'  Ixy: {Ixy:.4f}')
print(f'  Ixz: {Ixz:.4f}')
print(f'  Iyz: {Iyz:.4f}')

# Radii of gyration
if mass > 0:
    kxx = np.sqrt(Ixx / mass)
    kyy = np.sqrt(Iyy / mass)
    kzz = np.sqrt(Izz / mass)

    print(f'\nRadii of Gyration (m):')
    print(f'  kxx: {kxx:.4f}')
    print(f'  kyy: {kyy:.4f}')
    print(f'  kzz: {kzz:.4f}')

# Compare with database
print('\n' + '='*70)
print('  COMPARISON WITH CURRENT DATABASE')
print('='*70)
db_mass = 3.63
db_Ixx = 0.42
db_Iyy = 0.55
db_Izz = 0.78

print(f'\nProperty        CAD Value       Database        Difference')
print('-'*70)
print(f'Mass (kg)       {mass:<15.3f} {db_mass:<15.3f} {(mass-db_mass)/db_mass*100:+.1f}%')
print(f'Ixx (kg·m²)     {Ixx:<15.4f} {db_Ixx:<15.4f} {(Ixx-db_Ixx)/db_Ixx*100:+.1f}%')
print(f'Iyy (kg·m²)     {Iyy:<15.4f} {db_Iyy:<15.4f} {(Iyy-db_Iyy)/db_Iyy*100:+.1f}%')
print(f'Izz (kg·m²)     {Izz:<15.4f} {db_Izz:<15.4f} {(Izz-db_Izz)/db_Izz*100:+.1f}%')

print('\n' + '='*70)
print('  CODE FOR aircraft_database.py')
print('='*70)
print(f'''
    mass=MassProperties(
        mass={mass:.3f},              # kg (from CAD)
        Ixx={Ixx:.4f},               # kg·m² (roll)
        Iyy={Iyy:.4f},               # kg·m² (pitch)
        Izz={Izz:.4f},               # kg·m² (yaw)
        Ixz={Ixz:.4f},               # kg·m² (product of inertia)
    ),
''')

print('='*70)
print('\nNOTE: Verify coordinate system alignment with NED frame:')
print('  X = Forward (nose), Y = Right wing, Z = Down')
print('  Rotate tensor if CAD uses different convention.')
print('='*70)
