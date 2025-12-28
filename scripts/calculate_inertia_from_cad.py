#!/usr/bin/env python3
"""
Calculate Moments of Inertia from CAD Models

Uses trimesh library to analyze STL/mesh files and calculate:
- Mass
- Center of Gravity (CG)
- Moments of Inertia (Ixx, Iyy, Izz, Ixz)

For accurate results, the CAD model should:
1. Have proper density assigned (or use uniform density)
2. Be properly assembled with all components
3. Have CG at the correct location

Author: Kushal Koirala
Date: December 26, 2024
"""

import numpy as np
import trimesh
import argparse
from pathlib import Path

def calculate_inertia_from_mesh(mesh_file, density=1000.0, scale=1.0):
    """
    Calculate mass properties from mesh file.

    Args:
        mesh_file: Path to STL, OBJ, or other mesh file
        density: Material density in kg/m³
                 Examples: Aluminum=2700, ABS plastic=1040, Carbon fiber=1600
        scale: Scale factor if model units are not meters
               (e.g., scale=0.001 if model is in mm)

    Returns:
        dict with mass, CG, and inertia properties
    """
    print(f"\nLoading mesh: {mesh_file}")

    # Load mesh
    mesh = trimesh.load(mesh_file)

    # Apply scale if needed
    if scale != 1.0:
        mesh.apply_scale(scale)
        print(f"Applied scale factor: {scale}")

    # Check if mesh is watertight (closed volume)
    if not mesh.is_watertight:
        print("⚠️  WARNING: Mesh is not watertight (has holes)!")
        print("   Inertia calculations may be inaccurate.")
        print("   Consider fixing the mesh in CAD software.")

    # Calculate volume
    volume = mesh.volume  # m³
    print(f"Volume: {volume:.6f} m³ ({volume * 1e6:.2f} cm³)")

    # Calculate mass
    mass = volume * density  # kg
    print(f"Density: {density:.1f} kg/m³")
    print(f"Mass: {mass:.3f} kg ({mass * 2.205:.2f} lbs)")

    # Get center of mass
    center_of_mass = mesh.center_mass
    print(f"\nCenter of Mass (m):")
    print(f"  X: {center_of_mass[0]:.4f}")
    print(f"  Y: {center_of_mass[1]:.4f}")
    print(f"  Z: {center_of_mass[2]:.4f}")

    # Calculate moment of inertia tensor about origin
    # Trimesh returns inertia in the mesh's coordinate system
    inertia_tensor = mesh.moment_inertia

    print(f"\nInertia Tensor (kg·m²) about origin:")
    print(f"  {inertia_tensor[0, 0]:.4f}  {inertia_tensor[0, 1]:.4f}  {inertia_tensor[0, 2]:.4f}")
    print(f"  {inertia_tensor[1, 0]:.4f}  {inertia_tensor[1, 1]:.4f}  {inertia_tensor[1, 2]:.4f}")
    print(f"  {inertia_tensor[2, 0]:.4f}  {inertia_tensor[2, 1]:.4f}  {inertia_tensor[2, 2]:.4f}")

    # Extract principal moments
    Ixx = inertia_tensor[0, 0]
    Iyy = inertia_tensor[1, 1]
    Izz = inertia_tensor[2, 2]
    Ixy = inertia_tensor[0, 1]
    Ixz = inertia_tensor[0, 2]
    Iyz = inertia_tensor[1, 2]

    print(f"\nPrincipal Moments of Inertia (kg·m²):")
    print(f"  Ixx (roll):  {Ixx:.4f}")
    print(f"  Iyy (pitch): {Iyy:.4f}")
    print(f"  Izz (yaw):   {Izz:.4f}")

    print(f"\nProducts of Inertia (kg·m²):")
    print(f"  Ixy: {Ixy:.4f}")
    print(f"  Ixz: {Ixz:.4f}")
    print(f"  Iyz: {Iyz:.4f}")

    # Radii of gyration
    if mass > 0:
        kxx = np.sqrt(Ixx / mass)
        kyy = np.sqrt(Iyy / mass)
        kzz = np.sqrt(Izz / mass)

        print(f"\nRadii of Gyration (m):")
        print(f"  kxx: {kxx:.4f}")
        print(f"  kyy: {kyy:.4f}")
        print(f"  kzz: {kzz:.4f}")

    return {
        'mass': mass,
        'volume': volume,
        'density': density,
        'center_of_mass': center_of_mass,
        'inertia_tensor': inertia_tensor,
        'Ixx': Ixx,
        'Iyy': Iyy,
        'Izz': Izz,
        'Ixy': Ixy,
        'Ixz': Ixz,
        'Iyz': Iyz,
    }


def compare_with_database(calculated, database_values):
    """Compare calculated values with database values."""
    print("\n" + "="*70)
    print("  COMPARISON WITH AIRCRAFT DATABASE")
    print("="*70)

    db_mass = database_values.get('mass', 3.63)
    db_Ixx = database_values.get('Ixx', 0.42)
    db_Iyy = database_values.get('Iyy', 0.55)
    db_Izz = database_values.get('Izz', 0.78)

    print(f"\n{'Property':<15} {'Calculated':<15} {'Database':<15} {'Difference':<15}")
    print("-"*70)

    mass_diff = (calculated['mass'] - db_mass) / db_mass * 100
    print(f"{'Mass (kg)':<15} {calculated['mass']:<15.3f} {db_mass:<15.3f} {mass_diff:+.1f}%")

    Ixx_diff = (calculated['Ixx'] - db_Ixx) / db_Ixx * 100
    print(f"{'Ixx (kg·m²)':<15} {calculated['Ixx']:<15.4f} {db_Ixx:<15.4f} {Ixx_diff:+.1f}%")

    Iyy_diff = (calculated['Iyy'] - db_Iyy) / db_Iyy * 100
    print(f"{'Iyy (kg·m²)':<15} {calculated['Iyy']:<15.4f} {db_Iyy:<15.4f} {Iyy_diff:+.1f}%")

    Izz_diff = (calculated['Izz'] - db_Izz) / db_Izz * 100
    print(f"{'Izz (kg·m²)':<15} {calculated['Izz']:<15.4f} {db_Izz:<15.4f} {Izz_diff:+.1f}%")

    print("\n" + "="*70)


def generate_aircraft_database_code(props, name="udaan"):
    """Generate Python code for aircraft_database.py"""
    print("\n" + "="*70)
    print("  AIRCRAFT DATABASE CODE")
    print("="*70)
    print("\nCopy this into aircraft_database.py:")
    print("-"*70)

    print(f"""
    mass=MassProperties(
        mass={props['mass']:.3f},              # kg (from CAD)
        Ixx={props['Ixx']:.4f},               # kg·m² (roll)
        Iyy={props['Iyy']:.4f},               # kg·m² (pitch)
        Izz={props['Izz']:.4f},               # kg·m² (yaw)
        Ixz={props['Ixz']:.4f},               # kg·m² (product of inertia)
    ),
    """)
    print("-"*70)


def main():
    parser = argparse.ArgumentParser(
        description='Calculate moments of inertia from CAD mesh files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with STL file
  python calculate_inertia_from_cad.py Udaan-Product4.stl

  # Specify density (ABS plastic)
  python calculate_inertia_from_cad.py Udaan-Product4.stl --density 1040

  # If CAD model is in mm instead of meters
  python calculate_inertia_from_cad.py Udaan.stl --scale 0.001

  # Use STEP file (requires additional dependencies)
  python calculate_inertia_from_cad.py Udaan.stp --density 1500

Material Densities (kg/m³):
  ABS Plastic:      1040
  PLA Plastic:      1250
  Carbon Fiber:     1600
  Aluminum:         2700
  Balsa Wood:       150
  Plywood:          550
  Foam (EPS):       20-30
        """
    )

    parser.add_argument('mesh_file', type=str,
                       help='Path to mesh file (STL, OBJ, etc.)')
    parser.add_argument('--density', type=float, default=1000.0,
                       help='Material density in kg/m³ (default: 1000)')
    parser.add_argument('--scale', type=float, default=1.0,
                       help='Scale factor if units are not meters (default: 1.0)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare with database values')

    args = parser.parse_args()

    # Check if file exists
    mesh_path = Path(args.mesh_file)
    if not mesh_path.exists():
        print(f"Error: File not found: {args.mesh_file}")
        return 1

    print("="*70)
    print("  CALCULATE INERTIA FROM CAD MODEL")
    print("="*70)

    # Calculate properties
    props = calculate_inertia_from_mesh(
        args.mesh_file,
        density=args.density,
        scale=args.scale
    )

    # Compare with database if requested
    if args.compare:
        database_values = {
            'mass': 3.63,  # kg (from aircraft_database.py)
            'Ixx': 0.42,
            'Iyy': 0.55,
            'Izz': 0.78,
        }
        compare_with_database(props, database_values)

    # Generate code
    generate_aircraft_database_code(props)

    print("\n" + "="*70)
    print("  NOTES")
    print("="*70)
    print("""
1. If mass differs significantly from expected:
   - Check if all components are included in the CAD model
   - Verify density value is correct for your materials
   - Check if scale factor is correct

2. If mesh is not watertight:
   - Fix holes in CAD software (e.g., SolidWorks: Tools → Check)
   - Export as STEP file instead of STL for better accuracy

3. For multi-material assemblies:
   - Export each part separately with correct density
   - Calculate inertia for each part
   - Use parallel axis theorem to combine:
     I_total = I_part + m_part * d²
     where d is distance from part CG to total CG

4. Coordinate system alignment:
   - X-axis: Forward (nose direction)
   - Y-axis: Right wing
   - Z-axis: Down (NED frame)

   If your CAD uses different axes, rotate the inertia tensor accordingly.
    """)

    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
