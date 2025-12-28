# Extracting Mass Properties from CAD for Udaan

## Why Direct CAD Export is Best

The STL and GLTF files have quality issues (mesh not watertight, missing data). The **most accurate** method is to extract mass properties directly from your CAD software, which has:
- Exact geometry (no mesh approximation)
- Proper material assignments
- Assembly structure preserved
- No conversion artifacts

---

## Method 1: SolidWorks (Recommended)

### Step 1: Assign Materials
1. Right-click on each part → **Material**
2. Assign appropriate materials:
   - **Airframe/wings**: ABS Plastic (ρ = 1040 kg/m³) or Carbon Fiber (ρ = 1600 kg/m³)
   - **Electronics**: Custom material (ρ ≈ 2000 kg/m³)
   - **Battery**: Custom material (adjust density for actual mass)

### Step 2: Calculate Mass Properties
1. Tools → **Mass Properties**
2. Set options:
   - ☑ Include hidden bodies
   - ☑ Create Center of Mass feature
   - Output coordinate system: **Default** (or define NED frame)

### Step 3: Record Values
Copy these values from the Mass Properties window:

```
Mass = _____ kg

Center of mass (mm):
  X = _____
  Y = _____
  Z = _____

Moments of inertia (kg·mm²) - taken at the CENTER OF MASS:
  Ixx = _____
  Iyy = _____
  Izz = _____
  Ixy = _____
  Ixz = _____
  Iyz = _____
```

### Step 4: Convert Units
SolidWorks gives inertia in **kg·mm²**. Convert to **kg·m²**:

```
Ixx (kg·m²) = Ixx (kg·mm²) × 1e-6
Iyy (kg·m²) = Iyy (kg·mm²) × 1e-6
Izz (kg·m²) = Izz (kg·mm²) × 1e-6
```

### Step 5: Coordinate System Alignment

**SolidWorks Default:**
- X = Right
- Y = Up
- Z = Out (toward viewer)

**NED Frame (required for flight dynamics):**
- X = Forward (nose direction)
- Y = Right (right wing)
- Z = Down

**To align:**
1. In SolidWorks: Insert → Reference Geometry → Coordinate System
2. Define origin at desired CG location
3. Align axes:
   - X-axis along fuselage (forward)
   - Y-axis along right wing
   - Z-axis downward
4. Recalculate Mass Properties using this coordinate system

---

## Method 2: Fusion 360

### Steps:
1. Assign materials to all components
2. Browser → Right-click assembly → **Physical Material**
3. Inspect → **Center of Mass** (creates visual indicator)
4. Browser → Right-click assembly → **Properties**
5. View **Physical** tab for mass and inertia

**Note:** Fusion 360 displays inertia in **kg·cm²**, convert:
```
Ixx (kg·m²) = Ixx (kg·cm²) × 1e-4
```

---

## Method 3: CATIA

### Steps:
1. Analyze → **Measure Inertia**
2. Select entire assembly
3. Choose coordinate system
4. Export results

---

## Method 4: Using Python with STEP File (Advanced)

If you have **FreeCAD** installed, we can script the extraction:

```python
import FreeCAD
import Part

# Load STEP file
doc = FreeCAD.open("/home/AIDA/Udaan.stp")

# Get all solid objects
solids = [obj for obj in doc.Objects if hasattr(obj, 'Shape') and obj.Shape.Solids]

# Calculate properties
total_mass = 0
total_volume = 0

for solid in solids:
    shape = solid.Shape
    volume = shape.Volume / 1e9  # mm³ to m³
    density = 1040  # kg/m³ (ABS)
    mass = volume * density

    total_mass += mass
    total_volume += volume

# Get center of mass and inertia
assembly_shape = Part.makeCompound([obj.Shape for obj in solids])
com = assembly_shape.CenterOfMass
inertia = assembly_shape.MatrixOfInertia

print(f"Mass: {total_mass:.3f} kg")
print(f"Center of Mass: {com}")
print(f"Inertia: {inertia}")
```

---

## Quick Reference: Expected Values

Based on PropShox documentation and previous estimates:

| Property | Expected Value | Notes |
|----------|---------------|-------|
| Mass | 3.5 - 4.0 kg | Including payload |
| Ixx (roll) | 0.3 - 0.5 kg·m² | Smallest (about wing span) |
| Iyy (pitch) | 0.4 - 0.6 kg·m² | Medium (fuselage length) |
| Izz (yaw) | 0.6 - 0.9 kg·m² | Largest (combined) |
| Ixz | < 0.1 kg·m² | Small for symmetric design |

If your CAD results differ significantly:
- Check material assignments
- Verify all components are included
- Check coordinate system alignment

---

## What to Do Next

Once you have the values from your CAD software:

1. **Update aircraft_database.py:**
   ```python
   mass=MassProperties(
       mass=YOUR_VALUE_HERE,
       Ixx=YOUR_VALUE_HERE,
       Iyy=YOUR_VALUE_HERE,
       Izz=YOUR_VALUE_HERE,
       Ixz=YOUR_VALUE_HERE,  # Usually small for symmetric aircraft
   ),
   ```

2. **Regenerate BC dataset:**
   ```bash
   cd /home/AIDA
   python scripts/generate_bc_dataset_gpu.py
   ```

3. **Retrain PPO:**
   ```bash
   python scripts/train_ppo_flight.py --device cuda
   ```

---

## Troubleshooting

**Q: Mass is much higher than expected (e.g., 10+ kg)**
- Check if density is in g/cm³ instead of kg/m³
- Verify scale (should be in meters, not mm)
- Check for duplicate components in assembly

**Q: Ixx, Iyy, Izz are negative**
- This indicates inverted mesh or wrong coordinate system
- Recalculate with proper reference frame

**Q: Products of inertia (Ixy, Ixz, Iyz) are large**
- Aircraft should be mostly symmetric → Ixy, Iyz ≈ 0
- Ixz may be small but non-zero
- Large values suggest misaligned coordinate system

---

**Author:** Claude Code
**Date:** December 27, 2024
**For:** Udaan UAV (AIDA Project)
