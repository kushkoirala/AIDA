# Udaan Inertia Properties from CATIA

## Raw Data from CATIA "Measure Inertia"

**Selection:** Udaan Assembly
**Calculation mode:** Exact
**Type:** Volume

### Characteristics:
- **Volume:** 1874.293 in³
- **Area:** 2.822 m²
- **Mass:** 67.713 lb = **30.71 kg**
- **Density:** Not uniform (multi-material assembly)

### Center of Gravity (inches):
- **Gx:** 274.116 in = 6.963 m
- **Gy:** 41.917 in = 1.065 m
- **Gz:** 48.755 in = 1.238 m

### Inertia Matrix / G (at CG):
**Units shown: kg·gxm²** (needs interpretation)

```
IxxG = 1.618 kg·gxm²      IyyG = 1.147 kg·gxm²      IzzG = 2.704 kg·gxm²
IxyG = -0.009 kg·gxm²     IxzG = 0.068 kg·gxm²      IyzG = 8.84e-5 kg·gxm²
```

### Principal Moments / G:
- **M1:** 1.147 kg·gxm²
- **M2:** 1.614 kg·gxm²
- **M3:** 2.709 kg·gxm²

---

## Analysis and Issues

### Issue 1: Mass Discrepancy
**CATIA reports:** 30.71 kg
**Expected (from PropShox):** 3.63 kg
**Ratio:** 8.46x higher

**Possible causes:**
1. **Density settings incorrect** - Materials may be set to actual density instead of scaled
2. **Scale factor** - Model may be in different units (mm vs inches)
3. **Missing components vs extra components** - Assembly may include test stand or extra parts
4. **Solid vs hollow structures** - Actual aircraft has hollow tubes, foam core, etc.

### Issue 2: Unit "kg·gxm²"
The unit **"gxm²"** is non-standard. In CATIA, this likely means:
- **gx** = unit of length in the x-direction (possibly mm or in)
- Combined with kg, this creates a moment of inertia unit

**Most likely interpretation:**
- If model is in **inches**: kg·gxm² = kg·in² → need to convert to kg·m²
- If model is in **mm**: kg·gxm² = kg·mm² → need to convert to kg·m²

### Issue 3: Coordinate System
The CG location (274 in, 41 in, 48 in) suggests the origin is far from the aircraft CG. For flight dynamics, we need inertia **about the CG**, which CATIA provides in the "Inertia / G" section.

---

## Unit Conversion Strategies

### Strategy 1: Assume model is in INCHES (most likely based on CG values)

If the inertia is in **kg·in²**:

```python
# Convert kg·in² to kg·m²
conversion_factor = (0.0254)**2  # inches to meters squared
conversion_factor = 0.00064516

Ixx = 1.618 * 0.00064516 = 0.001044 kg·m²
Iyy = 1.614 * 0.00064516 = 0.001041 kg·m²
Izz = 2.704 * 0.00064516 = 0.001745 kg·m²
```

**BUT** - these values are WAY too small for a 30 kg aircraft!

### Strategy 2: The density scale is wrong, but inertia ratios are correct

If we assume the **shape** is correct but density is 8.46x too high:

```python
# Scale factor for inertia (mass appears in I = m*r²)
# If mass is 8.46x too high, and it's uniformly distributed,
# then I is also 8.46x too high

scale_factor = 3.63 / 30.71  # = 0.118

# Assuming the units are already in kg·m² (which would make sense for the magnitude):
Ixx = 1.618 * 0.118 = 0.191 kg·m²
Iyy = 1.614 * 0.118 = 0.190 kg·m²
Izz = 2.704 * 0.118 = 0.319 kg·m²
```

These are still **smaller** than current database values (0.42, 0.55, 0.78).

### Strategy 3: Direct interpretation (kg·m² already)

If CATIA is showing **kg·m²** directly (and "gxm" is just a display quirk):

```python
Ixx = 1.618 kg·m²
Iyy = 1.614 kg·m²
Izz = 2.704 kg·m²
```

These are **larger** than current database but within plausible range for a heavier aircraft.

---

## Recommended Actions

### 1. Verify CATIA Units Settings
Check in CATIA:
- **Tools → Options → General → Parameters and Measure → Units**
- Confirm if Length = inches or mm or m
- Confirm if Mass = kg or lb
- Confirm inertia units

### 2. Check Material Density Assignments
In CATIA:
- **Analyze → Mass Properties** (different from Measure Inertia)
- List all parts and their materials
- Verify density values are reasonable:
  - ABS plastic: ~1.04 g/cm³
  - Carbon fiber: ~1.6 g/cm³
  - Electronics: ~2.0 g/cm³

### 3. Scale Correction Options

**Option A: Correct the mass in CATIA**
- Adjust material densities so total mass = 3.63 kg
- Re-run Measure Inertia
- This will give correctly scaled inertia values

**Option B: Scale the inertia mathematically**
If geometry is correct but density/mass is wrong:
```python
correct_mass = 3.63  # kg
catia_mass = 30.71   # kg
scale = correct_mass / catia_mass

Ixx_corrected = Ixx_catia * scale
Iyy_corrected = Iyy_catia * scale
Izz_corrected = Izz_catia * scale
```

---

## Questions for User

1. **What are the CATIA model units?** (Tools → Options → Units)
   - Length: inches, mm, or meters?
   - Mass: kg or lb?
   - Inertia: what does it show?

2. **Are all materials assigned?**
   - Does the assembly include only the aircraft, or also test stand/fixtures?
   - Are hollow structures modeled as solid (would increase mass)?

3. **Is 67.713 lb the actual aircraft mass?**
   - If yes, then database value (3.63 kg = 8 lb) is wrong
   - If no, then CATIA densities need correction

4. **Can you export the Mass Properties report?**
   - File → Export → or screenshot of full mass properties breakdown

---

## Temporary Solution: Use Current Best Estimate

Given the uncertainties, I recommend using the **principal moments directly with mass correction**:

```python
# From CATIA principal moments (assuming kg·m² with mass correction)
catia_mass = 30.71  # kg (from CATIA)
correct_mass = 3.63 # kg (from PropShox/database)
scale = correct_mass / catia_mass  # 0.118

# Scale the principal moments
M1_scaled = 1.147 * scale = 0.135 kg·m²
M2_scaled = 1.614 * scale = 0.190 kg·m²
M3_scaled = 2.709 * scale = 0.320 kg·m²

# Assign to axes (need to verify orientation)
# Typically for aircraft: M1=roll (smallest), M2=pitch, M3=yaw (largest)
Ixx = 0.135  # Roll (about wing span)
Iyy = 0.190  # Pitch (about lateral axis)
Izz = 0.320  # Yaw (about vertical axis)
```

These are **more conservative** (smaller) than current database values, which might be safer for initial testing.

---

## Next Steps

1. **Verify CATIA units** - most critical
2. **Check material assignments** - explains mass discrepancy
3. **Confirm coordinate system** - verify axes match NED frame
4. **Export full mass properties** - breakdown by component

Once confirmed, update [aircraft_database.py](\\wsl.localhost\Ubuntu-22.04\home\AIDA\gpu-flight-dynamics\python\aircraft_database.py).

---

**Generated:** December 27, 2024
**Source:** CATIA V5 Measure Inertia (udaan.png)
