#!/usr/bin/env python
"""Sync provided CAD exports into runtime locations.
- Copies glTF to viewer/public/
- Copies STL to aida_sim/assets/
"""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent.parent
GLTF_SRC = ROOT / "Udaan-Product4.gltf"
STL_SRC = ROOT / "Udaan-Product4.stl"
GLTF_DST = ROOT / "viewer" / "public" / "Udaan-Product4.gltf"
STL_DST = ROOT / "aida_sim" / "assets" / "Udaan-Product4.stl"


def main():
    GLTF_DST.parent.mkdir(parents=True, exist_ok=True)
    STL_DST.parent.mkdir(parents=True, exist_ok=True)

    if GLTF_SRC.exists():
        shutil.copy2(GLTF_SRC, GLTF_DST)
        print(f"Copied {GLTF_SRC} -> {GLTF_DST}")
    else:
        print(f"Missing glTF source: {GLTF_SRC}")

    if STL_SRC.exists():
        shutil.copy2(STL_SRC, STL_DST)
        print(f"Copied {STL_SRC} -> {STL_DST}")
    else:
        print(f"Missing STL source: {STL_SRC}")


if __name__ == "__main__":
    main()
