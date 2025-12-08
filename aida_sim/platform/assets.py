from pathlib import Path

# Expected asset locations (updated to provided files)
DEFAULT_STEP = Path(__file__).resolve().parent.parent / ".." / "Udaan.stp"
DEFAULT_COLLISION = Path(__file__).resolve().parent.parent / ".." / "Udaan-Product4.stl"
DEFAULT_RENDER = Path(__file__).resolve().parent.parent / ".." / "Udaan-Product4.gltf"


def resolve_collision_mesh(path: Path | None = None) -> Path:
    # Provide the collision mesh path (e.g., STL). Caller ensures the file exists.
    return Path(path) if path else DEFAULT_COLLISION


def resolve_render_mesh(path: Path | None = None) -> Path:
    # Provide the render mesh path (glTF for the viewer). Caller ensures the file exists.
    return Path(path) if path else DEFAULT_RENDER
