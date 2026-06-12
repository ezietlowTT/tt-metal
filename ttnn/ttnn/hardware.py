from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareConfig:
    name: str  # e.g. "bh_p150_1card"
    grid_x: int  # usable Tensix columns
    grid_y: int  # usable Tensix rows
    l1_bytes: int  # full L1 SRAM per core in bytes (1_572_864 for p150)
    dram_bytes: int  # total on-card DRAM in bytes (nominal board capacity)
    tile_size: int = 32  # element tile dimension (32 for all current TT hardware)

    def l1_budget(self) -> int:
        return int(self.l1_bytes * 0.915)

    def dram_budget(self) -> int:
        """Usable DRAM for model weights + KV cache + activations, after a margin
        for firmware/runtime reservation. Capacity admission checks must use THIS,
        not dram_bytes — overstating risks OOM. (p150a reports ~31.3 GB available
        of 32 GB nominal; 0.90 keeps the budget safely under that either way.)"""
        return int(self.dram_bytes * 0.90)

    def grid_shape(self, n_heads: int) -> tuple[int, int]:
        """Returns plain Python ints -- required for ttl kernel closure capture."""
        for cols in range(min(n_heads, self.grid_x), 0, -1):
            if n_heads % cols == 0:
                rows = n_heads // cols
                if rows <= self.grid_y:
                    return int(cols), int(rows)
        raise ValueError(f"Cannot map {n_heads} heads onto {self.name} " f"({self.grid_x}x{self.grid_y} grid).")


def detect_hardware() -> "HardwareConfig":
    """Open device briefly, read grid dimensions, match against KNOWN_CONFIGS.

    Closes the device handle before returning so callers can open their own.
    Matches on compute grid (x, y) — sufficient to identify all current configs.

    NOTE: dram_bytes is a per-board CONSTANT, not read from the device. The
    MeshDevice returned by open_device() exposes no total-DRAM API
    (dram_size_per_bank()/num_dram_channels() do not exist on it; only
    dram_grid_size(), which gives the 8-bank count but no per-bank size). So the
    value in KNOWN_CONFIGS must be kept correct by hand against the real board.
    """
    import ttnn as _ttnn

    device = _ttnn.open_device(device_id=0)
    try:
        g = device.compute_with_storage_grid_size()
    finally:
        _ttnn.close_device(device)

    for cfg in KNOWN_CONFIGS.values():
        if cfg.grid_x == g.x and cfg.grid_y == g.y:
            return cfg

    raise RuntimeError(
        f"Unknown hardware: grid={g.x}x{g.y}. "
        f"Add an entry to ttnn/ttnn/hardware.py KNOWN_CONFIGS and file an issue."
    )


KNOWN_CONFIGS: dict[str, HardwareConfig] = {
    # Blackhole p150a: 32 GB GDDR6 (8 channels). Confirmed on the 208 box via
    # tt-smi: board_type=p150a, ENABLED_GDDR=0xff, Memory≈31.3 GB.
    "bh_p150_1card": HardwareConfig(
        name="bh_p150_1card",
        grid_x=13,
        grid_y=10,
        l1_bytes=1_572_864,
        dram_bytes=32 * 1024**3,
    ),
    "bh_p150_2card": HardwareConfig(
        name="bh_p150_2card",
        grid_x=13,
        grid_y=10,
        l1_bytes=1_572_864,
        dram_bytes=64 * 1024**3,
    ),
    "gs_e150_1card": HardwareConfig(
        name="gs_e150_1card",
        grid_x=9,
        grid_y=12,
        l1_bytes=1_048_576,
        dram_bytes=8 * 1024**3,
    ),
}
