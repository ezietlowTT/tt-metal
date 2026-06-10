from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareConfig:
    name: str           # e.g. "bh_p150_1card"
    grid_x: int         # usable Tensix columns
    grid_y: int         # usable Tensix rows
    l1_bytes: int       # full L1 SRAM per core in bytes (1_572_864 for p150)
    dram_bytes: int     # total DRAM in bytes
    tile_size: int = 32 # element tile dimension (32 for all current TT hardware)

    def l1_budget(self) -> int:
        return int(self.l1_bytes * 0.915)

    def grid_shape(self, n_heads: int) -> tuple[int, int]:
        """Returns plain Python ints -- required for ttl kernel closure capture."""
        for cols in range(min(n_heads, self.grid_x), 0, -1):
            if n_heads % cols == 0:
                rows = n_heads // cols
                if rows <= self.grid_y:
                    return int(cols), int(rows)
        raise ValueError(
            f"Cannot map {n_heads} heads onto {self.name} "
            f"({self.grid_x}x{self.grid_y} grid)."
        )


KNOWN_CONFIGS: dict[str, HardwareConfig] = {
    "bh_p150_1card": HardwareConfig(
        name="bh_p150_1card", grid_x=13, grid_y=10,
        l1_bytes=1_572_864,
        dram_bytes=12 * 1024**3,
    ),
    "bh_p150_2card": HardwareConfig(
        name="bh_p150_2card", grid_x=13, grid_y=10,
        l1_bytes=1_572_864,
        dram_bytes=24 * 1024**3,
    ),
    "gs_e150_1card": HardwareConfig(
        name="gs_e150_1card", grid_x=9, grid_y=12,
        l1_bytes=1_048_576,
        dram_bytes=8 * 1024**3,
    ),
}
