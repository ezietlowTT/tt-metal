"""Unit tests for ttnn.hardware — HardwareConfig and KNOWN_CONFIGS."""
import pytest
import sys
import os

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "hardware",
    os.path.join(os.path.dirname(__file__), "..", "ttnn", "ttnn", "hardware.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
HardwareConfig = _mod.HardwareConfig
KNOWN_CONFIGS = _mod.KNOWN_CONFIGS


def test_import():
    assert HardwareConfig is not None
    assert KNOWN_CONFIGS is not None


def test_known_configs_present():
    assert "bh_p150_1card" in KNOWN_CONFIGS
    assert "bh_p150_2card" in KNOWN_CONFIGS
    assert "gs_e150_1card" in KNOWN_CONFIGS


def test_grid_shape_p150_32heads():
    cols, rows = KNOWN_CONFIGS["bh_p150_1card"].grid_shape(32)
    assert (cols, rows) == (8, 4)
    assert type(cols) is int
    assert type(rows) is int


def test_grid_shape_gs_e150_32heads():
    cols, rows = KNOWN_CONFIGS["gs_e150_1card"].grid_shape(32)
    assert cols * rows == 32
    assert cols <= 9
    assert rows <= 12
    assert type(cols) is int
    assert type(rows) is int


def test_grid_shape_overflow_raises():
    # 131 is prime; only factor ≤ 13 is 1, giving rows=131 which exceeds grid_y=10
    with pytest.raises(ValueError, match="Cannot map 131 heads"):
        KNOWN_CONFIGS["bh_p150_1card"].grid_shape(131)


def test_l1_budget():
    budget = KNOWN_CONFIGS["bh_p150_1card"].l1_budget()
    assert budget == int(1_572_864 * 0.915)  # 1_439_170


def test_frozen():
    cfg = KNOWN_CONFIGS["bh_p150_1card"]
    with pytest.raises(Exception):
        cfg.grid_x = 999
