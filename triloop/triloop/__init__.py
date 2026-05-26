"""triloop — three-loop magnetic-field array analysis."""
from .geometry import (
    LOOP_NORMALS_DEFAULT,
    build_N_matrix,
    az_el_to_khat,
    perp_projector,
    perp_orthonormal_basis,
)
from .extract import extract_complex_baseband
from .analyze import analyze, analyze_z_loops, AnalysisResult
from .bands import extract_bands, BandExtraction
from .multiband import (
    analyze_all_bands, MultiBandResult, ModeSeparation, make_multiband_figure,
)
from .magnetoionic import (
    appleton_hartree_polarization, mode_unit_vectors,
    mode_ellipticity_deg, mode_orientation_deg, mode_axial_ratio,
    exit_point_geometry, entry_point_geometry, modes_at_exit,
    integrated_faraday_rotation_rad,
    ExitGeometry, EntryGeometry, ModeAtExit,
)
from .view import make_view_figure
from .stokes import compute_stokes
from .beamform import beamform_grid
from .direction import (
    estimate_direction_from_z_lab,
    null_search,
    perp_residual_energy,
    parabolic_refine,
    lock_and_analyze,
)
from .io_hdf5 import read_capture, write_capture
from .config import default_loops_config

__version__ = "0.1.0"

__all__ = [
    "LOOP_NORMALS_DEFAULT",
    "build_N_matrix",
    "az_el_to_khat",
    "perp_projector",
    "perp_orthonormal_basis",
    "extract_complex_baseband",
    "analyze",
    "analyze_z_loops",
    "AnalysisResult",
    "extract_bands",
    "BandExtraction",
    "analyze_all_bands",
    "MultiBandResult",
    "ModeSeparation",
    "make_multiband_figure",
    "make_view_figure",
    "appleton_hartree_polarization",
    "mode_unit_vectors",
    "mode_ellipticity_deg",
    "mode_orientation_deg",
    "mode_axial_ratio",
    "exit_point_geometry",
    "entry_point_geometry",
    "modes_at_exit",
    "integrated_faraday_rotation_rad",
    "ExitGeometry",
    "EntryGeometry",
    "ModeAtExit",
    "compute_stokes",
    "beamform_grid",
    "estimate_direction_from_z_lab",
    "null_search",
    "perp_residual_energy",
    "parabolic_refine",
    "lock_and_analyze",
    "read_capture",
    "write_capture",
    "default_loops_config",
]
