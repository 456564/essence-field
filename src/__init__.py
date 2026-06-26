"""essence-field — 几何基元本质场视觉物质理解"""
from .operators import ColorFixedOperatorLayer, BASE_OPS, BAGUA_NAMES
from .pipeline import BaguaPipeline
from .field_core import (essence_field_compute, edge_aware_diffusion,
                          field_to_materials, describe_material)
from .diffusion import FieldDiffusion, bilateral_diffusion
from .primitive import discover_primitives, match_primitives
from .material import MaterialReader
