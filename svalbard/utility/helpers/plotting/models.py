"""Data model for describing graph settings"""

from enum import Enum

from pydantic import BaseModel


class CursorType(Enum):
    """Cursor types for individual cursors."""

    VERTICAL = "Vertical"
    HORIZONTAL = "Horizontal"
    BOTH = "Both"
    OFF = "Off"


class RangeType(Enum):
    """Range cursor types, including the single cursors."""

    OFF = "Off"
    VERTICAL = "Vertical"
    HORIZONTAL = "Horizontal"
    BOTH = "Both"
    RANGE_VERTICAL = "Range, vertical"
    RANGE_HORIZONTAL = "Range, horizontal"
    RANGE = "Range, both"


class Colormap(Enum):
    """Available colormaps, and their matplotlib names"""

    bone = "Bone"
    CMRmap = "CMRmap"
    gist_earth = "Earth"
    gray = "Gray"
    hot = "Hot"
    inferno = "Inferno"
    jet = "Jet"
    pink = "Pink"
    RdBu = "Red-White-Blue"
    RdGy = "Red-White-Gray"
    terrain = "Terrain"
    twilight = "Twilight"
    twilight_shifted = "Twilight shifted"
    viridis = "Viridis"


class CursorModel(BaseModel):
    """Model for defining graph cursor."""

    style: RangeType = RangeType.OFF
    x1: float = 0.0
    x2: float = 0.0
    y1: float = 0.0
    y2: float = 0.0


class BaseGraphModel(BaseModel):
    """Model for graph data."""

    # define axes settings and limits
    log_x: bool = False
    log_y: bool = False
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    # autoscale on/off
    autoscale_x: bool = True
    autoscale_y: bool = True
    # cursor settings
    cursors: CursorModel = CursorModel()


class ImageGraphModel(BaseGraphModel):
    """Model for image data."""

    # define extra settings for images
    transpose: bool = False
    # define colormap, use default if None
    colormap: Colormap = None
    invert_colormap: bool = False
    # define contrast settings
    autoscale_z: bool = True
    z_min: float = 0.0
    z_max: float = 1.0
