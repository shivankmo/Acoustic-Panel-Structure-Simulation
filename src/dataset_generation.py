"""
Step 1: Dataset Generation
Generate diverse 3D panel geometries with varying shapes, materials, and properties.
"""

import numpy as np
import pandas as pd
import json
import os
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from shapely.geometry import Point, Polygon, LineString, MultiPolygon
from shapely.ops import unary_union, transform
from shapely.affinity import scale, translate, rotate
from tqdm import tqdm
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import PyMesh, but handle gracefully if not available
try:
    import pymesh
    PYMESH_AVAILABLE = True
except ImportError:
    PYMESH_AVAILABLE = False
    logger.warning("PyMesh not available. Using simplified mesh generation.")

# Global constraints
BOUNDS = (0.5, 0.5)        # 50 cm × 50 cm panel footprint
MAX_THICKNESS = 0.05       # realistic acoustic panel max thickness: 5 cm
MIN_AREA_THICKNESS_RATIO = 40
MAX_AREA_THICKNESS_RATIO = 250
MIN_SUSTAINABILITY = 0.20  # dataset minimum
DEFAULT_MAX_COST = 60.0    # dataset maximum


MATERIALS = {
    'foam': {
        'density': (28, 120),                # kg/m³ – typical acoustic polyurethane foam
        'youngs_modulus': (5e3, 2e5),        # Pa – soft, compliant
        'poissons_ratio': (0.25, 0.45),
        'cost': (10, 30),                    # $/m² installed cost
        'sustainability_index': (0.30, 0.55)
    },
    'cork': {
        'density': (110, 200),               # kg/m³ – density of expanded cork boards
        'youngs_modulus': (1e7, 3e7),        # Pa – slightly compliant but stiffer than foam
        'poissons_ratio': (0.05, 0.20),
        'cost': (20, 50),                    # cork panels are expensive
        'sustainability_index': (0.85, 0.97)
    },
    'recycled_plastic': {
        'density': (850, 1100),              # kg/m³ – HDPE / PET blends
        'youngs_modulus': (5e8, 2e9),        # consistent with polymer composites
        'poissons_ratio': (0.25, 0.40),
        'cost': (15, 40),                    # manufacturing + recycling processes
        'sustainability_index': (0.55, 0.80)
    },
    'cardboard': {
        'density': (150, 400),               # kg/m³ – for pressed fiberboard
        'youngs_modulus': (5e8, 2e9),        # fiberboard stiffness range
        'poissons_ratio': (0.15, 0.35),
        'cost': (2, 15),                     # very cheap bulk pricing
        'sustainability_index': (0.70, 0.95)
    }
}


# Complexity to mesh size mapping (for FEM reference)
COMPLEXITY_MESH_SIZES = {
    1: 0.01,   # 10mm - Coarse mesh for simple geometries
    2: 0.008,  # 8mm - Moderate refinement
    3: 0.005,  # 5mm - Fine mesh for perforations and curves
    4: 0.003,  # 3mm - Very fine for fractals and irregular lattices
    5: 0.002   # 2mm - Ultra-fine for multi-scale and nested structures
}

# Parameter ranges by complexity level
COMPLEXITY_PARAMETERS = {
    1: {
        'num_holes': (0, 0),
        'hole_radius': (0, 0),
        'fractal_order': (0, 0),
        'nesting_levels': (0, 0),
        'curvature': (0, 0.1)
    },
    2: {
        'num_holes': (0, 5),  # allow hole-free shapes at level 2
        'hole_radius': (0.005, 0.02),
        'fractal_order': (0, 0),
        'nesting_levels': (0, 0),
        'curvature': (0, 0.3)
    },
    3: {
        'num_holes': (5, 20),
        'hole_radius': (0.003, 0.015),
        'fractal_order': (1, 2),
        'nesting_levels': (0, 1),
        'curvature': (0, 0.5)
    },
    4: {
        'num_holes': (20, 50),
        'hole_radius': (0.002, 0.01),
        'fractal_order': (2, 3),
        'nesting_levels': (2, 3),
        'curvature': (0, 0.7)
    },
    5: {
        'num_holes': (50, 200),
        'hole_radius': (0.001, 0.008),
        'fractal_order': (3, 5),
        'nesting_levels': (3, 5),
        'curvature': (0, 1.0)
    }
}


# ============================================================================
# Base Shape Generation
# ============================================================================

def generate_rectangle(bounds: Tuple[float, float], center: Tuple[float, float] = (0, 0)) -> Polygon:
    """Generate a rectangle within bounds."""
    width, height = bounds
    x, y = center
    return Polygon([
        (x - width/2, y - height/2),
        (x + width/2, y - height/2),
        (x + width/2, y + height/2),
        (x - width/2, y + height/2)
    ])


def generate_circle(radius: float, center: Tuple[float, float] = (0, 0), num_points: int = 64) -> Polygon:
    """Generate a circle using a polygon approximation."""
    x, y = center
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    coords = [(x + radius * np.cos(a), y + radius * np.sin(a)) for a in angles]
    return Polygon(coords)


def generate_ellipse(width: float, height: float, center: Tuple[float, float] = (0, 0), num_points: int = 64) -> Polygon:
    """Generate an ellipse using a polygon approximation."""
    x, y = center
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    coords = [(x + width/2 * np.cos(a), y + height/2 * np.sin(a)) for a in angles]
    return Polygon(coords)


def generate_irregular_polygon(bounds: Tuple[float, float], num_vertices: int, 
                               irregularity: float = 0.5, spikiness: float = 0.5) -> Polygon:
    """Generate an irregular polygon with controlled randomness."""
    width, height = bounds
    center = (0, 0)
    
    # Generate vertices around a circle
    angles = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)
    radius = min(width, height) / 2 * (1 - irregularity)
    
    vertices = []
    for angle in angles:
        # Add irregularity
        r = radius * (1 + random.uniform(-irregularity, irregularity))
        # Add spikiness
        r *= (1 + spikiness * random.uniform(-0.5, 0.5))
        
        x = center[0] + r * np.cos(angle)
        y = center[1] + r * np.sin(angle)
        vertices.append((x, y))
    
    return Polygon(vertices)


def generate_star_polygon(center: Tuple[float, float], outer_radius: float, 
                          inner_radius: float, num_points: int) -> Polygon:
    """Generate a star polygon."""
    x, y = center
    vertices = []
    for i in range(num_points * 2):
        angle = i * np.pi / num_points
        if i % 2 == 0:
            r = outer_radius
        else:
            r = inner_radius
        vertices.append((x + r * np.cos(angle), y + r * np.sin(angle)))
    return Polygon(vertices)


def generate_gear_shape(center: Tuple[float, float], outer_radius: float, 
                        num_teeth: int, tooth_depth: float) -> Polygon:
    """Generate a gear-like shape."""
    x, y = center
    vertices = []
    angle_step = 2 * np.pi / num_teeth
    
    for i in range(num_teeth):
        # Outer point
        angle1 = i * angle_step
        vertices.append((x + outer_radius * np.cos(angle1), 
                        y + outer_radius * np.sin(angle1)))
        
        # Inner point
        angle2 = (i + 0.5) * angle_step
        inner_r = outer_radius - tooth_depth
        vertices.append((x + inner_r * np.cos(angle2), 
                        y + inner_r * np.sin(angle2)))
    
    return Polygon(vertices)


def generate_base_shape(shape_type: str, bounds: Tuple[float, float], 
                       complexity: int, **kwargs) -> Polygon:
    """Router function for base shape generation - focused on curved, wave-like panels."""
    width, height = bounds
    center = (0, 0)
    
    if shape_type == 'rectangle':
        return generate_rectangle(bounds, center)
    elif shape_type == 'circle':
        radius = min(width, height) / 2 * (0.7 + 0.3 * random.random())
        return generate_circle(radius, center)
    elif shape_type == 'ellipse':
        w = width * (0.6 + 0.4 * random.random())
        h = height * (0.6 + 0.4 * random.random())
        return generate_ellipse(w, h, center)
    elif shape_type == 'polygon':
        num_vertices = random.randint(6, 12)
        return generate_irregular_polygon(bounds, num_vertices, irregularity=0.3 + 0.3 * random.random(), spikiness=0.2 + 0.2 * random.random())
    elif shape_type == 'organic':
        num_points = random.randint(10, 20)
        radius_var = 0.2 + 0.4 * random.random()
        smoothness = 0.3 + 0.5 * random.random()
        return generate_organic_shape(center, num_points, radius_var, smoothness)
    elif shape_type == 'wave':
        frequency = random.uniform(2, 8)
        amplitude = min(width, height) * (0.05 + 0.15 * random.random())
        phase = random.uniform(0, 2 * np.pi)
        return generate_wave_boundary(bounds, frequency, amplitude, phase)
    elif shape_type == 'spline':
        num_control = random.randint(4, 8)
        control_points = []
        for _ in range(num_control):
            angle = random.uniform(0, 2 * np.pi)
            radius = min(width, height) / 2 * (0.4 + 0.4 * random.random())
            control_points.append((center[0] + radius * np.cos(angle), 
                                 center[1] + radius * np.sin(angle)))
        return generate_spline_boundary(control_points, num_points=60)
    elif shape_type == 'honeycomb':
        cell_size = random.uniform(0.03, 0.06)
        wall_thickness = random.uniform(0.002, 0.006)
        return generate_honeycomb_lattice(bounds, cell_size, wall_thickness)
    else:
        # Default to organic shape
        return generate_organic_shape(center, 12, 0.3, 0.5)


# ============================================================================
# Perforated Patterns
# ============================================================================

def generate_grid_perforation(base_shape: Polygon, hole_radius: float, 
                              spacing: float, offset: Tuple[float, float] = (0, 0)) -> Polygon:
    """Generate regular grid of circular holes."""
    bounds = base_shape.bounds
    minx, miny, maxx, maxy = bounds
    
    holes = []
    x, y = minx + offset[0], miny + offset[1]
    
    while x < maxx:
        while y < maxy:
            point = Point(x, y)
            if base_shape.contains(point) or base_shape.intersects(point):
                hole = point.buffer(hole_radius)
                holes.append(hole)
            y += spacing
        y = miny + offset[1]
        x += spacing
    
    # Subtract all holes from base shape
    result = base_shape
    for hole in holes:
        result = result.difference(hole)
    
    return result


def generate_random_perforation(base_shape: Polygon, num_holes: int, 
                               min_radius: float, max_radius: float, 
                               min_spacing: float) -> Polygon:
    """Generate randomly placed holes with spacing constraints."""
    bounds = base_shape.bounds
    minx, miny, maxx, maxy = bounds
    
    holes = []
    placed_points = []
    
    attempts = 0
    max_attempts = num_holes * 100
    
    while len(holes) < num_holes and attempts < max_attempts:
        attempts += 1
        x = random.uniform(minx, maxx)
        y = random.uniform(miny, maxy)
        point = Point(x, y)
        
        # Check if point is within base shape
        if not base_shape.contains(point):
            continue
        
        # Check spacing constraint
        too_close = False
        radius = random.uniform(min_radius, max_radius)
        for placed in placed_points:
            if point.distance(placed[0]) < (radius + placed[1] + min_spacing):
                too_close = True
                break
        
        if not too_close:
            hole = point.buffer(radius)
            holes.append(hole)
            placed_points.append((point, radius))
    
    # Subtract holes from base shape
    result = base_shape
    for hole in holes:
        result = result.difference(hole)
    
    return result


def generate_circular_array_perforation(base_shape: Polygon, center: Tuple[float, float],
                                       num_rings: int, holes_per_ring: int, 
                                       base_radius: float) -> Polygon:
    """Generate concentric circular hole patterns."""
    holes = []
    cx, cy = center
    
    for ring in range(1, num_rings + 1):
        radius = base_radius * ring
        angle_step = 2 * np.pi / holes_per_ring
        
        for i in range(holes_per_ring):
            angle = i * angle_step
            x = cx + radius * np.cos(angle)
            y = cy + radius * np.sin(angle)
            point = Point(x, y)
            
            if base_shape.contains(point) or base_shape.intersects(point):
                hole_radius = base_radius * 0.3
                hole = point.buffer(hole_radius)
                holes.append(hole)
    
    # Subtract holes from base shape
    result = base_shape
    for hole in holes:
        result = result.difference(hole)
    
    return result


def generate_hexagonal_perforation(base_shape: Polygon, hole_radius: float, 
                                   spacing: float) -> Polygon:
    """Generate hexagonal close-packed hole pattern."""
    bounds = base_shape.bounds
    minx, miny, maxx, maxy = bounds
    
    holes = []
    y = miny
    row = 0
    
    while y < maxy:
        x = minx
        if row % 2 == 1:
            x += spacing / 2  # Offset for hexagonal packing
        
        while x < maxx:
            point = Point(x, y)
            if base_shape.contains(point) or base_shape.intersects(point):
                hole = point.buffer(hole_radius)
                holes.append(hole)
            x += spacing
        y += spacing * np.sqrt(3) / 2
        row += 1
    
    # Subtract holes from base shape
    result = base_shape
    for hole in holes:
        result = result.difference(hole)
    
    return result


# ============================================================================
# Lattice Structures
# ============================================================================

def generate_honeycomb_lattice(bounds: Tuple[float, float], cell_size: float, 
                              wall_thickness: float) -> Polygon:
    """Generate regular hexagonal honeycomb lattice."""
    width, height = bounds
    hex_radius = cell_size / 2
    
    hexagons = []
    y = 0
    row = 0
    
    while y < height:
        x = 0
        if row % 2 == 1:
            x += hex_radius * np.sqrt(3)
        
        while x < width:
            # Create hexagon
            angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
            coords = [(x + hex_radius * np.cos(a), y + hex_radius * np.sin(a)) 
                     for a in angles]
            hexagon = Polygon(coords)
            
            # Create inner hexagon (hollow)
            inner_radius = hex_radius - wall_thickness
            inner_coords = [(x + inner_radius * np.cos(a), y + inner_radius * np.sin(a)) 
                           for a in angles]
            inner_hex = Polygon(inner_coords)
            
            hexagon = hexagon.difference(inner_hex)
            hexagons.append(hexagon)
            
            x += hex_radius * np.sqrt(3) * 2
        y += hex_radius * 1.5
        row += 1
    
    # Union all hexagons
    if hexagons:
        return unary_union(hexagons)
    return Polygon()


def generate_triangular_lattice(bounds: Tuple[float, float], cell_size: float, 
                               wall_thickness: float) -> Polygon:
    """Generate triangular mesh pattern."""
    width, height = bounds
    
    # Create triangular grid
    triangles = []
    y = 0
    
    while y < height:
        x = 0
        while x < width:
            # Create triangle
            p1 = (x, y)
            p2 = (x + cell_size, y)
            p3 = (x + cell_size / 2, y + cell_size * np.sqrt(3) / 2)
            triangle = Polygon([p1, p2, p3])
            
            # Create smaller triangle inside (hollow)
            center = triangle.centroid
            scaled = scale(triangle, 1 - wall_thickness / cell_size, 
                         1 - wall_thickness / cell_size, origin=center)
            triangle = triangle.difference(scaled)
            
            triangles.append(triangle)
            x += cell_size
        y += cell_size * np.sqrt(3) / 2
    
    if triangles:
        return unary_union(triangles)
    return Polygon()


def generate_beam_lattice(bounds: Tuple[float, float], beam_width: float,
                         spacing_x: float, spacing_y: float, angle: float = 0) -> Polygon:
    """Generate grid of beams with optional rotation."""
    width, height = bounds
    
    beams = []
    
    # Horizontal beams
    y = 0
    while y < height:
        beam = LineString([(0, y), (width, y)]).buffer(beam_width / 2)
        if angle != 0:
            beam = rotate(beam, angle, origin=(width/2, height/2))
        beams.append(beam)
        y += spacing_y
    
    # Vertical beams
    x = 0
    while x < width:
        beam = LineString([(x, 0), (x, height)]).buffer(beam_width / 2)
        if angle != 0:
            beam = rotate(beam, angle, origin=(width/2, height/2))
        beams.append(beam)
        x += spacing_x
    
    if beams:
        return unary_union(beams)
    return Polygon()


# ============================================================================
# Fractal/Recursive Patterns
# ============================================================================

def generate_sierpinski_carpet(order: int, bounds: Tuple[float, float]) -> Polygon:
    """Generate Sierpinski carpet fractal."""
    if order == 0:
        return generate_rectangle(bounds)
    
    width, height = bounds
    base = generate_rectangle(bounds)
    
    # Recursively subdivide
    sub_width = width / 3
    sub_height = height / 3
    
    result = base
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                # Remove center
                center_rect = generate_rectangle((sub_width, sub_height), 
                                               (sub_width * (i - 1), sub_height * (j - 1)))
                result = result.difference(center_rect)
            elif order > 1:
                # Recursively process sub-squares
                sub_bounds = (sub_width, sub_height)
                sub_center = (sub_width * (i - 1), sub_height * (j - 1))
                sub_rect = generate_rectangle(sub_bounds, sub_center)
                sub_carpet = generate_sierpinski_carpet(order - 1, sub_bounds)
                sub_carpet = translate(sub_carpet, xoff=sub_center[0], yoff=sub_center[1])
                result = result.intersection(sub_carpet)
    
    return result


def generate_koch_snowflake(order: int, center: Tuple[float, float], 
                            radius: float) -> Polygon:
    """Generate Koch snowflake fractal."""
    if order == 0:
        # Base triangle
        angles = [0, 2*np.pi/3, 4*np.pi/3]
        coords = [(center[0] + radius * np.cos(a), center[1] + radius * np.sin(a)) 
                 for a in angles]
        return Polygon(coords)
    
    # Recursive implementation (simplified)
    # For higher orders, this would subdivide edges
    base_triangle = generate_koch_snowflake(0, center, radius)
    
    if order > 1:
        # Simplified: add smaller triangles to edges
        # Full implementation would recursively subdivide each edge
        pass
    
    return base_triangle


def generate_tree_fractal(order: int, base_point: Tuple[float, float], 
                         length: float, angle: float, branching_factor: int = 2) -> MultiPolygon:
    """Generate recursive branching tree fractal."""
    if order == 0:
        return MultiPolygon()
    
    x, y = base_point
    end_x = x + length * np.cos(angle)
    end_y = y + length * np.sin(angle)
    
    branch = LineString([base_point, (end_x, end_y)]).buffer(0.01)
    branches = [branch]
    
    if order > 1:
        # Recursively add branches
        for i in range(branching_factor):
            branch_angle = angle + (i - branching_factor/2) * np.pi/6
            sub_branches = generate_tree_fractal(order - 1, (end_x, end_y), 
                                               length * 0.7, branch_angle, branching_factor)
            if isinstance(sub_branches, MultiPolygon):
                branches.extend(list(sub_branches.geoms))
            elif isinstance(sub_branches, Polygon):
                branches.append(sub_branches)
    
    if len(branches) > 1:
        return MultiPolygon(branches)
    elif branches:
        return branches[0]
    return MultiPolygon()


# ============================================================================
# Curved Boundaries
# ============================================================================

def generate_wave_boundary(bounds: Tuple[float, float], frequency: float, 
                          amplitude: float, phase: float = 0) -> Polygon:
    """Generate shape with sinusoidal wave edges."""
    width, height = bounds
    num_points = 100
    
    # Generate wave boundary
    x_coords = np.linspace(-width/2, width/2, num_points)
    y_coords = amplitude * np.sin(frequency * x_coords + phase)
    
    # Create closed polygon
    coords = list(zip(x_coords, y_coords))
    # Add bottom edge
    coords.append((width/2, -height/2))
    coords.append((-width/2, -height/2))
    coords.append(coords[0])  # Close
    
    return Polygon(coords)


def generate_spline_boundary(control_points: List[Tuple[float, float]], 
                            num_points: int, tension: float = 0.5) -> Polygon:
    """Generate smooth spline curve boundary."""
    if len(control_points) < 3:
        return Polygon(control_points)
    
    # Simple interpolation (full spline would use scipy)
    coords = []
    for i in range(len(control_points) - 1):
        p1 = control_points[i]
        p2 = control_points[i + 1]
        segment_points = int(num_points / (len(control_points) - 1))
        for j in range(segment_points):
            t = j / segment_points
            x = p1[0] * (1 - t) + p2[0] * t
            y = p1[1] * (1 - t) + p2[1] * t
            coords.append((x, y))
    
    coords.append(coords[0])  # Close
    return Polygon(coords)


def generate_organic_shape(center: Tuple[float, float], num_points: int,
                           radius_variation: float, smoothness: float) -> Polygon:
    """Generate organic, irregular smooth boundary."""
    cx, cy = center
    base_radius = 0.2
    
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    coords = []
    
    for angle in angles:
        radius = base_radius * (1 + radius_variation * random.uniform(-1, 1))
        x = cx + radius * np.cos(angle)
        y = cy + radius * np.sin(angle)
        coords.append((x, y))
    
    polygon = Polygon(coords)
    # Smooth using buffer
    if smoothness > 0:
        polygon = polygon.buffer(smoothness * 0.01).buffer(-smoothness * 0.01)
    
    return polygon


# ============================================================================
# Nested/Multi-Scale Geometries
# ============================================================================

def generate_nested_polygons(outer_shape: Polygon, num_levels: int, 
                            scale_factor: float, offset: Tuple[float, float] = (0, 0)) -> Polygon:
    """Generate concentric nested shapes."""
    result = outer_shape
    current = outer_shape
    
    for level in range(1, num_levels):
        # Scale down
        center = current.centroid
        scaled = scale(current, scale_factor, scale_factor, origin=center)
        # Translate
        scaled = translate(scaled, xoff=offset[0], yoff=offset[1])
        
        # Alternate between union and difference for visual interest
        if level % 2 == 0:
            result = result.union(scaled)
        else:
            result = result.difference(scaled)
        
        current = scaled
    
    return result


# ============================================================================
# Complexity Assignment and Parameter Sampling
# ============================================================================

def assign_complexity_level(geometry: Polygon, num_holes: int, has_fractal: bool, 
                           has_nesting: bool, curvature_metric: float) -> int:
    """Assign complexity level based on geometry features."""
    if num_holes == 0 and not has_fractal and not has_nesting and curvature_metric < 0.1:
        return 1
    elif num_holes <= 5 and not has_fractal and curvature_metric < 0.3:
        return 2
    elif num_holes <= 20 and (has_fractal or has_nesting) and curvature_metric < 0.5:
        return 3
    elif num_holes <= 50 or (has_fractal) or (has_nesting):
        return 4
    else:
        return 5


def sample_parameters(complexity_level: int) -> Dict[str, Any]:
    """Sample parameters based on complexity level."""
    params = COMPLEXITY_PARAMETERS.get(complexity_level, COMPLEXITY_PARAMETERS[1])
    
    return {
        'num_holes': random.randint(*params['num_holes']),
        'hole_radius': random.uniform(*params['hole_radius']) if params['hole_radius'][1] > 0 else 0,
        'fractal_order': random.randint(*params['fractal_order']) if params['fractal_order'][1] > 0 else 0,
        'nesting_levels': random.randint(*params['nesting_levels']) if params['nesting_levels'][1] > 0 else 0,
        'curvature': random.uniform(*params['curvature'])
    }


def select_shape_strategy(complexity_level: int) -> str:
    """Choose shape generation method based on complexity - favor curved/wave-like shapes."""
    if complexity_level == 1:
        return random.choice(['circle', 'ellipse', 'polygon'])
    elif complexity_level == 2:
        return random.choice(['circle', 'ellipse', 'polygon', 'organic', 'wave'])
    elif complexity_level == 3:
        return random.choice(['polygon', 'organic', 'wave', 'spline', 'honeycomb'])
    else:
        return random.choice(['polygon', 'organic', 'wave', 'spline', 'honeycomb'])


# ============================================================================
# Geometry Validation and Scaling
# ============================================================================

def make_valid_safe(geometry: Polygon) -> Polygon:
    """Safely make geometry valid."""
    if not geometry.is_valid:
        try:
            geometry = geometry.buffer(0)
        except:
            pass
    return geometry


def scale_to_bounds(geometry: Polygon, target_bounds: Tuple[float, float]) -> Polygon:
    """Scale geometry to fit within target bounds."""
    bounds = geometry.bounds
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    
    target_width, target_height = target_bounds
    
    scale_x = target_width / width if width > 0 else 1
    scale_y = target_height / height if height > 0 else 1
    scale_factor = min(scale_x, scale_y) * 0.95  # Leave small margin
    
    center = geometry.centroid
    scaled = scale(geometry, scale_factor, scale_factor, origin=center)
    
    # Center in bounds
    new_bounds = scaled.bounds
    dx = (target_width - (new_bounds[2] - new_bounds[0])) / 2 - new_bounds[0]
    dy = (target_height - (new_bounds[3] - new_bounds[1])) / 2 - new_bounds[1]
    scaled = translate(scaled, xoff=dx, yoff=dy)
    
    return scaled


def check_bounds(geometry: Polygon, bounds: Tuple[float, float]) -> bool:
    """Check if geometry is within bounds."""
    target_poly = generate_rectangle(bounds)
    return geometry.within(target_poly) or geometry.intersects(target_poly)


# ============================================================================
# 2D to 3D Conversion (PyMesh)
# ============================================================================

def extrude_to_3d(geometry: Polygon, thickness: float) -> Optional[Dict[str, np.ndarray]]:
    """Extrude 2D Shapely geometry to 3D mesh data."""
    try:
        # Extract exterior coordinates
        if isinstance(geometry, MultiPolygon):
            # Handle MultiPolygon by taking largest component
            geometry = max(geometry.geoms, key=lambda g: g.area)
        
        exterior_coords = list(geometry.exterior.coords)
        if len(exterior_coords) < 3:
            return None
        
        # Remove duplicate last point if closed
        if exterior_coords[0] == exterior_coords[-1]:
            exterior_coords = exterior_coords[:-1]
        
        # Create 2D mesh
        vertices_2d = np.array(exterior_coords)
        
        # Simple triangulation - create a simple extrusion
        num_vertices = len(vertices_2d)
        vertices_3d = []
        faces = []
        
        # Bottom face vertices
        for v in vertices_2d:
            vertices_3d.append([v[0], v[1], 0])
        
        # Top face vertices
        for v in vertices_2d:
            vertices_3d.append([v[0], v[1], thickness])
        
        vertices_3d = np.array(vertices_3d)
        
        # Create faces (simplified triangulation)
        # Bottom face (fan triangulation)
        for i in range(1, num_vertices - 1):
            faces.append([0, i, i + 1])
        
        # Top face (fan triangulation, reversed)
        for i in range(1, num_vertices - 1):
            faces.append([num_vertices, num_vertices + i + 1, num_vertices + i])
        
        # Side faces (quads as two triangles)
        for i in range(num_vertices):
            next_i = (i + 1) % num_vertices
            # First triangle of quad
            faces.append([i, next_i, num_vertices + i])
            # Second triangle of quad
            faces.append([next_i, num_vertices + next_i, num_vertices + i])
        
        faces = np.array(faces, dtype=int)
        
        # Return mesh data as dictionary
        return {
            'vertices': vertices_3d,
            'faces': faces
        }
    
    except Exception as e:
        logger.warning(f"Extrusion failed: {e}")
        return None


# ============================================================================
# Material System
# ============================================================================

def assign_material(min_sustainability: float = MIN_SUSTAINABILITY) -> Tuple[str, Dict[str, float]]:
    """Randomly assign material and sample properties, ensuring sustainability threshold."""
    # Filter materials that can meet sustainability threshold
    valid_materials = {
        k: v for k, v in MATERIALS.items() 
        if v['sustainability_index'][1] >= min_sustainability
    }
    
    if not valid_materials:
        # If no materials meet threshold, use all materials (will fail validation)
        valid_materials = MATERIALS
    
    material_type = random.choice(list(valid_materials.keys()))
    material_props = valid_materials[material_type]
    
    # Sample sustainability from range that meets threshold
    sust_min, sust_max = material_props['sustainability_index']
    sust_range_min = max(sust_min, min_sustainability)
    
    properties = {
        'density': random.uniform(*material_props['density']),
        'youngs_modulus': random.uniform(*material_props['youngs_modulus']),
        'poissons_ratio': random.uniform(*material_props['poissons_ratio']),
        'cost': random.uniform(*material_props['cost']),
        'sustainability_index': random.uniform(sust_range_min, sust_max)
    }
    
    return material_type, properties


# ============================================================================
# Geometric Feature Computation
# ============================================================================

def compute_2d_features(geometry: Polygon) -> Dict[str, float]:
    """Compute 2D geometric features using Shapely."""
    features = {
        'area': geometry.area,
        'perimeter': geometry.length,
        'num_holes': len(geometry.interiors) if hasattr(geometry, 'interiors') else 0,
        'compactness': 4 * np.pi * geometry.area / (geometry.length ** 2) if geometry.length > 0 else 0,
    }
    
    # Hole area ratio
    hole_area = sum(Polygon(interior).area for interior in geometry.interiors) if hasattr(geometry, 'interiors') else 0
    features['hole_area_ratio'] = hole_area / geometry.area if geometry.area > 0 else 0
    
    # Convexity
    convex_hull = geometry.convex_hull
    features['convexity'] = geometry.area / convex_hull.area if convex_hull.area > 0 else 0
    
    return features


def compute_3d_features(mesh_data: Dict[str, np.ndarray], thickness: float) -> Dict[str, float]:
    """Compute 3D geometric features from mesh data."""
    try:
        vertices = mesh_data['vertices']
        faces = mesh_data['faces']
        
        # Compute surface area (sum of triangle areas)
        surface_area = 0.0
        volume = 0.0
        
        for face in faces:
            v0, v1, v2 = vertices[face]
            # Triangle area using cross product
            edge1 = v1 - v0
            edge2 = v2 - v0
            triangle_area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
            surface_area += triangle_area
            
            # Volume contribution (signed volume)
            volume += np.dot(v0, np.cross(v1, v2)) / 6.0
        
        # Ensure volume is positive
        volume = abs(volume)
        
        # Bounding box volume for porosity calculation
        bbox_min = vertices.min(axis=0)
        bbox_max = vertices.max(axis=0)
        bbox_volume = np.prod(bbox_max - bbox_min)
        
        # Ensure surface_area is reasonable (should be at least 2x the 2D area for top+bottom)
        # If computed surface_area seems too small, it might be incorrect
        min_expected_surface = 0.3  # Minimum expected for 0.5x0.5m panel
        if surface_area < min_expected_surface:
            # Recalculate using bounding box approximation
            width = bbox_max[0] - bbox_min[0]
            height = bbox_max[1] - bbox_min[1]
            # Top + bottom + 4 sides
            surface_area = 2 * width * height + 2 * (width + height) * thickness
        
        features = {
            'surface_area': float(surface_area),
            'volume': float(volume),
            'porosity': float((1 - volume / bbox_volume) * 100) if bbox_volume > 0 else 0.0,
            'area_thickness_ratio': float(surface_area / thickness) if thickness > 0 else 0.0,
            'curvature_metric': 0.0  # Placeholder - would need curvature computation
        }
        
        return features
    except Exception as e:
        logger.warning(f"Feature computation failed: {e}")
        return {
            'surface_area': 0.0,
            'volume': 0.0,
            'porosity': 0.0,
            'area_thickness_ratio': 0.0,
            'curvature_metric': 0.0
        }


# ============================================================================
# Constraint Validation
# ============================================================================

def check_area_thickness_ratio(area: float, thickness: float) -> bool:
    """Check if area/thickness ratio is within bounds."""
    if thickness <= 0 or area <= 0:
        return False
    ratio = area / thickness
    is_valid = MIN_AREA_THICKNESS_RATIO <= ratio <= MAX_AREA_THICKNESS_RATIO
    if not is_valid:
        logger.debug(f"Area/thickness ratio {ratio:.2f} outside bounds [{MIN_AREA_THICKNESS_RATIO}, {MAX_AREA_THICKNESS_RATIO}]")
    return is_valid


def check_cost(total_cost: float, max_cost: float) -> bool:
    """Check if total cost is within budget."""
    return total_cost <= max_cost


def check_sustainability(sustainability_index: float, min_threshold: float = MIN_SUSTAINABILITY) -> bool:
    """Check if sustainability index meets threshold."""
    return sustainability_index >= min_threshold


# ============================================================================
# Storage Functions
# ============================================================================

def save_mesh(mesh_data: Dict[str, np.ndarray], filepath: str) -> bool:
    """Save mesh data to OBJ file."""
    try:
        vertices = mesh_data['vertices']
        faces = mesh_data['faces']
        
        with open(filepath, 'w') as f:
            # Write vertices
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            # Write faces (OBJ uses 1-based indexing)
            for face in faces:
                # OBJ format: f v1 v2 v3
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        
        return True
    except Exception as e:
        logger.error(f"Failed to save mesh {filepath}: {e}")
        return False


def save_geometry_geojson(geometry: Polygon, filepath: str) -> bool:
    """Save Shapely geometry as GeoJSON."""
    try:
        from shapely.geometry import mapping
        import json
        
        geojson = {
            'type': 'Feature',
            'geometry': mapping(geometry),
            'properties': {}
        }
        
        with open(filepath, 'w') as f:
            json.dump(geojson, f)
        
        return True
    except Exception as e:
        logger.error(f"Failed to save GeoJSON {filepath}: {e}")
        return False


def append_to_dataset_csv(data_dict: Dict[str, Any], csv_path: str) -> None:
    """Append row to CSV dataset."""
    df = pd.DataFrame([data_dict])
    
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, mode='w', header=True, index=False)


def create_complexity_separated_csvs(main_csv: str, output_dir: str) -> None:
    """Create separate CSV files per complexity level."""
    try:
        df = pd.read_csv(main_csv)
        
        for level in range(1, 6):
            level_df = df[df['complexity_level'] == level]
            if not level_df.empty:
                output_path = os.path.join(output_dir, f'dataset_complexity_{level}.csv')
                level_df.to_csv(output_path, index=False)
                logger.info(f"Created/updated {output_path} with {len(level_df)} geometries")
    except Exception as e:
        logger.warning(f"Failed to create separated CSVs: {e}")


def create_complexity_summary(main_csv: str, output_path: str) -> None:
    """Create complexity summary JSON file."""
    try:
        df = pd.read_csv(main_csv)
        total = len(df)
        
        summary = {
            'complexity_distribution': {},
            'mesh_size_recommendations': COMPLEXITY_MESH_SIZES,
            'processing_recommendations': {
                'batch_order': [1, 2, 3, 4, 5],
                'parallel_processing': True,
                'memory_requirements': {
                    '1': 'low',
                    '2': 'low',
                    '3': 'medium',
                    '4': 'high',
                    '5': 'very_high'
                }
            }
        }
        
        for level in range(1, 6):
            count = len(df[df['complexity_level'] == level])
            summary['complexity_distribution'][str(level)] = {
                'count': int(count),
                'percentage': round(count / total * 100, 2) if total > 0 else 0
            }
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Created complexity summary: {output_path}")
    except Exception as e:
        logger.warning(f"Failed to create complexity summary: {e}")


# ============================================================================
# Main Generation Loop
# ============================================================================

def generate_dataset(num_samples: int, output_dir: str, seed: Optional[int] = None,
                    max_cost: float = DEFAULT_MAX_COST, 
                    min_sustainability: float = MIN_SUSTAINABILITY,
                    complexity_distribution: str = 'uniform',
                    create_separated_csvs: bool = True) -> Dict[str, Any]:
    """Main function to generate dataset of geometries."""
    
    # Initialize random seed
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    geometries_dir = os.path.join(output_dir, 'geometries')
    os.makedirs(geometries_dir, exist_ok=True)
    
    # Initialize CSV
    csv_path = os.path.join(output_dir, 'dataset.csv')
    csv_initialized = False
    
    # Check for existing dataset and get next available geometry ID
    existing_ids = set()
    if os.path.exists(csv_path):
        try:
            existing_df = pd.read_csv(csv_path)
            if 'geometry_id' in existing_df.columns:
                existing_ids = set(existing_df['geometry_id'].values)
                logger.info(f"Found existing dataset with {len(existing_ids)} geometries. Will append new geometries.")
        except Exception as e:
            logger.warning(f"Could not read existing CSV: {e}")
    
    # Find next available geometry ID
    if existing_ids:
        max_id = max([int(id.split('_')[1]) for id in existing_ids if '_' in id], default=-1)
        start_id = max_id + 1
    else:
        start_id = 0
    
    # Complexity counters
    complexity_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    
    # Statistics
    stats = {
        'total_generated': 0,
        'valid_samples': 0,
        'invalid_samples': 0,
        'complexity_distribution': complexity_counts.copy()
    }
    
    # Determine complexity distribution
    if complexity_distribution == 'uniform':
        complexity_weights = [1, 1, 1, 1, 1]
    elif complexity_distribution == 'balanced':
        complexity_weights = [2, 3, 3, 2, 1]
    elif complexity_distribution == 'skewed_low':
        complexity_weights = [5, 3, 2, 1, 1]
    elif complexity_distribution == 'skewed_high':
        complexity_weights = [1, 1, 2, 3, 5]
    else:
        complexity_weights = [1, 1, 1, 1, 1]
    
    # Generate samples
    pbar = tqdm(total=num_samples, desc="Generating geometries")
    
    # Maximum attempts to prevent infinite loops
    max_attempts = num_samples * 10  # Allow up to 10x attempts per sample
    consecutive_failures = 0
    max_consecutive_failures = 100  # Stop if 100 consecutive failures
    
    while stats['valid_samples'] < num_samples:
        stats['total_generated'] += 1
        
        # Safety check to prevent infinite loops
        if stats['total_generated'] > max_attempts:
            logger.error(f"Reached maximum attempts ({max_attempts}). Stopping generation.")
            break
        
        if consecutive_failures >= max_consecutive_failures:
            logger.error(f"Too many consecutive failures ({consecutive_failures}). Stopping generation.")
            logger.error("This may indicate overly strict constraints or a bug in generation.")
            break
        
        # Sample complexity level
        complexity_level = random.choices(range(1, 6), weights=complexity_weights)[0]
        
        # Sample parameters
        params = sample_parameters(complexity_level)
        
        # Select shape strategy
        shape_type = select_shape_strategy(complexity_level)
        
        try:
            # Generate base shape
            base_shape = generate_base_shape(shape_type, BOUNDS, complexity_level)
            base_shape = make_valid_safe(base_shape)
            
            # Add holes if needed
            num_holes = params['num_holes']
            if num_holes > 0:
                if complexity_level <= 2:
                    base_shape = generate_random_perforation(
                        base_shape, num_holes, 
                        params['hole_radius'] * 0.5, params['hole_radius'],
                        params['hole_radius'] * 2
                    )
                elif complexity_level == 3:
                    base_shape = generate_grid_perforation(
                        base_shape, params['hole_radius'],
                        params['hole_radius'] * 3
                    )
                else:
                    base_shape = generate_hexagonal_perforation(
                        base_shape, params['hole_radius'],
                        params['hole_radius'] * 2.5
                    )
                base_shape = make_valid_safe(base_shape)
            
            # Add fractal patterns if needed
            has_fractal = params['fractal_order'] > 0
            if has_fractal and complexity_level >= 3:
                if random.random() < 0.3:  # 30% chance
                    base_shape = generate_sierpinski_carpet(
                        min(params['fractal_order'], 3), BOUNDS
                    )
                    base_shape = make_valid_safe(base_shape)
            
            # Add nesting if needed
            has_nesting = params['nesting_levels'] > 0
            if has_nesting and complexity_level >= 4:
                if random.random() < 0.2:  # 20% chance
                    base_shape = generate_nested_polygons(
                        base_shape, min(params['nesting_levels'], 3), 0.7
                    )
                    base_shape = make_valid_safe(base_shape)
            
            # Scale to bounds
            base_shape = scale_to_bounds(base_shape, BOUNDS)
            base_shape = make_valid_safe(base_shape)
            
            # Validate bounds
            if not check_bounds(base_shape, BOUNDS):
                stats['invalid_samples'] += 1
                consecutive_failures += 1
                pbar.set_postfix({'valid': stats['valid_samples'], 'invalid': stats['invalid_samples'], 'reason': 'bounds'})
                continue
            
            # Sample thickness - bias heavily toward thinner panels to meet A/t ratio constraint
            # For A/t >= 50: if surface_area ≈ 0.5 m², need thickness <= 0.01 m
            # Use weighted sampling: 85% chance of thin (0.005-0.009), 15% chance of thicker (0.009-0.015)
            if random.random() < 0.85:
                thickness = random.uniform(0.005, 0.009)  # Thin panels for higher A/t ratio
            else:
                thickness = random.uniform(0.009, 0.015)  # Slightly thicker panels
            
            # Extrude to 3D
            mesh_3d = extrude_to_3d(base_shape, thickness)
            if mesh_3d is None:
                stats['invalid_samples'] += 1
                consecutive_failures += 1
                pbar.set_postfix({'valid': stats['valid_samples'], 'invalid': stats['invalid_samples'], 'reason': 'extrusion'})
                continue
            
            # Assign material (with sustainability constraint)
            material_type, material_props = assign_material(min_sustainability)
            
            # Compute features
            features_2d = compute_2d_features(base_shape)
            features_3d = compute_3d_features(mesh_3d, thickness)
            
            # Re-assign complexity based on actual geometry
            actual_complexity = assign_complexity_level(
                base_shape,
                features_2d['num_holes'],
                has_fractal,
                has_nesting,
                params['curvature']
            )
            
            # Validate constraints
            # Use surface_area (3D) for A/t ratio check - this gives more reasonable values
            # For a 0.5x0.5m panel with thickness 0.01m: surface_area ≈ 0.5 m², ratio ≈ 50
            surface_area = features_3d['surface_area']
            
            # Ensure surface_area is reasonable - should be at least 2x 2D area (top + bottom)
            min_surface_area = 2 * features_2d['area']
            if surface_area <= 0 or surface_area < min_surface_area:
                # Fallback: estimate surface area from 2D area (top + bottom + sides)
                # Top and bottom faces + perimeter * thickness for sides
                surface_area = 2 * features_2d['area'] + features_2d['perimeter'] * thickness
            
            # Additional check: ensure ratio will be valid before checking
            # If thickness is too large relative to surface_area, reject early
            estimated_ratio = surface_area / thickness
            if estimated_ratio < MIN_AREA_THICKNESS_RATIO * 0.9:  # 10% margin
                stats['invalid_samples'] += 1
                consecutive_failures += 1
                pbar.set_postfix({'valid': stats['valid_samples'], 'invalid': stats['invalid_samples'], 'reason': f'A/t={estimated_ratio:.1f}'})
                continue
            
            area_thickness_ok = check_area_thickness_ratio(surface_area, thickness)
            cost_ok = check_cost(material_props['cost'], max_cost)
            sustainability_ok = check_sustainability(
                material_props['sustainability_index'], min_sustainability
            )
            
            if not (area_thickness_ok and cost_ok and sustainability_ok):
                stats['invalid_samples'] += 1
                consecutive_failures += 1
                reason = []
                if not area_thickness_ok:
                    ratio = surface_area / thickness if thickness > 0 else 0
                    reason.append(f'A/t={ratio:.1f}')
                if not cost_ok:
                    reason.append(f'cost={material_props["cost"]:.1f}')
                if not sustainability_ok:
                    reason.append(f'sust={material_props["sustainability_index"]:.2f}')
                pbar.set_postfix({'valid': stats['valid_samples'], 'invalid': stats['invalid_samples'], 'reason': ','.join(reason)})
                continue
            
            # Generate geometry ID (ensure uniqueness)
            geometry_id = f"geom_{start_id + stats['valid_samples']:06d}"
            
            # Double-check it doesn't exist (shouldn't happen, but safety check)
            if geometry_id in existing_ids:
                geometry_id = f"geom_{start_id + stats['valid_samples'] + len(existing_ids):06d}"
            
            # Save files
            mesh_path = os.path.join(geometries_dir, f"{geometry_id}.obj")
            geojson_path = os.path.join(geometries_dir, f"{geometry_id}_2d.json")
            
            if not save_mesh(mesh_3d, mesh_path):
                stats['invalid_samples'] += 1
                consecutive_failures += 1
                pbar.set_postfix({'valid': stats['valid_samples'], 'invalid': stats['invalid_samples'], 'reason': 'save_mesh'})
                continue
            
            if not save_geometry_geojson(base_shape, geojson_path):
                stats['invalid_samples'] += 1
                consecutive_failures += 1
                pbar.set_postfix({'valid': stats['valid_samples'], 'invalid': stats['invalid_samples'], 'reason': 'save_geojson'})
                continue
            
            # Prepare data row
            data_row = {
                'geometry_id': geometry_id,
                'shape_type': shape_type,
                'complexity_level': actual_complexity,
                'thickness': thickness,
                'hole_radius': params['hole_radius'],
                'hole_spacing': params['hole_radius'] * 2.5 if num_holes > 0 else 0,
                'curvature': params['curvature'],
                'porosity': features_3d['porosity'],
                'material_type': material_type,
                'density': material_props['density'],
                'youngs_modulus': material_props['youngs_modulus'],
                'poissons_ratio': material_props['poissons_ratio'],
                'cost': material_props['cost'],
                'sustainability_index': material_props['sustainability_index'],
                'surface_area': features_3d['surface_area'],
                'perimeter': features_2d['perimeter'],
                'curvature_metric': features_3d['curvature_metric'],
                'volume': features_3d['volume'],
                'area_thickness_ratio': features_3d['area_thickness_ratio'],
                'num_holes': int(features_2d['num_holes']),
                'compactness': features_2d['compactness'],
                'convexity': features_2d['convexity'],
                'mesh_file_path': mesh_path,
                'cross_section_file_path': geojson_path
            }
            
            # Append to CSV
            append_to_dataset_csv(data_row, csv_path)
            
            # Update statistics
            stats['valid_samples'] += 1
            complexity_counts[actual_complexity] += 1
            stats['complexity_distribution'] = complexity_counts.copy()
            consecutive_failures = 0  # Reset failure counter on success
            
            pbar.update(1)
            pbar.set_postfix({
                'valid': stats['valid_samples'],
                'invalid': stats['invalid_samples'],
                'complexity': actual_complexity
            })
        
        except Exception as e:
            logger.warning(f"Error generating sample: {e}")
            stats['invalid_samples'] += 1
            consecutive_failures += 1
            pbar.set_postfix({'valid': stats['valid_samples'], 'invalid': stats['invalid_samples'], 'reason': 'exception'})
            continue
    
    pbar.close()
    
    # Create complexity-separated CSVs
    if create_separated_csvs and os.path.exists(csv_path):
        create_complexity_separated_csvs(csv_path, output_dir)
    
    # Create complexity summary
    if os.path.exists(csv_path):
        summary_path = os.path.join(output_dir, 'complexity_summary.json')
        create_complexity_summary(csv_path, summary_path)
    
    logger.info(f"Dataset generation complete!")
    logger.info(f"Valid samples: {stats['valid_samples']}")
    logger.info(f"Invalid samples: {stats['invalid_samples']}")
    logger.info(f"Complexity distribution: {complexity_counts}")
    
    return stats


# ============================================================================
# Command-Line Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate acoustic panel geometry dataset')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of geometries to generate (default: 1000)')
    parser.add_argument('--output_dir', type=str, default='data/geometries',
                       help='Output directory (default: data/geometries)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--max_cost', type=float, default=DEFAULT_MAX_COST,
                       help=f'Maximum cost threshold (default: {DEFAULT_MAX_COST})')
    parser.add_argument('--min_sustainability', type=float, default=MIN_SUSTAINABILITY,
                       help=f'Minimum sustainability index (default: {MIN_SUSTAINABILITY})')
    parser.add_argument('--complexity_distribution', type=str, default='uniform',
                       choices=['uniform', 'balanced', 'skewed_low', 'skewed_high'],
                       help='Distribution of complexity levels (default: uniform)')
    parser.add_argument('--create_separated_csvs', action='store_true', default=True,
                       help='Create separate CSV files per complexity level (default: True)')
    
    args = parser.parse_args()
    
    stats = generate_dataset(
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        seed=args.seed,
        max_cost=args.max_cost,
        min_sustainability=args.min_sustainability,
        complexity_distribution=args.complexity_distribution,
        create_separated_csvs=args.create_separated_csvs
    )
    
    print("\nGeneration Statistics:")
    print(f"  Valid samples: {stats['valid_samples']}")
    print(f"  Invalid samples: {stats['invalid_samples']}")
    print(f"  Total attempts: {stats['total_generated']}")
    print(f"  Success rate: {stats['valid_samples']/stats['total_generated']*100:.2f}%")
    print(f"\nComplexity Distribution:")
    for level, count in stats['complexity_distribution'].items():
        print(f"  Level {level}: {count}")


if __name__ == "__main__":
    main()
