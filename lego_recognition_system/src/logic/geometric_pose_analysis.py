import os
import numpy as np
from scipy.spatial import ConvexHull
import logging

logger = logging.getLogger("LegoVision")

def parse_ldraw_vertices(ldraw_path, resolver=None, visited=None, transform=None):
    """
    Recursively parses LDraw file to extract all unique vertex positions.
    transform: 4x4 matrix
    """
    if visited is None:
        visited = set()
    if transform is None:
        transform = np.eye(4)
    
    abs_path = os.path.abspath(ldraw_path)
    if abs_path in visited:
        return []
    visited.add(abs_path)
    
    vertices = []
    
    try:
        with open(ldraw_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                line_type = parts[0]
                
                if line_type == '1': # Sub-file
                    if len(parts) < 15: continue
                    # 1 <colour> x y z a b c d e f g h i <file>
                    x, y, z = map(float, parts[2:5])
                    a, b, c, d, e, f, g, h, i = map(float, parts[5:14])
                    sub_file = " ".join(parts[14:])
                    
                    # Local transform matrix
                    local_m = np.array([
                        [a, b, c, x],
                        [d, e, f, y],
                        [g, h, i, z],
                        [0, 0, 0, 1]
                    ])
                    # Cumulative transform
                    new_transform = transform @ local_m
                    
                    if resolver:
                        sub_path = resolver.find_part(sub_file)
                        if sub_path:
                            vertices.extend(parse_ldraw_vertices(sub_path, resolver, visited, new_transform))
                
                elif line_type in ('3', '4'): # Tri/Quad
                    # Tri: 3 <col> x1 y1 z1 x2 y2 z2 x3 y3 z3
                    # Quad: 4 <col> x1 y1 z1 x2 y2 z2 x3 y3 z3 x4 y4 z4
                    v_count = 3 if line_type == '3' else 4
                    coords = parts[2:2+v_count*3]
                    for j in range(0, len(coords), 3):
                        v_local = np.array([float(coords[j]), float(coords[j+1]), float(coords[j+2]), 1.0])
                        v_world = transform @ v_local
                        vertices.append(v_world[:3])
                        
    except Exception as e:
        logger.warning(f"Error parsing {ldraw_path}: {e}")
        
    return vertices

def get_stable_poses(vertices):
    """
    Phase 1: Stable Poses Determination.
    Uses Convex Hull and Center of Mass (CoM).
    Returns a list of transformation matrices (4x4) that place the part 
    stably on the XY plane (Z=0).
    """
    if len(vertices) < 4:
        return [np.eye(4)]
    
    pts = np.array(vertices)
    # 1. Center of Mass
    com = np.mean(pts, axis=0)
    
    # 2. Convex Hull
    try:
        hull = ConvexHull(pts)
    except:
        return [np.eye(4)]
    
    stable_poses = []
    
    # Scale of the part for thresholds
    dims = np.max(pts, axis=0) - np.min(pts, axis=0)
    eps = np.mean(dims) * 0.05  # Increased eps for stability

    # hull.equations returns (normal_x, normal_y, normal_z, offset) for each face
    # where normal is outward-pointing.
    for eq in hull.equations:
        normal = eq[:3]
        # Distance from CoM to face plane: dot(normal, CoM) + offset
        # Since equations are normal * X + offset = 0, and normal is outward,
        # points inside the hull have normal * X + offset < 0.
        # So dist_to_plane = normal * CoM + offset should be < 0.
        dist = np.dot(normal, com) + eq[3]
        
        # If dist is positive, CoM is outside? That shouldn't happen for a hull.
        # We want the face that is 'closest' to the CoM in the direction of gravity.
        # But we actually want ALL stable faces.
        
        # Projection of CoM onto face plane
        com_proj = com - dist * normal
        
        # Check if projection is inside polygon (2D check)
        # Use simple barycentric or cross-product sum for triangle
        # For simplicity in 3D: check if com_proj is within convex combinations of face_pts
        # Or even simpler: if it's a stable rest, the CoM should be 'above' the face
        
        # To decide if it's stable:
        # 1. The projection must be within the convex hull of the face points
        # (This is equivalent to saying the part won't tip)
        
        # Find all vertices on this face plane
        # eq: ax + by + cz + d = 0
        dist_all = np.dot(pts, normal) + eq[3]
        face_pts = pts[np.abs(dist_all) < eps]
        if len(face_pts) < 3: continue

        # Project 3D points to 2D onto the plane of the face
        # Choose two orthogonal axes on the plane
        # Find a vector not parallel to normal
        ref_vec = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
        u = np.cross(normal, ref_vec)
        u /= np.linalg.norm(u)
        w = np.cross(normal, u)
        
        face_2d = np.array([[np.dot(p - face_pts[0], u), np.dot(p - face_pts[0], w)] for p in face_pts])
        com_2d = np.array([np.dot(com_proj - face_pts[0], u), np.dot(com_proj - face_pts[0], w)])
        
        # Use a 2D Convex Hull check for com_2d inside face_2d
        try:
            face_hull = ConvexHull(face_2d)
            # If we add com_2d and it doesn't change the hull, it's inside
            new_hull = ConvexHull(np.vstack([face_2d, com_2d]))
            if len(new_hull.vertices) == len(face_hull.vertices):
                # Potential stable pose!
                # We need a matrix that rotates 'normal' to -Z axis [0, 0, -1]
                # And translates to put face at Z=0
                
                # Z_axis moves to normal (or -normal to face down)
                # target_z = -normal (if we want the face to touch the ground)
                target_z = normal
                
                # Find rotation matrix from target_z to [0, 0, 1]
                # Then the face will be flat on XY plane
                z_vec = np.array([0, 0, 1])
                
                if np.allclose(target_z, z_vec):
                    rot = np.eye(3)
                elif np.allclose(target_z, -z_vec):
                    rot = np.diag([1, -1, -1])
                else:
                    v = np.cross(target_z, z_vec)
                    s = np.linalg.norm(v)
                    c = np.dot(target_z, z_vec)
                    v_skew = np.array([
                        [0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]
                    ])
                    rot = np.eye(3) + v_skew + (v_skew @ v_skew) * ((1 - c) / (s**2))
                
                # Full matrix
                m = np.eye(4)
                m[:3, :3] = rot
                
                # Translation to put min Z at 0
                rotated_pts = pts @ rot.T
                m[2, 3] = -np.min(rotated_pts[:, 2])
                
                # Deduplication: Check if this rotation is very similar to already found ones
                is_duplicate = False
                for existing_m in stable_poses:
                    # Normals must be aligned to be the same pose
                    existing_rot = existing_m[:3, :3]
                    # The normal we used was 'normal' -> [0,0,1]
                    # So current rotation matrix 'rot' satisfies rot @ normal = [0,0,1]
                    # Let's just compare the rotation matrices directly
                    if np.allclose(rot, existing_rot, atol=0.08):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    stable_poses.append(m)
        except:
            continue
            
    return stable_poses if stable_poses else [np.eye(4)]

def analyze_2d_symmetry(vertices, pose_matrix):
    """
    Phase 2: Zenithal Symmetry Analysis.
    Projects vertices onto XY plane after applying pose_matrix.
    Returns symmetry order: 1 (C1), 2 (C2), or 4 (C4).
    """
    pts = np.array(vertices)
    # Apply pose
    pts_hom = np.hstack([pts, np.ones((len(pts), 1))])
    pts_posed = (pose_matrix @ pts_hom.T).T[:, :3]
    
    # Project to 2D
    pts_2d = pts_posed[:, :2]
    # Center 2D points
    center = np.mean(pts_2d, axis=0)
    pts_2d -= center
    
    # Calculate 2D moments or simple point-cloud symmetry
    # Try 90 deg and 180 deg rotations
    def get_diff(angle_deg):
        rad = np.radians(angle_deg)
        c, s = np.cos(rad), np.sin(rad)
        rot_m = np.array([[c, -s], [s, c]])
        pts_rot = pts_2d @ rot_m.T
        
        # Simple Hausdorf-like metric: sum of distances to nearest point
        # For LEGO, exact matching is expected
        # Since point clouds can have different density, this is tricky
        # But we parsed triangles, so we have vertices.
        
        # Let's use 2D moment of inertia as a hint
        return np.allclose(pts_2d.T @ pts_2d, pts_rot.T @ pts_rot, atol=1e-1)

    # Check for C4 (90 deg)
    if get_diff(90):
        return 4
    # Check for C2 (180 deg)
    if get_diff(180):
        return 2
    
    return 1

def get_pose_universe(ldraw_path, resolver=None):
    """
    Master function for the unified pipeline.
    """
    vertices = parse_ldraw_vertices(ldraw_path, resolver)
    if not vertices:
        return [{"matrix": np.eye(4).tolist(), "symmetry": 1}]
        
    stable_poses = get_stable_poses(vertices)
    results = []
    for m in stable_poses:
        sym = analyze_2d_symmetry(vertices, m)
        results.append({
            "matrix": m.tolist(),
            "symmetry": sym
        })
    return results
