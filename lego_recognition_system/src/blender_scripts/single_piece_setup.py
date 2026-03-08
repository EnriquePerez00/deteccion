import bpy
import sys
import json
import os
import random
import time
import shutil
from pathlib import Path
import math
from mathutils import Vector, Euler, Matrix
from bpy_extras.object_utils import world_to_camera_view

# Add current dir to path to find local modules if needed (optional)
import addon_utils

# Add current dir to path to find local modules
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

def setup_render_engine(engine='CYCLES', resolution=(2048, 2048)):
    """Configure render engine. Uses CYCLES or EEVEE_NEXT with optimized settings."""
    scene = bpy.context.scene
    print(f"🛠️ Configuring Render Engine: {engine} | Res: {resolution}")
    
    # Common render settings
    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]
    scene.render.resolution_percentage = 100
    
    if engine.upper() == 'EEVEE':
        eevee_names = ['BLENDER_EEVEE_NEXT', 'BLENDER_EEVEE']
        set_success = False
        for ee_name in eevee_names:
            try:
                scene.render.engine = ee_name
                set_success = True
                break
            except TypeError:
                continue
        
        if not set_success:
            print("  ⚠️ EEVEE not found in this version. Falling back to CYCLES.")
            scene.render.engine = 'CYCLES'
        else:
            if hasattr(scene, "eevee"):
                scene.eevee.use_shadows = True
                if hasattr(scene.eevee, "use_raytracing"):
                    scene.eevee.use_raytracing = True
            print(f"  ⚡ EEVEE ({scene.render.engine}) mode activated.")
        return

    # CYCLES Settings
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    prefs = bpy.context.preferences
    cprefs = prefs.addons['cycles'].preferences
    cprefs.get_devices()
    
    device_configured = False
    try:
        cprefs.compute_device_type = 'METAL'
        for device in cprefs.devices:
            device.use = True
            device_configured = True
            print(f"  🎨 DEVICE ENABLED: {device.name} ({device.type})")
        if hasattr(cprefs, "use_metalrt"):
            cprefs.use_metalrt = True
            print("  🚀 METAL RT: Hardware Ray Tracing activated!")
    except Exception as e:
        print(f"  ⚠️ Error configuring custom METAL setup: {e}")
        
    if not device_configured:
        for device_type in ['OPTIX', 'CUDA']:
            try:
                cprefs.compute_device_type = device_type
                for device in cprefs.devices:
                    if device.type != 'CPU':
                        device.use = True
                        device_configured = True
                if device_configured:
                    print(f"  🔬 CYCLES: Fallback to {device_type} hardware ray-tracing.")
                    break
            except Exception:
                pass
    
    if not device_configured:
        print("  ⚠️ CYCLES: No GPU found, falling back to CPU.")
        scene.cycles.device = 'CPU'
    
    scene.cycles.samples = 32  
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.05
    scene.cycles.glossy_bounces = 2
    
    denoiser_set = False
    for denoiser_name in ['OPENIMAGEDENOISE', 'OPTIX']:
        try:
            scene.cycles.use_denoising = True
            scene.cycles.denoiser = denoiser_name
            print(f"  🧠 Denoiser: {denoiser_name} active.")
            denoiser_set = True
            break
        except (TypeError, AttributeError):
            continue
    if not denoiser_set:
        scene.cycles.use_denoising = False
        print("  ℹ️ Denoising not supported on this Blender version. Disabled.")
    
    scene.render.image_settings.file_format = 'JPEG'
    scene.render.image_settings.quality = 90
    print("  🖼️ Output Format: JPEG (Quality: 90) active.")
    
    def set_safe(obj, attr, val):
        if hasattr(obj, attr):
            setattr(obj, attr, val)
            
    scene.cycles.max_bounces = 2
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 1
    set_safe(scene.cycles, 'transparent_max_bounces', 2) 
    set_safe(scene.cycles, 'transparent_bounces', 2)     
    scene.cycles.transmission_bounces = 0
    scene.cycles.caustics_reflective = False
    scene.cycles.caustics_refractive = False
    scene.render.use_persistent_data = True
    scene.cycles.use_auto_tile = False
    scene.render.film_transparent = True

    if hasattr(scene.view_settings, "view_transform"):
        vtrans = 'AgX' if bpy.app.version >= (4, 0, 0) else 'Filmic'
        try:
            scene.view_settings.view_transform = vtrans
            scene.view_settings.exposure = -0.7  
            try:
                if vtrans == 'AgX':
                    scene.view_settings.look = 'AgX - High Contrast'
                else:
                    scene.view_settings.look = 'High Contrast'
            except:
                pass 
        except Exception as e:
            print(f"  ⚠️ Warning: Color Management error: {e}")
    
    if hasattr(scene.cycles, "use_fast_gi"):
        scene.cycles.use_fast_gi = True
        try:
            scene.cycles.fast_gi_method = 'AO'
        except:
            scene.cycles.fast_gi_method = 'REPLACE'
    
    print(f"  ⚡ Render Config: 64 samples + {scene.view_settings.view_transform} + High Contrast active.")

def register_ldraw_addon(addon_path=None):
    """Register the ImportLDraw addon."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not addon_path:
        potential_path = os.path.join(script_dir, "ImportLDraw")
        if os.path.isdir(potential_path):
            addon_path = script_dir
    if addon_path and os.path.exists(addon_path):
        if addon_path not in sys.path:
            sys.path.append(addon_path)
    success = False
    for module_name in ['ImportLDraw', 'io_scene_importldraw']:
        try:
            mod = __import__(module_name)
            mod.register()
            success = True
            if bpy.app.version >= (4, 0, 0):
                orig_new = bpy.types.NodeTreeNodes.new
                def patched_new(self, node_type):
                    if node_type == 'ShaderNodeSeparateHSV':
                        actual_name = 'ShaderNodeSeparateColor' if 'ShaderNodeSeparateColor' in dir(bpy.types) else 'ShaderNodeSepColor'
                        node = orig_new(self, actual_name)
                        node.mode = 'HSV'
                        return node
                    if node_type == 'ShaderNodeSeparateRGB':
                        actual_name = 'ShaderNodeSeparateColor' if 'ShaderNodeSeparateColor' in dir(bpy.types) else 'ShaderNodeSepColor'
                        node = orig_new(self, actual_name)
                        node.mode = 'RGB'
                        return node
                    return orig_new(self, node_type)
                bpy.types.NodeTreeNodes.new = patched_new
            break
        except Exception:
            continue
    return success

def clean_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def snap_to_ground(objects, ground_z=0.0, max_hover=0.0005):
    snapped = 0
    for obj in objects:
        bpy.context.view_layer.update() 
        corners = get_hierarchy_corners(obj)
        if not corners: continue
        min_z = min(c.z for c in corners)
        if min_z > (ground_z + max_hover) or min_z < (ground_z - 0.0001):
            offset = ground_z - min_z + 0.0001
            obj.location.z += offset
            snapped += 1
    return snapped

def set_origin_to_center(obj):
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    obj.select_set(False)

def setup_camera():
    bpy.ops.object.camera_add(location=(0, 0, 0.7))
    cam = bpy.context.object
    cam.rotation_euler = (0, 0, 0)
    cam.data.lens = 26.0           
    cam.data.sensor_width = 36.0   
    cam.data.dof.use_dof = True
    cam.data.dof.focus_distance = 0.7  
    cam.data.dof.aperture_fstop = 11.0 
    bpy.context.scene.camera = cam
    return cam

def setup_lighting():
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if obj.type == 'LIGHT':
            obj.select_set(True)
    bpy.ops.object.delete()
    scenario = random.choice(['soft_window', 'overcast_sky', 'studio_softbox'])
    if scenario == 'soft_window':
        sign = random.choice([1, -1])
        bpy.ops.object.light_add(type='AREA', location=(2.0 * sign, 1.0, 1.5))
        key = bpy.context.object
        key.data.energy = random.uniform(1500, 2500)
        key.data.size = 2.5
        key.rotation_euler = (math.radians(-40), math.radians(50 * sign), 0)
        bpy.ops.object.light_add(type='AREA', location=(-2.0 * sign, -1.0, 1.0))
        fill = bpy.context.object
        fill.data.energy = random.uniform(300, 600)
        fill.data.size = 3.0
        fill.rotation_euler = (math.radians(-30), math.radians(-45 * sign), 0)
    elif scenario == 'overcast_sky':
        angles = [0, 120, 240]
        offset = random.uniform(0, 360)
        for ang in angles:
            rad = math.radians(ang + offset)
            x, y = math.cos(rad) * 2.0, math.sin(rad) * 2.0
            bpy.ops.object.light_add(type='AREA', location=(x, y, 1.2))
            l = bpy.context.object
            l.data.energy = random.uniform(400, 700)
            l.data.size = 4.0
            track = l.constraints.new('TRACK_TO')
            track.target = bpy.data.objects.get("Plane") 
            track.track_axis = 'TRACK_NEGATIVE_Z'
            track.up_axis = 'UP_Y'
    elif scenario == 'studio_softbox':
        bpy.ops.object.light_add(type='AREA', location=(1.5, 0.0, 1.0))
        key1 = bpy.context.object
        key1.data.energy = random.uniform(1000, 1500)
        key1.data.size = 2.0
        key1.rotation_euler = (0, math.radians(45), 0)
        bpy.ops.object.light_add(type='AREA', location=(-1.5, 0.0, 1.5))
        key2 = bpy.context.object
        key2.data.energy = random.uniform(800, 1200)
        key2.data.size = 2.0
        key2.rotation_euler = (0, math.radians(-45), 0)
    for obj in bpy.context.scene.objects:
        if obj.type == 'LIGHT':
            light = obj.data
            if hasattr(light, 'use_contact_shadow'):
                light.use_contact_shadow = True
    return scenario

def setup_compositor():
    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree if hasattr(scene, "node_tree") else scene.compositing_node_group
    if not tree: return
    nodes = tree.nodes
    links = tree.links
    nodes.clear()
    rl = nodes.new('CompositorNodeRLayers')
    comp = nodes.new('CompositorNodeComposite')
    links.new(rl.outputs[0], comp.inputs[0])

def setup_world_hdri(assets_dir):
    if not bpy.context.scene.world:
        bpy.context.scene.world = bpy.data.worlds.new("World")
    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    node_out = nodes.new(type='ShaderNodeOutputWorld')
    node_bg = nodes.new(type='ShaderNodeBackground')
    node_bg.inputs['Color'].default_value = (0.0, 0.0, 0.0, 1.0)
    links.new(node_bg.outputs['Background'], node_out.inputs['Surface'])

def setup_ground_texture(ground_obj, assets_dir):
    if not ground_obj.data.materials:
        mat = bpy.data.materials.new(name="GroundMaterial")
        ground_obj.data.materials.append(mat)
    else:
        mat = ground_obj.data.materials[0]
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    node_out = nodes.new(type='ShaderNodeOutputMaterial')
    node_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    node_bsdf.inputs['Base Color'].default_value = (0.0, 0.0, 0.0, 1.0)
    node_bsdf.inputs['Roughness'].default_value = 1.0
    mat.node_tree.links.new(node_bsdf.outputs['BSDF'], node_out.inputs['Surface'])

def _apply_bevel_to_mesh(obj):
    def bevel_object(o):
        if o.type != 'MESH':
            for child in o.children: bevel_object(child)
            return
        if any(m.name == "LEGO_Bevel" for m in o.modifiers):
            for child in o.children: bevel_object(child)
            return
        try:
            bevel = o.modifiers.new(name="LEGO_Bevel", type='BEVEL')
            bevel.width = 0.0004       
            bevel.segments = 2         
            bevel.limit_method = 'ANGLE'  
            bevel.angle_limit = math.radians(35) 
            bevel.use_clamp_overlap = True
        except: pass
        for child in o.children: bevel_object(child)
    bevel_object(obj)

def _apply_lego_material(obj, color_id=None):
    rgba = (0.5, 0.5, 0.5, 1.0)
    def apply_to_object(o):
        if o.type == 'MESH' and o.data:
            mat_name = f"LEGO_Color_{color_id}"
            mat = bpy.data.materials.get(mat_name)
            if not mat:
                mat = bpy.data.materials.new(name=mat_name)
                mat.use_nodes = True
                bsdf = next((n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
                if bsdf:
                    bsdf.inputs["Base Color"].default_value = rgba
                    bsdf.inputs["Roughness"].default_value = 0.20
            if not o.data.materials: o.data.materials.append(mat)
            else:
                for i in range(len(o.data.materials)): o.data.materials[i] = mat
        for child in o.children: apply_to_object(child)
    apply_to_object(obj)

def import_ldraw_part(filepath, ldraw_lib=None, color_id=None):
    try:
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.import_scene.importldraw(filepath=filepath, ldrawPath=ldraw_lib, addEnvironment=False)
        selected = bpy.context.selected_objects
        if not selected: return None
        obj = selected[0]
        while obj.parent: obj = obj.parent
        if color_id is not None:
            _apply_lego_material(obj, color_id)
            _apply_bevel_to_mesh(obj)
        return obj
    except: return None

def get_hierarchy_corners(obj):
    corners = []
    if obj.type == 'MESH':
        mw = obj.matrix_world
        corners.extend([mw @ Vector(corner) for corner in obj.bound_box])
    for child in obj.children: corners.extend(get_hierarchy_corners(child))
    return corners

def copy_hierarchy(obj, parent=None):
    new_obj = obj.copy()
    if obj.data: new_obj.data = obj.data.copy()
    if parent: new_obj.parent = parent
    bpy.context.collection.objects.link(new_obj)
    for key, value in obj.items(): new_obj[key] = value
    for child in obj.children: copy_hierarchy(child, parent=new_obj)
    return new_obj

def get_hierarchy_vertices(obj):
    verts = []
    if obj.type == 'MESH':
        mw = obj.matrix_world
        verts.extend([mw @ v.co for v in obj.data.vertices])
    for child in obj.children: verts.extend(get_hierarchy_vertices(child))
    return verts

def get_geometry_aabb(obj):
    verts = get_hierarchy_vertices(obj)
    if not verts: return Vector((0,0,0)), 0.05
    xs, ys, zs = [v.x for v in verts], [v.y for v in verts], [v.z for v in verts]
    min_v, max_v = Vector((min(xs), min(ys), min(zs))), Vector((max(xs), max(ys), max(zs)))
    center = (min_v + max_v) / 2.0
    max_d = max([(v - center).length for v in verts])
    return center, max_d

def get_convex_hull(points):
    points = sorted(set((round(p[0], 6), round(p[1], 6)) for p in points))
    if len(points) <= 1: return points
    def cross(o, a, b): return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0: lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0: upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]

def generate_yolo_bbox_label(obj, scene, class_id):
    verts_3d = get_hierarchy_vertices(obj)
    if not verts_3d: return None
    coords_2d = [world_to_camera_view(scene, scene.camera, v) for v in verts_3d]
    x_vals = [c.x for c in coords_2d if c.z > 0]; y_vals = [c.y for c in coords_2d if c.z > 0]
    if not x_vals or not y_vals: return None
    x_center, y_center = (min(x_vals) + max(x_vals)) / 2.0, 1.0 - ((min(y_vals) + max(y_vals)) / 2.0)
    w_box, h_box = max(x_vals) - min(x_vals), max(y_vals) - min(y_vals)
    return f"{class_id} {x_center:.6f} {y_center:.6f} {w_box:.6f} {h_box:.6f}"

def generate_yolo_seg_label(obj, scene):
    verts_3d = get_hierarchy_vertices(obj)
    if not verts_3d: return None
    coords_2d = [world_to_camera_view(scene, scene.camera, coord) for coord in verts_3d]
    visible_points = [(max(0.0, min(1.0, c.x)), max(0.0, min(1.0, c.y))) for c in coords_2d if c.z > 0]
    if not visible_points: return None
    hull = get_convex_hull(visible_points)
    if not hull or len(hull) < 3: return None
    pts = []
    for corner in hull: pts.extend([f"{corner[0]:.6f}", f"{1.0 - corner[1]:.6f}"])
    return f"{obj.get('class_id', 0)} {' '.join(pts)}"

def write_image_meta(meta_path, img_prefix, ids, color_ids, color_names):
    with open(meta_path, 'a') as mf:
        mf.write(json.dumps({"img": f"{img_prefix}.jpg", "ids": ids, "color_ids": color_ids, "color_names": color_names}) + "\n")

def main():
    scene = bpy.context.scene
    argv = sys.argv
    try:
        idx = argv.index("--")
        data_file = argv[idx + 1]
    except: return
    with open(data_file, 'r') as f: data = json.load(f)
    render_mode = data.get('render_mode', 'images_mix')
    pieces_config = data.get('pieces_config', [])
    output_base = data.get('output_base', '')
    assets_dir = data.get('assets_dir')
    ldraw_path_base = data.get('ldraw_path')
    register_ldraw_addon(data.get('addon_path'))
    clean_scene()
    setup_render_engine(resolution=(data.get('resolution_x', 768), data.get('resolution_y', 768)))
    setup_compositor()
    cam = setup_camera()
    setup_lighting()
    setup_world_hdri(assets_dir)
    bpy.ops.mesh.primitive_plane_add(size=0.5, location=(0, 0, 0))
    ground = bpy.context.object
    ground.name = "Plane"
    ground.hide_render = True

    try:
        from ldraw_resolver import LDrawResolver
        resolver = LDrawResolver(ldraw_path_base)
    except: resolver = None

    unique_meshes = []
    for i, pc in enumerate(pieces_config):
        part = pc['part']
        ldraw_id = part['ldraw_id']
        class_id = 0 if os.environ.get("UNIVERSAL_DETECTOR", "0") == "1" else i
        part_path = resolver.find_part(ldraw_id) if resolver else os.path.join(ldraw_path_base, "parts", f"{ldraw_id}.dat")
        obj = import_ldraw_part(part_path, ldraw_path_base, color_id=part.get('color_id'))
        if obj:
            set_origin_to_center(obj)
            unique_meshes.append({'obj': obj, 'id': class_id, 'ldraw_id': ldraw_id, 'color_id': part.get('color_id'), 'color_name': part.get('color_name')})
            obj.location = (0, 0, -10)

    images_dir, labels_dir = os.path.join(output_base, "images"), os.path.join(output_base, "labels")
    os.makedirs(images_dir, exist_ok=True); os.makedirs(labels_dir, exist_ok=True)

    if render_mode == 'ref_pieza' and unique_meshes:
        template = unique_meshes[0]
        template_obj = template['obj']
        p_id = template['ldraw_id']
        template_obj.location = (0,0,0)
        
        # Priority: nested config -> root config -> default
        pc0 = pieces_config[0] if pieces_config else {}
        num_meta = pc0.get('numbering_meta', {})
        
        stable_matrix = pc0.get('stable_matrix') or data.get('stable_matrix')
        symmetry_order = pc0.get('symmetry_order') or data.get('symmetry_order', 1)
        num_rot_images = num_meta.get('ref_num_images') or pc0.get('ref_num_images') or data.get('ref_num_images', 24)
        offset_idx = num_meta.get('offset_idx') if 'offset_idx' in num_meta else (pc0.get('offset_idx') if 'offset_idx' in pc0 else data.get('offset_idx', 0))
        
        rotation_step_deg = (360.0 / symmetry_order) / max(1, num_rot_images)
        for r in range(num_rot_images):
            if stable_matrix: template_obj.matrix_world = Matrix(stable_matrix)
            bpy.context.view_layer.update()
            spin_rad = (r * rotation_step_deg) * (math.pi / 180.0)
            template_obj.matrix_world = template_obj.matrix_world @ Matrix.Rotation(spin_rad, 4, 'Z')
            template_obj.location.x = template_obj.location.y = 0
            bpy.context.view_layer.update()
            corners = get_hierarchy_corners(template_obj)
            if corners: template_obj.location.z += (0.0001 - min(c.z for c in corners))
            aabb_center, aabb_radius = get_geometry_aabb(template_obj)
            target_dist = (max(aabb_radius, 0.005) * 26.0) / (0.40 * 18.0)
            real_dist = max(0.01, 0.70 - (aabb_center.z + template_obj.dimensions.z / 2.0))
            cam.location = (aabb_center.x, aabb_center.y, 0.70)
            cam.data.sensor_width = 36.0 * (target_dist / real_dist)
            cam.data.dof.focus_distance = real_dist
            
            # --- NEW NAMING CONVENTION: {PART_ID}_XXXX.jpg ---
            img_prefix = f"{p_id}_{r + offset_idx:04d}"
            scene.render.filepath = os.path.join(images_dir, f"{img_prefix}.jpg")
            bpy.ops.render.render(write_still=True)
            with open(os.path.join(labels_dir, f"{img_prefix}.txt"), 'w') as lf:
                lf.write(generate_yolo_bbox_label(template_obj, bpy.context.scene, template['id']) + "\n")
            write_image_meta(os.path.join(output_base, "image_meta.jsonl"), img_prefix, [template['ldraw_id']], [template['color_id']], [template['color_name']])
        return

    # --- MODE: images_mix ---
    num_images = data.get('num_images', 30)
    num_drops = max(1, num_images // 2)
    for drop_idx in range(num_drops):
        scene.frame_set(1); bpy.ops.ptcache.free_bake_all()
        current_parts_count = random.randint(5, 12)
        selected_templates = random.sample(unique_meshes, min(len(unique_meshes), current_parts_count))
        active_pieces = []
        for template in selected_templates:
            new_obj = copy_hierarchy(template['obj'])
            new_obj.location = (random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), random.uniform(0.05, 0.15))
            new_obj.rotation_euler = (random.random()*6.28, random.random()*6.28, random.random()*6.28)
            new_obj.hide_render = False
            bpy.context.view_layer.objects.active = new_obj
            bpy.ops.rigidbody.object_add(); new_obj.rigid_body.type = 'ACTIVE'
            active_pieces.append(new_obj)
        for f in range(1, 150): scene.frame_set(f)
        snap_to_ground(active_pieces)
        for var_idx in range(2):
            local_idx = (drop_idx * 2) + var_idx
            if local_idx >= num_images: break
            img_prefix = f"img_{local_idx + data.get('offset_idx', 0):04d}"
            scene.render.filepath = os.path.join(images_dir, f"{img_prefix}.jpg")
            bpy.ops.render.render(write_still=True)
            with open(os.path.join(labels_dir, f"{img_prefix}.txt"), 'w') as lf:
                for obj in active_pieces:
                    seg = generate_yolo_seg_label(obj, scene)
                    if seg: lf.write(seg + "\n")

if __name__ == "__main__":
    main()
