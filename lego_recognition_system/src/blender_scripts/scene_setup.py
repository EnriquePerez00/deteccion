import bpy
import sys
import json
import os
import random
import time
import shutil
from pathlib import Path
import math
from mathutils import Vector, Euler
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
        # Try different EEVEE internal names (Blender 4.2+ vs Blender 3.x/5.0+)
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
            # Basic enhancements for EEVEE
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
    
    # Apple Silicon M4 Strategy: Force METAL with Hardware Ray Tracing (MetalRT)
    # Also attempt HYBRID rendering (GPU + CPU) since they share memory.
    try:
        cprefs.compute_device_type = 'METAL'
        for device in cprefs.devices:
            # Enable ALL devices for hybrid rendering (CPU + GPU)
            device.use = True
            device_configured = True
            print(f"  🎨 DEVICE ENABLED: {device.name} ({device.type})")
        
        # Enable Hardware Ray Tracing if supported (M2/M3/M4)
        if hasattr(cprefs, "use_metalrt"):
            cprefs.use_metalrt = True
            print("  🚀 METAL RT: Hardware Ray Tracing activated!")
            
    except Exception as e:
        print(f"  ⚠️ Error configuring custom METAL setup: {e}")
        
    if not device_configured:
        # Fallback for standard PC configurations
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
    
    # Eco-Mode: fast rendering, try AI denoising if supported by this Blender version
    scene.cycles.samples = 32  
    scene.cycles.use_adaptive_sampling = True
    # 0.01 threshold (high precision "real resolution" for maximum detail)
    scene.cycles.adaptive_threshold = 0.05
    
    # glossy_bounces=2 to capture specular highlights on LEGO studs and edges
    scene.cycles.glossy_bounces = 2
    
    # Try to enable AI denoising — Blender 3.x and 4.x have different APIs
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
    
    # 📸 Output Format: JPEG (Quality 90) for storage efficiency
    scene.render.image_settings.file_format = 'JPEG'
    scene.render.image_settings.quality = 90
    print("  🖼️ Output Format: JPEG (Quality: 90) active.")
    
    # Minimal bounces for speed - Use safe attribute setting for version compatibility
    def set_safe(obj, attr, val):
        if hasattr(obj, attr):
            setattr(obj, attr, val)
            
    scene.cycles.max_bounces = 2
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 1
    set_safe(scene.cycles, 'transparent_max_bounces', 2) # New versions
    set_safe(scene.cycles, 'transparent_bounces', 2)     # Older versions
    scene.cycles.transmission_bounces = 0
    scene.cycles.caustics_reflective = False
    scene.cycles.caustics_refractive = False
    scene.render.use_persistent_data = True
    
    # ⚡ GPU RENDERING OPTIMIZATION (T4 specific)
    # Disable tiling to let the GPU render the whole frame at once. CPU doesn't have to interrupt the GPU.
    scene.cycles.use_auto_tile = False
    
    # 🎞️ Film transparency enabled to allow compositing over a background image
    scene.render.film_transparent = True

    # 🎨 Color Management: AgX for highlight preservation (prevents color wash-out)
    if hasattr(scene.view_settings, "view_transform"):
        # AgX is superior in Blender 4.0+, Filmic fallback for older
        vtrans = 'AgX' if bpy.app.version >= (4, 0, 0) else 'Filmic'
        try:
            scene.view_settings.view_transform = vtrans
            scene.view_settings.exposure = -0.7  # Prevent clipping in highlights
            
            # Set Look based on transform - AgX uses prefixes in 4.x/5.x
            try:
                if vtrans == 'AgX':
                    scene.view_settings.look = 'AgX - High Contrast'
                else:
                    scene.view_settings.look = 'High Contrast'
            except:
                pass # Look not supported or named differently
        except Exception as e:
            print(f"  ⚠️ Warning: Color Management error: {e}")
    
    # Shadows & AO: Fast GI Approximation in Cycles
    if hasattr(scene.cycles, "use_fast_gi"):
        scene.cycles.use_fast_gi = True
        try:
            scene.cycles.fast_gi_method = 'AO'
        except:
            # Blender 4.x/5.x fallback. Method 'AO' was renamed or is implicitly 'REPLACE' with AO settings.
            # Usually, the options are ('REPLACE', 'ADD') now.
            scene.cycles.fast_gi_method = 'REPLACE'
    
    print(f"  ⚡ Render Config: 64 samples + {scene.view_settings.view_transform} + High Contrast active.")

def register_ldraw_addon(addon_path=None):
    """Register the ImportLDraw addon from a dynamic path or local script dir."""
    print("🚀 Running register_ldraw_addon (CODE VERSION 2.2)")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not addon_path:
        # Try to find ImportLDraw folder next to this script
        potential_path = os.path.join(script_dir, "ImportLDraw")
        if os.path.isdir(potential_path):
            addon_path = script_dir
            print(f"📦 Found local ImportLDraw addon at: {potential_path}")

    if addon_path and os.path.exists(addon_path):
        if addon_path not in sys.path:
            sys.path.append(addon_path)
        print(f"📦 Added addon path to sys.path: {addon_path}")

    # Try to import and register. Support both 'ImportLDraw' and 'io_scene_importldraw' (standard)
    success = False
    import_errors = []
    for module_name in ['ImportLDraw', 'io_scene_importldraw']:
        try:
            mod = __import__(module_name)
            mod.register()
            print(f"✅ Addon registered successfully via module: {module_name}")
            success = True
            
            # Monkeypatch for Blender 4.0+ compatibility (ImportLDraw fix)
            # The addon uses ShaderNodeSeparateHSV which is now ShaderNodeSepColor
            if bpy.app.version >= (4, 0, 0):
                print("🐒 Monkeypatching NodeTreeNodes.new for Blender 4.0+ compatibility...")
                
                orig_new = bpy.types.NodeTreeNodes.new
                
                def patched_new(self, node_type):
                    if node_type == 'ShaderNodeSeparateHSV':
                        # In Blender 5.0 it is ShaderNodeSeparateColor
                        # In Blender 4.x it might be ShaderNodeSepColor or ShaderNodeSeparateColor
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
                
                # Apply the patch to the class
                bpy.types.NodeTreeNodes.new = patched_new
            break
        except Exception as e:
            import_errors.append(f"{module_name}: {e}")
            continue

    if not success:
        print("⚠️ Failed to register ImportLDraw addon via ImportLDraw or io_scene_importldraw.")
        print(f"   Import Exceptions: {import_errors}")
        print("   Checking if it's already registered via standard installation...")
        try:
            if "importldraw" not in dir(bpy.ops.import_scene):
                raise Exception("❌ CRITICAL: ImportLDraw addon is NOT available. Aborting.")
        except Exception as e:
            print(f"❌ Error checking bpy.ops.import_scene: {e}")
            raise

def clean_scene():
    """Remove all objects from the scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def snap_to_ground(objects, ground_z=0.0, max_hover=0.0005):
    """Force any piece still hovering above ground contact.
    Uses bounding box to find the absolute bottom point of the piece.
    """
    snapped = 0
    for obj in objects:
        bpy.context.view_layer.update() # Ensure matrices are current
        corners = get_hierarchy_corners(obj)
        if not corners: continue
        
        min_z = min(c.z for c in corners)
        
        # If it's too high or buried below surface
        if min_z > (ground_z + max_hover) or min_z < (ground_z - 0.0001):
            # Shift the whole object hierarchy up so min_z is at ground_z + tiny margin
            # Margin avoids Z-fighting and ensures shadows render correctly
            offset = ground_z - min_z + 0.0001
            obj.location.z += offset
            snapped += 1
            
    if snapped > 0:
        print(f"  📌 Hierarchy snap: {snapped} pieces adjusted to ground contact.")

def set_origin_to_center(obj):
    """Sets the origin of the object to its geometric center.
    This is critical because LDraw origins are often at the top/bottom or far from center,
    causing rotations to be eccentric or physics to biastype towards certain faces.
    """
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    # Origin to Geometry (Center of Mass / Bounds Center)
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    obj.select_set(False)
    print(f"  🎯 Origin centered for {obj.name}")

def setup_camera():
    """Create a camera at 70cm height with 26mm lens (iPhone 16 simulation).
    iPhone 16 Main Camera: 26mm focal length (35mm equivalent), 36mm sensor width.
    """
    bpy.ops.object.camera_add(location=(0, 0, 0.7))
    cam = bpy.context.object
    # In Blender 5.0, camera_add() creates a camera looking straight DOWN (-Z world)
    # with the default rotation_euler = (0, 0, 0). This is the cenital / top-down view.
    cam.rotation_euler = (0, 0, 0)
    cam.data.lens = 26.0           # 26mm focal length
    cam.data.sensor_width = 36.0   # Full-frame equivalent
    
    # Sharp Focus at 70cm
    cam.data.dof.use_dof = True
    cam.data.dof.focus_distance = 0.7  # Sharp on the ground/surface
    cam.data.dof.aperture_fstop = 11.0 # f/11 for deep focus (keeps piece tops sharp)
    cam.data.dof.aperture_blades = 7
    
    bpy.context.scene.camera = cam
    return cam

def setup_lighting():
    """Create realistic, soft natural lighting mimicking windows or overcast skies. No harsh points to avoid zenithal glare."""
    # Delete existing lights
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if obj.type == 'LIGHT':
            obj.select_set(True)
    bpy.ops.object.delete()

    # Base render color management for filmic/AgX to prevent highlight clipping (burnout)
    try:
        bpy.context.scene.view_settings.view_transform = 'AgX'
        bpy.context.scene.view_settings.look = 'Medium Contrast'
    except Exception as e:
        print(f"  ⚠️ Warning: AgX not found (using older Blender version?): {e}")

    # Scenarios focused on soft, wide, off-axis lighting
    scenario = random.choice(['soft_window', 'overcast_sky', 'studio_softbox'])

    if scenario == 'soft_window':
        # Large window-like light from the side, angled down
        # X: 2.0, Z: 1.5 -> approx 45-60 deg angle. Avoid X,Y=0 to prevent direct reflection to camera
        sign = random.choice([1, -1])
        bpy.ops.object.light_add(type='AREA', location=(2.0 * sign, 1.0, 1.5))
        key = bpy.context.object
        key.data.energy = random.uniform(1500, 2500)
        key.data.color = (1.0, random.uniform(0.95, 1.0), random.uniform(0.9, 1.0)) # Daylight
        key.data.shape = 'RECTANGLE'
        key.data.size = 2.5
        key.data.size_y = 3.5
        # Point the light roughly towards the center (0,0,0)
        key.rotation_euler = (math.radians(-40), math.radians(50 * sign), 0)
        
        # Soft, weak fill light on the opposite side to lift deep shadows
        bpy.ops.object.light_add(type='AREA', location=(-2.0 * sign, -1.0, 1.0))
        fill = bpy.context.object
        fill.data.energy = random.uniform(300, 600)
        fill.data.color = (random.uniform(0.8, 0.9), random.uniform(0.85, 0.95), 1.0) # Sky fill
        fill.data.size = 3.0
        fill.rotation_euler = (math.radians(-30), math.radians(-45 * sign), 0)

    elif scenario == 'overcast_sky':
        # Extremely soft, multi-directional weak light mimicking cloud cover
        # We put 3 large area lights in a ring, avoiding direct top
        angles = [0, 120, 240]
        offset = random.uniform(0, 360)
        for ang in angles:
            rad = math.radians(ang + offset)
            # Distance 2m, height 1m
            x = math.cos(rad) * 2.0
            y = math.sin(rad) * 2.0
            bpy.ops.object.light_add(type='AREA', location=(x, y, 1.2))
            l = bpy.context.object
            l.data.energy = random.uniform(400, 700)
            l.data.color = (random.uniform(0.95, 1.0), random.uniform(0.95, 1.0), 1.0)
            l.data.size = 4.0
            # Rough aim to center
            # pitch down ~30 deg, yaw towards center
            yaw = rad + math.pi
            l.rotation_euler = (math.radians(60), 0, yaw) # standard Euler rotation for tracking might vary, approximate:
            
            # Simple tracking constraint pointing to world origin
            track = l.constraints.new('TRACK_TO')
            track.target = bpy.data.objects.get("Plane") # track the ground plane, effectively origin
            track.track_axis = 'TRACK_NEGATIVE_Z'
            track.up_axis = 'UP_Y'
            
    elif scenario == 'studio_softbox':
        # Two massive softboxes on left and right, neutral color
        bpy.ops.object.light_add(type='AREA', location=(1.5, 0.0, 1.0))
        key1 = bpy.context.object
        key1.data.energy = random.uniform(1000, 1500)
        key1.data.shape = 'SQUARE'
        key1.data.size = 2.0
        key1.rotation_euler = (0, math.radians(45), 0)
        
        bpy.ops.object.light_add(type='AREA', location=(-1.5, 0.0, 1.5))
        key2 = bpy.context.object
        key2.data.energy = random.uniform(800, 1200)
        key2.data.shape = 'SQUARE'
        key2.data.size = 2.0
        key2.rotation_euler = (0, math.radians(-45), 0)

    print(f"  💡 Natural Lighting scenario: '{scenario}'")
    
    # --- EEVEE Contact Shadows: Activate on all lights to darken contact points ---
    for obj in bpy.context.scene.objects:
        if obj.type == 'LIGHT':
            light = obj.data
            # Contact Shadows (EEVEE only, safe to set even in CYCLES)
            if hasattr(light, 'use_contact_shadow'):
                light.use_contact_shadow = True
                light.contact_shadow_distance = 0.05  # 5cm range
                light.contact_shadow_thickness = 0.005
            # Soften shadows with larger light radius/size
            if light.type == 'POINT' or light.type == 'SPOT':
                if hasattr(light, 'shadow_soft_size'):
                    light.shadow_soft_size = max(light.shadow_soft_size, 0.05)
            elif light.type == 'AREA':
                if hasattr(light, 'size'):
                    light.size = max(light.size, 0.3)
    
    return scenario

def setup_compositor():
    """Add a subtle sensor noise layer to simulate iPhone camera grain.
    Hyper-robust version for Blender 4.x and 5.x.
    """
    print("🎨 Setting up Compositor Post-Processing...")
    # Prefer direct access to scene data
    scene = bpy.data.scenes[0] if bpy.data.scenes else bpy.context.scene
    scene.use_nodes = True
    
    # CASE: node_tree/compositing_node_group handling (Blender 5.0 compatibility)
    tree = None
    if hasattr(scene, "node_tree"):
        tree = scene.node_tree
    elif hasattr(scene, "compositing_node_group"):
        tree = scene.compositing_node_group
        
    if tree is None:
        print("  🔧 Forced reconstruction of Compositor NodeTree...")
        try:
            new_tree = bpy.data.node_groups.new("CompositorNodeTree", "CompositorNodeTree")
            if hasattr(scene, "node_tree"):
                scene.node_tree = new_tree
                tree = scene.node_tree
            elif hasattr(scene, "compositing_node_group"):
                scene.compositing_node_group = new_tree
                tree = scene.compositing_node_group
        except Exception as e:
            print(f"  ⚠️ Failed to create new NodeTree: {e}")
        
    if tree is None:
        print("  ❌ CRITICAL: Absolute Compositor failure - No node_tree/compositing_node_group.")
        return

    nodes = tree.nodes
    links = tree.links
    
    # Clear existing nodes to start fresh
    try:
        nodes.clear()
    except Exception as e:
        print(f"  ⚠️ Error clearing nodes: {e}")

    def safe_node_new(type_name, label=None):
        """Try to create a node by various possible names (version compatibility)."""
        node = None
        try:
            node = nodes.new(type=type_name)
        except:
            alt_name = type_name.replace('CompositorNode', '')
            try:
                node = nodes.new(type=alt_name)
            except:
                # Common aliases for Blender 5.0
                if type_name == 'CompositorNodeComposite':
                    for fallback in ['NodeGroupOutput', 'CompositorNodeGroupOutput', 'CompositorNodeViewer']:
                        try:
                            node = nodes.new(type=fallback)
                            print(f"    💡 Mapping {type_name} -> {fallback}")
                            break
                        except: pass
                elif type_name == 'CompositorNodeSharpen':
                    try:
                        node = nodes.new(type='CompositorNodeFilter')
                        node.filter_type = 'SHARPEN'
                        print(f"    💡 Mapping {type_name} -> Filter(SHARPEN)")
                    except: pass
        
        if node and label:
            node.label = label
        return node

    # Core nodes
    render_layers = safe_node_new('CompositorNodeRLayers', 'Render Layers')
    composite = safe_node_new('CompositorNodeComposite', 'Final Output')

    if not render_layers or not composite:
        print("  ⚠️ Vital compositor nodes missing. Direct output active.")
        return

    # --- 📸 REPLICATING iPHONE 16 REALISM (OPTIONAL ENHANCEMENTS) ---
    def safe_set(obj, attr, val):
        try: setattr(obj, attr, val)
        except: pass

    def safe_set_input(node, name_or_idx, val):
        try:
            if isinstance(name_or_idx, str):
                node.inputs[name_or_idx].default_value = val
            else:
                node.inputs[name_or_idx].default_value = val
        except: pass

    # 1. Distort (Lens Imperfections)
    distort = safe_node_new('CompositorNodeLensdist', 'Lens Imperfections')
    if distort:
        safe_set_input(distort, 'Distort', 0.005)
        safe_set_input(distort, 'Dispersion', 0.01)

    # 2. Gaussian Blur (Sensor Softness)
    # Replicates the slight lack of "infinite sharpness" in smartphone optics
    blur = safe_node_new('CompositorNodeBlur', 'Sensor Softness')
    if blur:
        safe_set(blur, 'filter_type', 'GAUSSIAN')
        safe_set_input(blur, 'Size X', 1.0)
        safe_set_input(blur, 'Size Y', 1.0)

    # 3. Glare (Bloom)
    glare = safe_node_new('CompositorNodeGlare', 'Lens Bloom')
    if glare:
        safe_set(glare, 'glare_type', 'FOG_GLOW')
        safe_set(glare, 'glare_quality', 'HIGH')

    # 4. Grain (Digital Sensor Noise)
    # High ISO simulation for smartphone sensors
    noise_mix = safe_node_new('CompositorNodeMixRGB', 'Sensor Grain Mix')
    noise_tex = None
    if noise_mix:
        safe_set(noise_mix, 'blend_type', 'OVERLAY')
        safe_set_input(noise_mix, 'Fac', 0.08) # Slightly more visible for vectors
        try:
            noise_tex = nodes.new(type='CompositorNodeTexture')
            if "SensorNoise" not in bpy.data.textures:
                tex = bpy.data.textures.new("SensorNoise", type='NOISE')
                if hasattr(tex, 'noise_scale'): tex.noise_scale = 0.002 # Finer grain
            noise_tex.texture = bpy.data.textures["SensorNoise"]
        except: pass

    # --- Linking Logic (CHAIN: Render -> Distort -> Blur -> Glare -> Grain -> Composite) ---
    try:
        last_out = render_layers.outputs[0]
        
        if distort:
            links.new(last_out, distort.inputs[0])
            last_out = distort.outputs[0]
        
        if blur:
            links.new(last_out, blur.inputs[0])
            last_out = blur.outputs[0]
            
        if glare:
            links.new(last_out, glare.inputs[0])
            last_out = glare.outputs[0]
            
        if noise_mix:
            links.new(last_out, noise_mix.inputs[1])
            if noise_tex:
                links.new(noise_tex.outputs[0], noise_mix.inputs[2])
            last_out = noise_mix.outputs[0]
            
        links.new(last_out, composite.inputs[0])
    except Exception as e:
        print(f"  ⚠️ Logic Error linking nodes: {e}")
        # Emergency bypass
        try: links.new(render_layers.outputs[0], composite.inputs[0])
        except: pass
        if viewer:
            try: links.new(last_out, viewer.inputs[0])
            except: pass
            
        print("  ✅ Compositor graph built successfully.")
        
    except Exception as e:
        print(f"  ⚠️ Error in compositor wiring: {e}")
        try: links.new(render_layers.outputs[0], composite.inputs[0])
        except: pass


def setup_world_hdri(assets_dir):
    """Sets a random HDRI from assets_dir/hdri as the world background."""
    hdri_dir = os.path.join(assets_dir, "hdri")
    if not os.path.exists(hdri_dir):
        # Fallback to backgrounds folder
        hdri_dir = os.path.join(assets_dir, "backgrounds")
        if not os.path.exists(hdri_dir):
            return

    hdri_files = [f for f in os.listdir(hdri_dir) if f.lower().endswith(('.hdr', '.exr'))]
    if not hdri_files:
        return

    hdri_path = os.path.join(hdri_dir, random.choice(hdri_files))
    print(f"🌍 Applying HDRI Background: {hdri_path}")

    # Configure World Nodes
    if not bpy.context.scene.world:
        bpy.context.scene.world = bpy.data.worlds.new("World")
    
    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    node_out = nodes.new(type='ShaderNodeOutputWorld')
    node_bg = nodes.new(type='ShaderNodeBackground')
    node_env = nodes.new(type='ShaderNodeTexEnvironment')
    node_mapping = nodes.new(type='ShaderNodeMapping')
    node_coord = nodes.new(type='ShaderNodeTexCoord')

    try:
        node_env.image = bpy.data.images.load(hdri_path)
    except Exception as e:
        print(f"  ❌ Error loading HDRI: {e}")
        return

    links.new(node_coord.outputs['Generated'], node_mapping.inputs['Vector'])
    links.new(node_mapping.outputs['Vector'], node_env.inputs['Vector'])
    links.new(node_env.outputs['Color'], node_bg.inputs['Color'])
    links.new(node_bg.outputs['Background'], node_out.inputs['Surface'])

    # Random Z Rotation and Intensity - Balanced for stability
    node_mapping.inputs['Rotation'].default_value[2] = random.uniform(0, 6.28)
    node_bg.inputs['Strength'].default_value = random.uniform(1.2, 1.8) # Boosted range
    print(f"  ✨ HDRI Strength: {node_bg.inputs['Strength'].default_value:.2f}")


def setup_ground_texture(ground_obj, assets_dir):
    """
    Sets the ground texture following the 50/25/25 rule:
    - 50%: Fixed background (fondo 50 x 50.jpg)
    - 25%: Color-jittered version of fixed background
    - 25%: Realistic textures from dynamic_pool
    """
    roll = random.random()
    
    fixed_bg_path = os.path.join(assets_dir, "backgrounds", "fondo 50 x 50.jpg")
    dynamic_pool_dir = os.path.join(assets_dir, "backgrounds", "dynamic_pool")
    
    mode = "FIXED"
    bg_path = fixed_bg_path
    
    if roll < 0.50:
        mode = "FIXED"
        bg_path = fixed_bg_path
    elif roll < 0.75:
        mode = "JITTER"
        bg_path = fixed_bg_path
    else:
        mode = "DYNAMIC"
        if os.path.exists(dynamic_pool_dir):
            bg_files = [f for f in os.listdir(dynamic_pool_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]
            if bg_files:
                bg_path = os.path.join(dynamic_pool_dir, random.choice(bg_files))
            else:
                # Fallback to fixed
                bg_path = fixed_bg_path
        else:
            bg_path = fixed_bg_path

    if not os.path.exists(bg_path):
        print(f"⚠️ Background path missing: {bg_path}. Falling back to grey.")
        # Fallback to simple color if file is missing
        if ground_obj.data.materials:
            mat = ground_obj.data.materials[0]
            mat.use_nodes = True
            bsdf = next((n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
            if bsdf: bsdf.inputs['Base Color'].default_value = (0.5, 0.5, 0.5, 1)
        return

    print(f"🖼️ Applying Ground Texture [{mode}]: {os.path.basename(bg_path)}")

    # Get or Create Material
    if not ground_obj.data.materials:
        mat = bpy.data.materials.new(name="GroundMaterial")
        ground_obj.data.materials.append(mat)
    else:
        mat = ground_obj.data.materials[0]

    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    node_out = nodes.new(type='ShaderNodeOutputMaterial')
    node_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    node_tex = nodes.new(type='ShaderNodeTexImage')
    node_mapping = nodes.new(type='ShaderNodeMapping')
    node_coord = nodes.new(type='ShaderNodeTexCoord')

    try:
        # Load or use existing image data
        img_name = os.path.basename(bg_path)
        if img_name in bpy.data.images:
            node_tex.image = bpy.data.images[img_name]
        else:
            node_tex.image = bpy.data.images.load(bg_path)
    except Exception as e:
        print(f"  ❌ Error loading ground texture: {e}")
        return

    links.new(node_coord.outputs['UV'], node_mapping.inputs['Vector'])
    links.new(node_mapping.outputs['Vector'], node_tex.inputs['Vector'])
    
    # --- Color Jitter Logic ---
    if mode == "JITTER":
        # Add Hue/Saturation/Value node for color randomization
        node_hsv = nodes.new(type='ShaderNodeHueSaturation')
        node_hsv.inputs['Hue'].default_value = random.uniform(0.45, 0.55)
        node_hsv.inputs['Saturation'].default_value = random.uniform(0.8, 1.2)
        node_hsv.inputs['Value'].default_value = random.uniform(0.8, 1.2)
        
        links.new(node_tex.outputs['Color'], node_hsv.inputs['Color'])
        links.new(node_hsv.outputs['Color'], node_bsdf.inputs['Base Color'])
    else:
        links.new(node_tex.outputs['Color'], node_bsdf.inputs['Base Color'])

    links.new(node_bsdf.outputs['BSDF'], node_out.inputs['Surface'])

    # Randomized Rotation and Roughness
    node_mapping.inputs['Rotation'].default_value[2] = random.uniform(0, 6.28)
    node_bsdf.inputs['Roughness'].default_value = random.uniform(0.6, 1.0)
    node_bsdf.inputs['Specular IOR Level'].default_value = 0.2 if mode == "DYNAMIC" else 0.5
    
def _apply_bevel_to_mesh(obj):
    """Add a bevel modifier to all meshes in hierarchy for realistic edge specular highlights.
    
    This makes LEGO stud edges and part boundaries catch light and cast micro-shadows,
    which dramatically helps distinguish detail in high-resolution renders.
    """
    def bevel_object(o):
        if o.type != 'MESH':
            for child in o.children:
                bevel_object(child)
            return
        
        # Avoid adding duplicate modifiers if already beveled
        if any(m.name == "LEGO_Bevel" for m in o.modifiers):
            for child in o.children:
                bevel_object(child)
            return
        
        try:
            bevel = o.modifiers.new(name="LEGO_Bevel", type='BEVEL')
            bevel.width = 0.0004       # 0.4mm — Increased for better visibility (stud edges)
            bevel.segments = 2         # 2 segments = smooth highlight, not faceted
            bevel.limit_method = 'ANGLE'  # Only bevel clean edges, protect complex geometry
            bevel.angle_limit = math.radians(35) # ~35°
            bevel.use_clamp_overlap = True
        except Exception as e:
            print(f"  ⚠️ Bevel modifier error on {o.name}: {e}")
        
        for child in o.children:
            bevel_object(child)
    
    bevel_object(obj)
    print(f"  🔷 Bevel modifiers applied to: {obj.name}")


def _apply_lego_material(obj, color_id=None):
    """Override all materials on obj (and children) with official LEGO ABS plastic."""
    if color_id is None:
        return
    
    try:
        # Use our new blender-specific color module
        import lego_colors_blender
        rgba = lego_colors_blender.get_blender_rgba(color_id)
        if hasattr(lego_colors_blender, 'LEGO_COLORS_HEX'):
             print(f"  🎨 Color mapped: {color_id} -> {lego_colors_blender.LEGO_COLORS_HEX.get(int(color_id), 'Default')}")
    except Exception as e:
        print(f"  ⚠️ Failed to use lego_colors_blender ({e}), using absolute default.")
        rgba = (0.8, 0.8, 0.8, 1.0)
    
    def apply_to_object(o):
        if o.type == 'MESH' and o.data:
            mat_name = f"LEGO_Color_{color_id}"
            mat = bpy.data.materials.get(mat_name)
            if not mat:
                mat = bpy.data.materials.new(name=mat_name)
                mat.use_nodes = True
                
                # Find the Principled BSDF node by type
                bsdf = None
                for n in mat.node_tree.nodes:
                    if n.type == 'BSDF_PRINCIPLED':
                        bsdf = n
                        break
                
                if bsdf:
                    # DISCONNECT any existing links to Base Color to ensure our value is used
                    base_color_input = bsdf.inputs.get("Base Color")
                    if base_color_input:
                        for link in list(mat.node_tree.links):
                            if link.to_socket == base_color_input:
                                mat.node_tree.links.remove(link)
                        base_color_input.default_value = rgba
                    
                    bsdf.inputs["Roughness"].default_value = 0.20  # More realistic, less wash-out
                    bsdf.inputs["IOR"].default_value = 1.45
                    
                    # Subsurface logic
                    if "Subsurface Weight" in bsdf.inputs:
                        bsdf.inputs["Subsurface Weight"].default_value = 0.01  # Reduced "waxiness"
                    elif "Subsurface" in bsdf.inputs:
                        bsdf.inputs["Subsurface"].default_value = 0.01
                        
                    # Clearcoat (Specular highlights on studs)
                    if "Coat Weight" in bsdf.inputs:
                        bsdf.inputs["Coat Weight"].default_value = 0.15
                        bsdf.inputs["Coat Roughness"].default_value = 0.03
                    elif "Clearcoat" in bsdf.inputs:
                        bsdf.inputs["Clearcoat"].default_value = 0.15
                        bsdf.inputs["Clearcoat Roughness"].default_value = 0.03
            
            # Apply material to all existing slots to ensure full coverage
            if not o.data.materials:
                o.data.materials.append(mat)
            else:
                for i in range(len(o.data.materials)):
                    o.data.materials[i] = mat
        
        for child in o.children:
            apply_to_object(child)
    
    apply_to_object(obj)


def import_ldraw_part(filepath, ldraw_lib=None, color_id=None):
    """Import LDraw part using ImportLDraw addon. Optionally applies official LEGO color."""
    try:
        # Explicitly pass the library path to the operator
        # Deselect all to ensure we only get the new object
        bpy.ops.object.select_all(action='DESELECT')

        import_kwargs = dict(filepath=filepath, addEnvironment=False, positionCamera=False)
        if ldraw_lib:
            import_kwargs['ldrawPath'] = ldraw_lib
        
        # NOTE: 'defaultColour' is NOT a valid kwarg in this addon version. 
        # We rely on _apply_lego_material() instead.
        
        bpy.ops.import_scene.importldraw(**import_kwargs)
        
        # --- CRITICAL FIX: Addon forces 400 samples on import. Override it back to 64. ---
        try:
            bpy.context.scene.cycles.samples = 64
            bpy.context.scene.cycles.diffuse_bounces = 1
            bpy.context.scene.cycles.glossy_bounces = 1
        except: pass

        selected = bpy.context.selected_objects
        if not selected:
             return None
             
        # Ideally, we want the parent object if it exists
        obj = selected[0]
        while obj.parent:
            obj = obj.parent
        
        # Apply official LEGO material override
        if color_id is not None:
            _apply_lego_material(obj, color_id)
            _apply_bevel_to_mesh(obj)  # Edge highlights for studs and borders
            
        return obj
    except Exception as e:
        print(f"Failed to import {filepath}: {e}")
        return None

def get_hierarchy_corners(obj):
    """Recursively find all bounding box corners in an object hierarchy in world space."""
    corners = []
    if obj.type == 'MESH':
        mw = obj.matrix_world
        corners.extend([mw @ Vector(corner) for corner in obj.bound_box])
    for child in obj.children:
        corners.extend(get_hierarchy_corners(child))
    return corners

def copy_hierarchy(obj, parent=None):
    """Recursively copy an object and its children hierarchy."""
    new_obj = obj.copy()
    if obj.data:
        new_obj.data = obj.data.copy()
    
    if parent:
        new_obj.parent = parent
    
    bpy.context.collection.objects.link(new_obj)
    
    # Copy custom properties (class_id, ldraw_id, etc.)
    for key, value in obj.items():
        new_obj[key] = value
    
    for child in obj.children:
        copy_hierarchy(child, parent=new_obj)
    
    return new_obj

def get_hierarchy_vertices(obj):
    """Recursively find all vertices in an object hierarchy in world space."""
    verts = []
    if obj.type == 'MESH':
        mw = obj.matrix_world
        verts.extend([mw @ v.co for v in obj.data.vertices])
    for child in obj.children:
        verts.extend(get_hierarchy_vertices(child))
    return verts

def get_geometry_aabb(obj):
    """Calculates the world-space AABB of a hierarchy. 
    Returns: (center_vector, max_radius_from_center)
    """
    verts = get_hierarchy_vertices(obj)
    if not verts:
        return Vector((0,0,0)), 0.05
    
    xs = [v.x for v in verts]
    ys = [v.y for v in verts]
    zs = [v.z for v in verts]
    
    min_v = Vector((min(xs), min(ys), min(zs)))
    max_v = Vector((max(xs), max(ys), max(zs)))
    
    center = (min_v + max_v) / 2.0
    # Radius is the distance from center to the furthest corner of AABB
    # or max distance to any vertex for better fit.
    max_d = max([(v - center).length for v in verts])
    
    return center, max_d

def get_convex_hull(points):
    """Computes the convex hull of a set of 2D points using Graham scan."""
    points = sorted(set((round(p[0], 6), round(p[1], 6)) for p in points))
    if len(points) <= 1:
        return points
    
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
        
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
        
    return lower[:-1] + upper[:-1]

def get_mabr(hull):
    """Computes the Minimum Area Bounding Rectangle (OBB) for a convex hull."""
    min_area = float('inf')
    best_rect = None
    n = len(hull)
    if n < 3:
        # Fallback for degenerate flat hulls (just make a straight box)
        if n == 0: return None
        xs = [p[0] for p in hull]
        ys = [p[1] for p in hull]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        return [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
        
    for i in range(n):
        p1 = hull[i]
        p2 = hull[(i+1)%n]
        
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = math.hypot(dx, dy)
        if length == 0: continue
        
        ux, uy = dx / length, dy / length
        vx, vy = -uy, ux
        
        min_u, max_u = float('inf'), float('-inf')
        min_v, max_v = float('inf'), float('-inf')
        
        for p in hull:
            u = p[0] * ux + p[1] * uy
            v = p[0] * vx + p[1] * vy
            if u < min_u: min_u = u
            if u > max_u: max_u = u
            if v < min_v: min_v = v
            if v > max_v: max_v = v
            
        area = (max_u - min_u) * (max_v - min_v)
        if area < min_area:
            min_area = area
            c1 = (min_u * ux + min_v * vx, min_u * uy + min_v * vy)
            c2 = (max_u * ux + min_v * vx, max_u * uy + min_v * vy)
            c3 = (max_u * ux + max_v * vx, max_u * uy + max_v * vy)
            c4 = (min_u * ux + max_v * vx, min_u * uy + max_v * vy)
            best_rect = [c1, c2, c3, c4]
            
    return best_rect

# Import LDrawResolver from local module
try:
    from ldraw_resolver import LDrawResolver
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════
# YOLO LABELING HELPERS
# ═══════════════════════════════════════════════════════════

def generate_yolo_bbox_label(obj, scene, class_id):
    """Generate a YOLO bounding box label for a single object.
    Returns: string 'class_id cx cy w h' or None if object is not visible.
    """
    verts_3d = get_hierarchy_vertices(obj)
    if not verts_3d:
        return None
    coords_2d = [world_to_camera_view(scene, scene.camera, v) for v in verts_3d]
    x_vals = [c.x for c in coords_2d if c.z > 0]
    y_vals = [c.y for c in coords_2d if c.z > 0]
    if not x_vals or not y_vals:
        return None
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)
    x_center = (x_min + x_max) / 2.0
    y_center = 1.0 - ((y_min + y_max) / 2.0)
    w_box = x_max - x_min
    h_box = y_max - y_min
    return f"{class_id} {x_center:.6f} {y_center:.6f} {w_box:.6f} {h_box:.6f}"


def generate_yolo_seg_label(obj, scene):
    """Generate a YOLO segmentation polygon label for a single object.
    Returns: string 'class_id x1 y1 x2 y2 ...' or None if object is not visible.
    """
    camera = scene.camera
    verts_3d = get_hierarchy_vertices(obj)
    if not verts_3d:
        return None
    coords_2d = [world_to_camera_view(scene, camera, coord) for coord in verts_3d]
    visible_points = []
    for c in coords_2d:
        if c.z > 0:
            cx = max(0.0, min(1.0, c.x))
            cy = max(0.0, min(1.0, c.y))
            visible_points.append((cx, cy))
    if not visible_points:
        return None
    hull = get_convex_hull(visible_points)
    if not hull or len(hull) < 3:
        return None
    # Check minimum size to filter microscopic noise
    xs = [p[0] for p in hull]
    ys = [p[1] for p in hull]
    if (max(xs) - min(xs)) < 0.0005 or (max(ys) - min(ys)) < 0.0005:
        return None
    # Build YOLO-Seg format: class x1 y1 x2 y2 ... xn yn
    if 'class_id' not in obj:
        return None
    pts = []
    for corner in hull:
        px = max(0.0, min(1.0, corner[0]))
        py = max(0.0, min(1.0, 1.0 - corner[1]))  # Invert Y for YOLO
        pts.extend([f"{px:.6f}", f"{py:.6f}"])
    return f"{obj['class_id']} {' '.join(pts)}"


def write_image_meta(meta_path, img_prefix, ids, color_ids, color_names):
    """Append one line of metadata to the JSONL file."""
    with open(meta_path, 'a') as mf:
        mf.write(json.dumps({
            "img": f"{img_prefix}.jpg",
            "ids": ids,
            "color_ids": color_ids,
            "color_names": color_names
        }) + "\n")


# ═══════════════════════════════════════════════════════════
# GEOMETRY HELPERS (Formerly inside main())
# ═══════════════════════════════════════════════════════════

def get_stable_orientations(obj):
    """Analyzes piece geometry using its Convex Hull to find
    all stable resting positions on a flat surface.
    Returns a list of Euler rotations, one per stable face.
    """
    import bmesh
    from mathutils import Matrix

    bpy.context.view_layer.update()
    me = obj.to_mesh()
    bm = bmesh.new()
    bm.from_mesh(me)

    bmesh.ops.convex_hull(bm, input=bm.verts)

    stable_faces = []
    com = Vector((0, 0, 0))
    for v in bm.verts:
        com += v.co
    com /= len(bm.verts)

    for face in bm.faces:
        if face.calc_area() < 1e-6:
            continue
        normal = face.normal.copy()
        center = face.calc_center_median()
        target_down = Vector((0, 0, -1))

        # Project CoM onto the face plane
        rel_com = com - center
        dist_to_plane = rel_com.dot(normal)
        projected_com = com - (dist_to_plane * normal)

        # Check if projected CoM is inside the face polygon
        is_inside = True
        min_dist_to_edge = float('inf')
        
        for edge in face.edges:
            v1 = edge.verts[0].co
            v2 = edge.verts[1].co
            edge_vec = v2 - v1
            edge_len = edge_vec.length
            
            if edge_len < 1e-6:
                continue
                
            to_com = projected_com - v1
            cross_vec = edge_vec.cross(to_com)
            side = cross_vec.dot(normal)
            
            dist = cross_vec.length / edge_len
            
            if side < -1e-6:
                is_inside = False
                break
                
            if dist < min_dist_to_edge:
                min_dist_to_edge = dist

        # Apply Real-World Physics Filter
        if is_inside:
            h_com = abs(dist_to_plane)
            area = face.calc_area()
            
            # Tipping Angle: How far you have to tilt the piece before the CoM 
            # crosses the edge and it falls over.
            tipping_angle_rad = math.atan2(min_dist_to_edge, h_com) if h_com > 1e-6 else math.pi/2
            tipping_angle_deg = math.degrees(tipping_angle_rad)
            
            # Realistic thresholds to survive micro-vibrations:
            # 1. Area > 1e-6 (reject microscopic spikes/corners)
            # 2. Tipping Angle > 2.0 degrees (reject standing a piece on a tiny edge)
            if area > 1e-6 and tipping_angle_deg > 2.0:
                euler_rot = normal.rotation_difference(target_down).to_euler()
                stable_faces.append(euler_rot)

    # Deduplicate similar orientations (symmetries)
    distinct_faces = []
    for rot in stable_faces:
        is_new = True
        for existing in distinct_faces:
            diff = sum(abs(a - b) for a, b in zip(rot, existing))
            if diff < 0.1:
                is_new = False
                break
        if is_new:
            distinct_faces.append(rot)

    bm.free()
    obj.to_mesh_clear()
    return distinct_faces


def apply_bevel_lod(obj, render_res=800, camera_fov_m=0.20):
    """Reduce bevel segments on small objects to speed up rendering."""
    try:
        bpy.context.view_layer.update()
        dims = obj.dimensions
        screen_fraction = max(dims.x, dims.y) / camera_fov_m
        pixel_size = screen_fraction * render_res
        for mod in obj.modifiers:
            if mod.type == 'BEVEL':
                if pixel_size < 80:
                    mod.segments = 1
                elif pixel_size < 150:
                    mod.segments = 2
        for child in obj.children:
            apply_bevel_lod(child, render_res, camera_fov_m)
    except Exception as e:
        print(f"  ⚠️ Bevel LOD skipped for {obj.name}: {e}")


def get_max_xy_radius(obj):
    """Get the max dimension in XY plane to prevent clipping."""
    corners = get_hierarchy_corners(obj)
    if not corners:
        return 0.02
    max_r = 0
    for c in corners:
        r = math.sqrt(c.x**2 + c.y**2)
        if r > max_r:
            max_r = r
    return max_r


def select_hierarchy(obj):
    """Select an object and all its children recursively."""
    obj.select_set(True)
    for c in obj.children:
        select_hierarchy(c)


def main():
    # Get arguments passed after "--"
    argv = sys.argv
    try:
        idx = argv.index("--")
        data_file = argv[idx + 1]
    except (ValueError, IndexError):
        print("No data file passed to Blender script.")
        return

    with open(data_file, 'r') as f:
        data = json.load(f)

    set_id = data.get('set_id', 'unknown')
    
    # In Universal Detection Plan B, we receive 'pieces_config'
    # Fallback to 'parts' for backward compatibility
    render_mode = data.get('render_mode', 'images_mix')
    PARTS_PER_IMAGE = data.get('parts_per_image', 20)
    pieces_config = data.get('pieces_config', [])
    if not pieces_config:
        parts = data.get('parts', [])
        global_render_engine = data.get('render_engine', 'CYCLES')
        res_x = data.get('resolution_x', 640)
        for p in parts:
            pieces_config.append({
                'part': p, 'tier': 'UNKNOWN', 'imgs': data.get('num_images', 30),
                'engine': global_render_engine, 'res': res_x
            })

    # Derive top-level 'parts' and 'num_images' for other script sections
    parts = [pc['part'] for pc in pieces_config]
    
    # For Strategy C workers, num_images per piece might vary, but for the drop loop
    # we take the max images requested by any piece in this worker's chunk.
    num_images = data.get('num_images')
    if num_images is None:
        num_images = max([pc.get('imgs', 30) for pc in pieces_config]) if pieces_config else 30

    output_base = data.get('output_base', '/content/dataset')
    assets_dir = data.get('assets_dir')
    ldraw_path_base = data.get('ldraw_path') 
    addon_path = data.get('addon_path') # Path to ImportLDraw folder parent

    print(f"🚀 Blender started for Set {set_id}")
    print(f"📍 Output path: {output_base}")

    # Register Addon from dynamic path or local script dir
    register_ldraw_addon(data.get('addon_path'))

    # Initialize variables to avoid NameError if logic is skipped
    unique_meshes = []
    physics_objects = []
    total_spawned = 0

    # Validate and Auto-Fix LDraw Path
    if ldraw_path_base:
        # Check root
        p_root = os.path.join(ldraw_path_base, "p")
        parts_root = os.path.join(ldraw_path_base, "parts")
        
        if not (os.path.isdir(p_root) and os.path.isdir(parts_root)):
            # Try to find a subfolder that contains them (case insensitive ldraw/LDraw)
            found_inner = False
            for d in os.listdir(ldraw_path_base):
                inner_path = os.path.join(ldraw_path_base, d)
                if os.path.isdir(inner_path):
                    if os.path.isdir(os.path.join(inner_path, "p")) and os.path.isdir(os.path.join(inner_path, "parts")):
                        ldraw_path_base = inner_path
                        found_inner = True
                        print(f"📂 Auto-detected LDraw subfolder: {ldraw_path_base}")
                        break
            
            if not found_inner:
                msg = f"❌ CRITICAL: LDraw path '{ldraw_path_base}' is invalid (missing 'p' or 'parts' subfolders)."
                print(msg)
                raise Exception(msg)
    else:
        print("⚠️ No LDraw path provided. Expecting global library setup.")

    # Apply to Addon Preferences
    if ldraw_path_base:
        try:
            bpy.context.preferences.addons['import_scene_importldraw'].preferences.ldrawPath = ldraw_path_base
        except Exception:
            try:
                bpy.context.preferences.addons['ImportLDraw'].preferences.ldrawPath = ldraw_path_base
            except Exception:
                pass

    os.makedirs(output_base, exist_ok=True)
    images_dir = os.path.join(output_base, "images")
    labels_dir = os.path.join(output_base, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    clean_scene()
    
    global_render_engine = data.get('render_engine', 'CYCLES')

    setup_compositor()  # Sensor noise + bloom glare post-processing
    cam = setup_camera()
    setup_lighting()
    setup_world_hdri(assets_dir)

    # Create Ground Plane (50x50cm training surface)
    bpy.ops.mesh.primitive_plane_add(size=0.5, location=(0, 0, 0))
    ground = bpy.context.object
    
    # Configure Rigid Body World for high accuracy with small LEGO parts
    if not bpy.context.scene.rigidbody_world:
        bpy.ops.rigidbody.world_add()
    
    bpy.context.scene.rigidbody_world.substeps_per_frame = 200  # Ultra-precision for small parts
    bpy.context.scene.rigidbody_world.solver_iterations = 60   # Ensures resting contact stability
    
    bpy.ops.rigidbody.object_add()
    ground.rigid_body.type = 'PASSIVE'
    ground.rigid_body.friction = 0.9
    ground.rigid_body.collision_shape = 'MESH'        # FIX: Exact surface collision (not bounding box)
    ground.rigid_body.use_margin = True
    ground.rigid_body.collision_margin = 0.0           # FIX: ZERO margin = no invisible air cushion
    ground.rigid_body.restitution = 0.05               # Almost no bounce
    
    # --- Ambient Occlusion: Darken contact points ---
    if hasattr(bpy.context.scene, 'eevee'):
        if hasattr(bpy.context.scene.eevee, 'use_gtao'):
            bpy.context.scene.eevee.use_gtao = True
            bpy.context.scene.eevee.gtao_distance = 0.02    # 2cm radius — tight contact darkening
            bpy.context.scene.eevee.gtao_factor = 1.5        # Slightly amplified for visibility
    # For CYCLES, AO is built-in. We boost it via world settings if possible.
    try:
        bpy.context.scene.world.light_settings.use_ambient_occlusion = True
        bpy.context.scene.world.light_settings.ao_factor = 0.5
        bpy.context.scene.world.light_settings.distance = 0.02
    except: pass
    
    # 🌑 Setup Ground Texture (Scanned Textures)
    setup_ground_texture(ground, assets_dir)
    
    # Ensure it maps correctly (zenithal cam + 20x20cm plane = perfect fit)

    # --- CONFIGURATION FOR 20x20cm ZONE (iPhone 16 @ 24MP) ---
    # Lens = (36 × 0.70) / 0.20 = 126mm → covers exactly 20x20cm at 70cm
    
    # Configuration for pieces is already extracted at the top of main()
    
    # 1. Setup Camera (Fixed Zenithal Centered - iPhone 16 Calibration)
    # Calibrated to 26mm @ 0.70m height
    cam.location = (0, 0, 0.70) 
    cam.rotation_euler = (0, 0, 0) 
    cam.data.sensor_width = 36.0 # Full-frame equivalent sensor
    cam.data.lens = 26.0 # 26mm focal length (iPhone 16)

    # Initialize Resolver
    resolver = LDrawResolver(ldraw_path_base)
    print(f"🔧 Render mode: {render_mode} | Parts per image: {PARTS_PER_IMAGE}")    


    # --- GLOBAL RENDER SETUP ---
    # Calculate final resolution and call setup_render_engine ONCE
    first_pc = pieces_config[0] if pieces_config else {'engine': 'CYCLES', 'res': 800}
    if render_mode == 'ref_pieza' and 'res' in first_pc:
        res_x = first_pc['res']
    else:
        res_x = data.get('resolution_x', first_pc.get('res', 1920))
    is_square = data.get('square', False)
    res_y = res_x if (is_square or render_mode != 'images_mix') else int(res_x * 0.75)
    setup_render_engine(engine=first_pc.get('engine', global_render_engine), resolution=(res_x, res_y))
    
    for i, pc in enumerate(pieces_config):
        part = pc['part']
        ldraw_id = part['ldraw_id']
        color_id = part.get('color_id', None)
        color_name = part.get('color_name', None)
        num_images = pc.get('imgs', 30)
        engine = pc.get('engine', 'CYCLES')
        res = pc.get('res', 800)
        
        color_info = f" | Color: {color_id} ({color_name or 'n/a'})" if color_id is not None else ""
        print(f"\n🚀 Loading part {ldraw_id}{color_info} | Tier: {pc.get('tier', 'REF')}")
        
        # Strategy C: Universal Detector mode forces all class IDs to 0
        if os.environ.get("UNIVERSAL_DETECTOR", "0") == "1":
            class_id = 0
            if i == 0: print("🛠️ Universal Detector Mode: All pieces will be labeled as class 0.")
        else:
            class_id = i
        
        part_path = resolver.find_part(ldraw_id)
        
        # Minifig Special Handling
        mf_components = pc.get('minifig_components') or part.get('minifig_components')
        if mf_components:
            print(f"🧩 Assembling Minifig: {ldraw_id}")
            import assemble_minifig
            minifig_root = assemble_minifig.assemble_minifig(
                mf_components, 
                Path(ldraw_path_base), 
                addon_path
            )
            if minifig_root:
                unique_meshes.append({'obj': minifig_root, 'id': class_id, 'color_id': color_id, 'color_name': color_name, 'ldraw_id': ldraw_id})
                minifig_root.location = (0, 0, -10)
        elif part_path:
            obj = import_ldraw_part(part_path, ldraw_path_base, color_id=color_id)
            if obj:
                print(f"PROGRESS: Loading templates {i+1}/{len(pieces_config)}")
                print(f"  📥 Loaded template: {ldraw_id}")
                set_origin_to_center(obj) # CRITICAL: Rotate around center, not LDraw anchor
                apply_bevel_lod(obj, render_res=first_pc.get('res', 800))
                unique_meshes.append({'obj': obj, 'id': class_id, 'color_id': color_id, 'color_name': color_name, 'ldraw_id': ldraw_id})
                obj.location = (0, 0, -10) # move template out of sight
            else:
                print(f"  ⚠️ Failed to load template: {ldraw_id}")
        else:
            print(f"  ⚠️ Part path not found for: {ldraw_id}")
            continue
            

    # ═══════════════════════════════════════════════════════════
    # MODE: ref_pieza — Multi-Worker Geometric Strategy
    # ═══════════════════════════════════════════════════════════
    if render_mode == 'ref_pieza':
        if not unique_meshes:
            print("❌ No meshes loaded for ref_pieza mode.")
            return

        template = unique_meshes[0]
        template_obj = template['obj']
        ldraw_id_ref = template['ldraw_id']
        
        # --- ANALYSIS PHASE ---
        if data.get('is_analyze_only'):
            print(f"🔍 Analyzing stable faces for {ldraw_id_ref}...")
            orientations = get_stable_orientations(template_obj)
            
            analysis_result = {
                'ldraw_id': ldraw_id_ref,
                'stable_faces_count': len(orientations),
                'orientations': [[r.x, r.y, r.z] for r in orientations]
            }
            
            # Write results back for the orchestrator
            # Use the same data-path but .result extension
            result_path = data_file.replace('.json', '.result')
            with open(result_path, 'w') as rf:
                json.dump(analysis_result, rf, indent=4)
            print(f"✅ Analysis complete: {len(orientations)} stable faces found.")
            return

        # --- RENDER PHASE ---
        print("📸 MODE: ref_pieza — Geometric Stable Face Renders")
        
        # Place piece at center
        template_obj.location = (0, 0, 0)
        set_origin_to_center(template_obj)
        
        # Load the specific stable face for this worker
        # Or if not provided, fallback to standard random physics
        target_face_idx = data.get('stable_face_idx', -1)
        orientations = data.get('orientations', [])
        
        if target_face_idx >= 0 and target_face_idx < len(orientations):
            # ROTATION MODE: We render 24 images for THIS face
            rot_vals = orientations[target_face_idx]
            template_obj.rotation_euler = (rot_vals[0], rot_vals[1], rot_vals[2])
            bpy.context.view_layer.update()
            snap_to_ground([template_obj])
            
            # Render 24 photos (15 degree steps)
            num_rot_images = 24
            rots_per_face = num_rot_images
        else:
            # FALLBACK: Use random physics (original behavior)
            print("⚠️ No stable face provided, falling back to random physics...")
            num_rot_images = data.get('ref_num_images', 300)
            rots_per_face = 1 # We'll do random drops
            num_drops = num_rot_images
            
        # 1. Optics & Resolution — iPhone 16 spec: 26mm / fixed 70cm height
        # Note: lens stays at 26mm as set by setup_camera(). No override here.
        # The 26mm at 70cm gives ~63cm horizontal FOV, covering the full 50x50cm surface.
        scene = bpy.context.scene
        # Render at 768×768 so that the downstream crop to 384×384 (FAISS/DINOv2)
        # operates on full-quality data without any resolution penalty.
        scene.render.resolution_x = 768
        scene.render.resolution_y = 768
        scene.render.resolution_percentage = 100
        
        # 2. Adaptive High-Contrast Background
        # Comprehensive list of dark LEGO color IDs to force white background
        dark_ids = [
            0,   # Black
            1,   # Blue
            2,   # Green
            3,   # Dark Turquoise
            6,   # Brown
            8,   # Dark Gray
            22,  # Purple
            26,  # Magenta
            28,  # Dark Tan (can be dark depending on lighting)
            70,  # Reddish Brown
            72,  # Dark Stone Gray
            85,  # Dark Bluish Gray
            272, # Dark Blue
            288, # Dark Green
            320, # Dark Red
            484, # Dark Orange
        ]
        
        try:
            color_id_ref = int(template.get('color_id', -1))
        except:
            color_id_ref = -1
            
        color_name_ref = template.get('color_name') or ''
        
        # Detect if piece is dark or if name contains "Dark"
        is_dark_piece = (color_id_ref in dark_ids) or ("Dark" in color_name_ref) or ("Black" in color_name_ref)
        ground_obj = bpy.data.objects.get("Plane")
        
        # Ensure ground plane is exactly 50×50cm as per physical setup spec
        if ground_obj:
            ground_obj.scale = (1.0, 1.0, 1.0)
            ground_obj.dimensions = (0.5, 0.5, 0.0)  # 50×50cm surface
            ground_obj.location = (0.0, 0.0, 0.0)    # Centered at origin
        
        print(f"  🌌 GrabCut Alignment Mode: Piece {ldraw_id_ref} -> Pure Black (0,0,0) with NO ground shadows.")
        
        # Enable transparency so the background is cleanly separated from the piece
        scene.render.film_transparent = True
        
        # Re-configure the compositor specifically for ref_pieza to composite over pure black
        scene.use_nodes = True
        if hasattr(scene, "node_tree") and scene.node_tree:
            tree = scene.node_tree
            tree.nodes.clear()
            rlayers = tree.nodes.new('CompositorNodeRLayers')
            comp = tree.nodes.new('CompositorNodeComposite')
            alpha_over = tree.nodes.new('CompositorNodeAlphaOver')
            # Input 1 is the background (solid black)
            alpha_over.inputs[1].default_value = (0.0, 0.0, 0.0, 1.0)
            
            tree.links.new(rlayers.outputs['Image'], alpha_over.inputs[2])
            tree.links.new(alpha_over.outputs['Image'], comp.inputs['Image'])
            
        # Hide the ground completely. We don't want any ground shadows, since the
        # real inference pipeline (GrabCut) removes all shadows alongside the carpet.
        if ground_obj:
            ground_obj.hide_render = True
            
        # The World can still cast light (HDRI is active), but it won't be seen in the background
        # because film_transparent = True. This is exactly what we want: HDRI lighting on the piece,
        # perfectly isolated mask.

        
        pc = pieces_config[0] if pieces_config else {}
        piece_tier = pc.get('tier', 'REF').upper()
        if hasattr(scene, 'cycles'):
            scene.cycles.use_adaptive_sampling = True
            scene.cycles.adaptive_threshold = 0.05
            if piece_tier == 'TIER1' or piece_tier == 'REF':
                scene.cycles.samples = 32
                scene.cycles.max_bounces = 2
            elif piece_tier == 'TIER2':
                scene.cycles.samples = 64
                scene.cycles.max_bounces = 4
            else: 
                scene.cycles.samples = 128
                scene.cycles.max_bounces = 6
            print(f"  ⚡ TIER {piece_tier} Optimization: {scene.cycles.samples} samples active.")
            
        # --- EXECUTION LOOP ---
        img_count = 0
        offset_idx = data.get('offset_idx', 0)
        worker_id = data.get('worker_id', '')
        prefix_base = f"w{worker_id}_" if worker_id != '' else ""
        class_id = template.get('id', 0)
        num_images_final = num_rot_images # Total images for this worker

        # Physics Setup (if needed)
        SETTLE_FRAMES = 150
        if target_face_idx < 0:
            bpy.context.view_layer.objects.active = template_obj
            if not template_obj.rigid_body:
                bpy.ops.rigidbody.object_add()
            template_obj.rigid_body.type = 'ACTIVE'
            template_obj.rigid_body.collision_shape = 'CONVEX_HULL'

        def place_piece_on_ground(obj):
            """After any rotation, shift the object so its absolute bottom vertex
            sits exactly at Z=0 (ground surface). This works for any piece shape,
            any orientation, any scale."""
            bpy.context.view_layer.update()
            corners = get_hierarchy_corners(obj)
            if not corners:
                return
            min_z = min(c.z for c in corners)
            obj.location.z += (0.0001 - min_z)  # 0.1mm margin above ground
            bpy.context.view_layer.update()

        # Loop logic
        num_iters = 1 if target_face_idx >= 0 else num_drops
        
        for i_iter in range(num_iters):
            if target_face_idx < 0:
                scene.frame_set(1)
                bpy.ops.ptcache.free_bake_all()
                # Initial drop: piece above ground. Use bbox to know current height
                bpy.context.view_layer.update()
                corners_now = get_hierarchy_corners(template_obj)
                half_height = 0
                if corners_now:
                    zs_now = [c.z for c in corners_now]
                    half_height = (max(zs_now) - min(zs_now)) / 2.0
                template_obj.location = (0, 0, half_height + 0.03)  # 3cm above ground
                template_obj.rotation_euler = (random.random()*6.28, random.random()*6.28, random.random()*6.28)
                for f in range(1, SETTLE_FRAMES + 1):
                    scene.frame_set(f)
                place_piece_on_ground(template_obj)

            for r in range(rots_per_face):
                if target_face_idx >= 0:
                    rot_vals = orientations[target_face_idx]
                    template_obj.rotation_euler = (rot_vals[0], rot_vals[1], rot_vals[2])
                    template_obj.rotation_euler[2] += (r * (2.0 * math.pi / 24.0))
                else:
                    template_obj.rotation_euler[2] += (2.0 * math.pi / rots_per_face) + random.uniform(-0.05, 0.05)
                
                # ─── CRITICAL: Place piece on ground after EVERY rotation ────────
                # Each rotation changes which face is at the bottom, so the bounding
                # box bottom changes. We must re-snap every time.
                place_piece_on_ground(template_obj)
                # ────────────────────────────────────────────────────────────────
                
                bpy.context.view_layer.update()
                
                # ─── CAMERA: Option B — Dynamic height for training ──────────────
                # ALIGNMENT: We MUST use the true geometric AABB center to target the camera.
                # Object.location is the origin, which might not be centered after physical rotations.
                bpy.context.view_layer.update()
                
                # Get true visual center and radius from geometry vertices
                aabb_center, aabb_radius = get_geometry_aabb(template_obj)
                obj_dim = template_obj.dimensions  # world-space size
                
                # Piece top = top of the bounding box
                piece_top_z = aabb_center.z + (obj_dim.z / 2.0)
                piece_center_x = aabb_center.x
                piece_center_y = aabb_center.y
                radius = aabb_radius


                cam_obj = scene.camera
                
                # Remove any left-over constraints (HDRI setup may leave constraints)
                for c in list(cam_obj.constraints):
                    cam_obj.constraints.remove(c)
                
                focal_mm = 26.0
                sensor_half_mm = 18.0  # 36mm sensor / 2
                # ALIGNMENT: GoldenCrop scales to ~80% occupancy. 
                # radius = 40% of sensor half.
                fill_ratio = 0.40

                # Use at least 5mm radius so tiny pieces don't get extreme close-ups
                effective_radius = max(radius, 0.005)
                # target_dist is the optical distance needed to achieve 80% occupancy
                target_dist = (effective_radius * focal_mm) / (fill_ratio * sensor_half_mm)
                
                # CRITICAL ALIGNMENT: Camera MUST stay fixed at 0.70m physical height to 
                # preserve the exact perspective of the physical setup (iPhone 16 on mount).
                cam_z = 0.70
                real_dist = max(0.01, cam_z - piece_top_z)
                
                # We achieve the framing of target_dist by using digital zoom (cropping the sensor)
                # This mathematically emulates cropping a 24MP image perfectly.
                new_sensor_width = 36.0 * (target_dist / real_dist)
                
                cam_obj.location = (piece_center_x, piece_center_y, cam_z)
                cam_obj.data.lens = focal_mm
                cam_obj.data.sensor_width = new_sensor_width
                cam_obj.data.dof.focus_distance = real_dist
                
                # Cenital: rotation (0,0,0) looks straight down in Blender 5.0
                cam_obj.rotation_euler = (0, 0, 0)
                bpy.context.view_layer.update()
                print(f"  📷 Camera: FIXED Z=70cm | piece_top={piece_top_z*1000:.1f}mm | "
                      f"optical_zoom_dist={target_dist*100:.1f}cm | new_sensor_w={new_sensor_width:.2f}mm")
                # ─────────────────────────────────────────────────────────────────




                # Render
                img_idx = img_count + offset_idx
                img_prefix = f"{prefix_base}img_{img_idx:04d}"
                render_path = os.path.join(images_dir, f"{img_prefix}.jpg")
                
                # SMART RESUME: Skip existing images ONLY for images_mix (YOLO) scenes.
                # Reference renders (ref_pieza) are wiped at piece-level to ensure consistency.
                if "images_mix" in render_path and os.path.exists(render_path):
                    print(f"  ⏭️ Image {img_prefix}.jpg already exists, skipping.")
                    img_count += 1
                    continue
                    
                scene.render.filepath = render_path
                
                if img_count % 10 == 0:
                    print(f"PROGRESS: {img_count + 1}/{num_images_final}")
                # == FINAL DEBUG: state dump before render ==
                sc = bpy.context.scene
                print(f"  ▶ render img_{img_idx}: scene.camera={sc.camera.name if sc.camera else 'NONE'}"
                      f" cam_loc={sc.camera.location!r} cam_rot={sc.camera.rotation_euler!r}"
                      f" cam_lens={sc.camera.data.lens}mm"
                      f" film_transparent={sc.render.film_transparent}"
                      f" use_nodes={sc.use_nodes}")
                print(f"    template_obj={template_obj.name} loc={template_obj.location!r}"
                      f" hide={template_obj.hide_render}")
                all_objs = [(o.name[:20], o.type[:4], o.hide_render, tuple(round(x,3) for x in o.location))
                            for o in bpy.data.objects if o.type in ('MESH','CAMERA','LIGHT')]
                print(f"    scene_objects: {all_objs}")
                # == END DEBUG ==
                # Compositor is already cleanly configured to do Alpha Over Black for GrabCut alignment.
                # We do NOT bypass it here anymore.
                bpy.ops.render.render(write_still=True)





                
                # Labeling (using helper)
                label_path = os.path.join(labels_dir, f"{img_prefix}.txt")
                bbox_line = generate_yolo_bbox_label(template_obj, scene, class_id)
                if bbox_line:
                    with open(label_path, 'w') as lf:
                        lf.write(bbox_line + "\n")
                
                # Metadata
                meta_path = os.path.join(output_base, "image_meta.jsonl")
                write_image_meta(meta_path, img_prefix, [ldraw_id_ref], [color_id_ref], [color_name_ref])
                
                img_count += 1
                if img_count >= num_images_final: break
            if img_count >= num_images_final: break

        print(f"✅ ref_pieza complete: {img_count} images rendered.")
        return

    # --- END REF_PIEZA ---
    
    # VALIDATION: Check if objects were imported
    if len(unique_meshes) == 0:
        msg = "❌ CRITICAL ERROR: No LEGO pieces were imported or spawned into the scene. Aborting render."
        print(msg)
        raise Exception(msg)
    
    if render_mode == 'images_mix':
        print(f"🎲 MODE: images_mix — Ready to dynamically spawn from {len(unique_meshes)} piece types per drop.")

    # Hide templates again or delete?
    # Keeping them far away is fine.

    # Save missing parts report
    resolver.save_report(output_base)

    # Simulation Loop
    
    scene = bpy.context.scene
    
    # 3. Physics Simulation (The Drop)
    # 150 frames ensures full settling at small LEGO scale
    SETTLE_FRAMES = 150 
    scene.frame_end = SETTLE_FRAMES + 10
    
    offset_idx = data.get('offset_idx', 0)
    
    # ─── Post-Physics Snap Function (Moved to global scope) ───
    # ──────────────────────────────────────────────────────────
    
    # We will do 2 variations (lighting/jitter) per physical drop to save CPU time
    # Total physical drops = num_images // 2
    num_drops = max(1, num_images // 2)
    
    for drop_idx in range(num_drops):
        scene.frame_set(1)
        bpy.ops.ptcache.free_bake_all()
        
        # 🎲 Dynamically spawn pieces from available types to reach PARTS_PER_IMAGE
        available_templates = unique_meshes
        
        # Variety Logic (D): How many distinct types in this image?
        # Passed from tiered model in run_local_render.py
        K = data.get('different_pieces', min(PARTS_PER_IMAGE, max(1, int(len(available_templates) * 0.75))))
        # Ensure K doesn't exceed available templates
        K = min(K, len(available_templates))
            
        selected_templates = random.sample(available_templates, K)
        
        active_pieces = []
        selected_class_ids = []
        
        # Distribute PARTS_PER_IMAGE across the K selected templates
        # Ensure equitable distribution with remainders
        base_instances = PARTS_PER_IMAGE // K
        extra_instances = PARTS_PER_IMAGE % K
        
        for i, template in enumerate(selected_templates):
            num_to_spawn = base_instances + (1 if i < extra_instances else 0)
            
            template_obj = template['obj']
            class_id = template['id']
            t_ldraw_id = template['ldraw_id']
            color_id = template.get('color_id', -1)
            color_name = template.get('color_name') or ''
            selected_class_ids.append(str(class_id))
            
            part_radius = get_max_xy_radius(template_obj)
            safe_limit = max(0.005, 0.10 - part_radius - 0.003)
            
            for _ in range(num_to_spawn):
                new_obj = copy_hierarchy(template_obj)
                
                new_obj['class_id'] = class_id
                new_obj['ldraw_id'] = t_ldraw_id
                new_obj['color_id_lego'] = color_id
                new_obj['color_name_lego'] = color_name
                new_obj['safe_limit'] = safe_limit
                
                new_obj.hide_render = False
                
                # Enable physics
                bpy.context.view_layer.objects.active = new_obj
                bpy.ops.rigidbody.object_add()
                new_obj.rigid_body.type = 'ACTIVE'
                new_obj.rigid_body.collision_shape = 'CONVEX_HULL'
                new_obj.rigid_body.friction = 1.0
                new_obj.rigid_body.restitution = 0.0
                # Anti-explosion for small pieces (1x1 etc)
                new_obj.rigid_body.use_margin = True
                new_obj.rigid_body.collision_margin = 0.0001
                new_obj.rigid_body.linear_damping = 0.5
                new_obj.rigid_body.angular_damping = 0.5
                
                active_pieces.append(new_obj)
        
        bpy.context.view_layer.update()
        
        selected_types_str = ', '.join([str(t['ldraw_id']) for t in selected_templates])
        print(f"  🏁 Drop {drop_idx+1}: {len(active_pieces)} pieces from types: {selected_types_str} (Classes: {', '.join(selected_class_ids)})")

        for obj in active_pieces:
            
            # Place at safe height for drop
            limit = obj.get('safe_limit', 0.08)
            base_z = random.uniform(0.003, 0.025)
            obj.location = (
                random.uniform(-limit, limit), 
                random.uniform(-limit, limit), 
                base_z
            )
            # Orientation diversity: force ~40% to start lying flat
            orientation_roll = random.random()
            if orientation_roll < 0.40:
                tilt_axis = random.choice(['x', 'y'])
                base_angle = random.choice([math.pi / 2, -math.pi / 2, math.pi])
                if tilt_axis == 'x':
                    obj.rotation_euler = (base_angle, random.uniform(-0.3, 0.3), random.uniform(0, 6.28))
                else:
                    obj.rotation_euler = (random.uniform(-0.3, 0.3), base_angle, random.uniform(0, 6.28))
            else:
                obj.rotation_euler = (random.uniform(0, 6.28), random.uniform(0, 6.28), random.uniform(0, 6.28))

        bpy.context.view_layer.update()
        
        # 4b. Run Settle Simulation — frame-by-frame for stability
        for frame in range(1, SETTLE_FRAMES + 1):
            scene.frame_set(frame)
            if frame == 1:
                bpy.context.view_layer.update()
        bpy.context.view_layer.update()
        
        # 4b.1 Post-Physics Safety Snap
        snap_to_ground(active_pieces)
        
        # 4b.2 Post-Physics Z-rotation jitter (adds 2D diversity as seen from top-down camera)
        for obj in active_pieces:
            current_z = obj.rotation_euler[2]
            obj.rotation_euler[2] = current_z + random.uniform(-0.5, 0.5)  # ~+/-30deg Z jitter
            
        bpy.context.view_layer.update()
        
        # 4c. Render 2 variations per drop (Lighting diversity)
        for var_idx in range(2):
            local_idx = (drop_idx * 2) + var_idx
            if local_idx >= num_images: break
            
            img_idx = local_idx + offset_idx
            print(f"PROGRESS: {local_idx + 1}/{num_images}")
            
            setup_lighting()
            setup_world_hdri(assets_dir)
            ground_obj = bpy.data.objects.get("Plane")
            if ground_obj:
                setup_ground_texture(ground_obj, assets_dir)
            
            # We will generate "Negative Samples" (empty backgrounds) ~10% of the time.
            is_empty_background = (random.random() < 0.10)
            
            # If empty background, move pieces out of view
            if is_empty_background:
                for obj in active_pieces: # Only active ones matter
                    obj.hide_render = True
                    obj.location.z -= 100.0 # Move them way below the floor
            else:
                # Ensure they are visible (they might have been hidden by a previous variation)
                for obj in active_pieces:
                    obj.hide_render = False
                    # Z will be restored after the render/labeling loop if it was moved
            
            bpy.context.view_layer.update()

            # Render - Ensure camera is active
            if not scene.camera:
                scene.camera = bpy.data.objects.get("Camera")
            
            worker_id = data.get('worker_id', '')
            prefix_base = f"w{worker_id}_" if worker_id != '' else ""
            img_prefix = f"{prefix_base}img_{img_idx:04d}"
            
            render_path = os.path.join(images_dir, f"{img_prefix}.jpg")
            scene.render.filepath = render_path
            
            if not scene.camera:
                print("⚠️ Camera still missing! Re-creating...")
                setup_camera()
                
            bpy.ops.render.render(write_still=True)
        
            # 5. Generate Labels (using helpers)
            label_path = os.path.join(labels_dir, f"{img_prefix}.txt")
            meta_path = os.path.join(output_base, "image_meta.jsonl")
            
            with open(label_path, 'w') as lf:
                if not is_empty_background:
                    active_ids_in_image = []
                    active_colors_in_image = []
                    print(f"  🔍 Labeling call: {len(active_pieces)} active pieces.")
                    for obj in active_pieces:
                        seg_line = generate_yolo_seg_label(obj, scene)
                        if seg_line:
                            lf.write(seg_line + "\n")
                            print(f"  ✅ Labeled {obj.name} ({obj.get('ldraw_id')}) at Z={obj.location.z:.4f}")
                            active_ids_in_image.append(obj.get('ldraw_id', 'unknown'))
                            active_colors_in_image.append(obj.get('color_id_lego', -1))
                    
                    # Write metadata for this image
                    active_color_names = [
                        obj.get('color_name_lego', '') for obj in active_pieces
                        if 'class_id' in obj and not obj.hide_render
                    ]
                    write_image_meta(meta_path, img_prefix, active_ids_in_image, active_colors_in_image, active_color_names)
                else:
                    # Empty background negative sample
                    write_image_meta(meta_path, img_prefix, [], [], [])
            
            # Restore Z position if it was a negative sample
            if is_empty_background:
                for obj in active_pieces:
                    obj.location.z += 100.0 # Return to ground level
                bpy.context.view_layer.update()

        # CLEANUP: Delete the dynamically spawned pieces for this drop
        bpy.ops.object.select_all(action='DESELECT')
        for obj in active_pieces:
            if obj.name in bpy.data.objects:
                select_hierarchy(obj)
        bpy.ops.object.delete()

    print("Blender script finished.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"\n❌ FATAL ERROR in Blender script:\n{e}")
        traceback.print_exc()
        sys.exit(1)
