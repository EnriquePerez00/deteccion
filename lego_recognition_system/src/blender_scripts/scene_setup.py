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
    scene.cycles.samples = 64  
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
    
    print("  ⚡ Cycles Eco-Mode: 64 samples + OptiX denoising. Tiling DISABLED. Film Transparent ENABLED.")

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

def snap_to_ground(objects, ground_z=0.0, max_hover=0.003):
    """Force any piece still hovering above max_hover down to ground contact.
    This is a safety net after the physics simulation."""
    snapped = 0
    for obj in objects:
        if obj.location.z > ground_z + max_hover:
            # Check if piece is supposed to rest on another piece
            if obj.location.z > ground_z + 0.01:  # > 1cm above ground
                # Special case for ref_pieza: move to ground
                obj.location.z = ground_z + 0.001
                snapped += 1
    if snapped > 0:
        print(f"  📌 Post-physics snap: {snapped} pieces adjusted to ground.")

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
    """Create a camera at 70cm height with 126mm lens to cover 20x20cm surface.
    iPhone 16 @ 24MP zenithal reference: sensor 36mm, distance 70cm, FOV 20cm.
    Lens = (sensor × distance) / field_width = (36 × 0.70) / 0.20 = 126mm
    """
    bpy.ops.object.camera_add(location=(0, 0, 0.7))
    cam = bpy.context.object
    cam.data.lens = 126            # 126mm at 0.7m covers ~20cm width on 36mm sensor
    cam.data.sensor_width = 36.0   # Full-frame equivalent
    
    # Sharp Focus at 70cm
    cam.data.dof.use_dof = True
    cam.data.dof.focus_distance = 0.7  # Sharp on the ground/surface
    cam.data.dof.aperture_fstop = 11.0 # f/11 for deep focus (keeps piece tops sharp)
    cam.data.dof.aperture_blades = 7
    
    bpy.context.scene.camera = cam
    return cam

def setup_lighting():
    """Create realistic randomized lighting. 5 presets simulating real environments."""
    # Delete existing lights
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if obj.type == 'LIGHT':
            obj.select_set(True)
    bpy.ops.object.delete()

    # --- Pick one of realistic scenarios ---
    # favoring environments where the HDRI dominates
    scenario = random.choice(['overhead', 'warm_side', 'cool_mixed', 'overcast', 'harsh', 'hdri_only', 'hdri_only'])

    if scenario == 'hdri_only':
        print("  🌍 Scenario: Pure HDRI illumination + Subtle Fill.")
        # Add a very subtle overhead fill just to lift the deep shadows
        bpy.ops.object.light_add(type='AREA', location=(0, 0, 1.5))
        key = bpy.context.object
        key.data.energy = random.uniform(100, 200)
        return scenario

    if scenario == 'overhead':
        # Studio overhead panel
        bpy.ops.object.light_add(type='AREA', location=(0, 0, 1.5))
        key = bpy.context.object
        key.data.energy = random.uniform(600, 1000) # Boosted
        key.data.color = (1.0, random.uniform(0.92, 1.0), random.uniform(0.85, 1.0))
        key.scale = (1.5, 1.5, 1)
        # Weak fill from side
        bpy.ops.object.light_add(type='POINT', location=(random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), 0.8))
        fill = bpy.context.object
        fill.data.energy = random.uniform(100, 250) # Boosted

    elif scenario == 'warm_side':
        # Warm sunlight from one side
        bpy.ops.object.light_add(type='AREA', location=(random.uniform(1.0, 1.5), random.uniform(-0.5, 0.5), random.uniform(0.8, 1.2)))
        key = bpy.context.object
        key.data.energy = random.uniform(800, 1200) # Boosted
        key.data.color = (1.0, random.uniform(0.80, 0.95), random.uniform(0.65, 0.80))  # subtle warm
        key.scale = (random.uniform(0.5, 1.0), random.uniform(0.5, 1.0), 1)
        key.rotation_euler = (0, random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5))
        # Soft fill opposite side
        bpy.ops.object.light_add(type='AREA', location=(-random.uniform(0.8, 1.2), 0, random.uniform(0.7, 1.0)))
        fill = bpy.context.object
        fill.data.energy = random.uniform(200, 600) # Boosted
        fill.data.color = (random.uniform(0.7, 0.9), random.uniform(0.8, 1.0), 1.0)  # cool blue fill

    elif scenario == 'cool_mixed':
        # Multiple colored lights
        bpy.ops.object.light_add(type='SPOT', location=(-1.2, 1.2, 1.0))
        l1 = bpy.context.object
        l1.data.energy = 800
        l1.data.color = (0.7, 0.8, 1.0) # Cool blue
        bpy.ops.object.light_add(type='POINT', location=(1.0, -1.0, 0.8))
        l2 = bpy.context.object
        l2.data.energy = 500
        l2.data.color = (1.0, 0.8, 0.7) # Warm orange
        bpy.ops.object.light_add(type='AREA', location=(0, 0, 1.5))
        l3 = bpy.context.object
        l3.data.energy = 300
        l3.data.color = (0.9, 1.0, 0.9) # Greenish fill
        l3.scale = (1.0, 1.0, 1)

    elif scenario == 'overcast':
        # Soft/diffuse: simulate cloudy day, multiple weak lights
        for _ in range(3):
            bpy.ops.object.light_add(type='AREA', location=(
                random.uniform(-0.8, 0.8), random.uniform(-0.8, 0.8), random.uniform(1.0, 1.5)))
            l = bpy.context.object
            l.data.energy = random.uniform(200, 400) # Boosted
            l.data.color = (random.uniform(0.9, 1.0), random.uniform(0.9, 1.0), random.uniform(0.9, 1.0))
            l.scale = (random.uniform(4, 8), random.uniform(4, 8), 1)

    else:  # harsh
        # Single strong point — simulates smartphone flash or task lamp
        bpy.ops.object.light_add(type='SPOT', location=(
            random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), random.uniform(0.8, 1.2)))
        key = bpy.context.object
        key.data.energy = random.uniform(600, 1200) # Boosted
        key.data.spot_size = random.uniform(0.5, 1.2)
        key.data.color = (1.0, random.uniform(0.88, 1.0), random.uniform(0.80, 1.0))
        # Very subtle fill to avoid pure black shadows
        bpy.ops.object.light_add(type='POINT', location=(0, 0, 0.7))
        fill = bpy.context.object
        fill.data.energy = random.uniform(50, 120) # Boosted

    print(f"  💡 Lighting scenario: '{scenario}'")
    
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
        safe_set_input(distort, 'Distort', 0.01)
        safe_set_input(distort, 1, 0.01)
        safe_set_input(distort, 'Dispersion', 0.02)
        safe_set_input(distort, 2, 0.02)

    # 2. Glare (Bloom)
    glare = safe_node_new('CompositorNodeGlare', 'Lens Bloom')
    if glare:
        safe_set(glare, 'glare_type', 'FOG_GLOW')
        safe_set(glare, 'glare_quality', 'HIGH')
        safe_set(glare, 'quality', 'HIGH')

    # 3. Grain (Sensor Mix)
    noise_mix = safe_node_new('CompositorNodeMixRGB', 'Sensor Grain Mix')
    noise_tex = None
    if noise_mix:
        safe_set(noise_mix, 'blend_type', 'OVERLAY')
        safe_set_input(noise_mix, 'Fac', 0.05)
        safe_set_input(noise_mix, 0, 0.05)
        try:
            noise_tex = nodes.new(type='CompositorNodeTexture')
            if "SensorNoise" not in bpy.data.textures:
                tex = bpy.data.textures.new("SensorNoise", type='NOISE')
                if hasattr(tex, 'noise_scale'): tex.noise_scale = 0.005
            noise_tex.texture = bpy.data.textures["SensorNoise"]
        except: pass

    # --- Linking Logic (CHAIN: Render -> Distort -> Glare -> Grain -> Composite) ---
    try:
        last_out = render_layers.outputs[0]
        
        if distort:
            links.new(last_out, distort.inputs[0])
            last_out = distort.outputs[0]

        if glare:
            links.new(last_out, glare.inputs[0])
            last_out = glare.outputs[0]
            
        if noise_mix and noise_tex:
            links.new(last_out, noise_mix.inputs[1])
            links.new(noise_tex.outputs[0], noise_mix.inputs[2])
            last_out = noise_mix.outputs[0]
            
        # Final connection
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
    """Sets a random texture from assets_dir/backgrounds for the ground plane."""
    bg_dir = os.path.join(assets_dir, "backgrounds")
    if not os.path.exists(bg_dir):
        return

    bg_files = [f for f in os.listdir(bg_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]
    if not bg_files:
        return

    bg_path = os.path.join(bg_dir, random.choice(bg_files))
    print(f"🖼️ Applying Ground Texture: {bg_path}")

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
        node_tex.image = bpy.data.images.load(bg_path)
    except Exception as e:
        print(f"  ❌ Error loading ground texture: {e}")
        return

    links.new(node_coord.outputs['UV'], node_mapping.inputs['Vector'])
    links.new(node_mapping.outputs['Vector'], node_tex.inputs['Vector'])
    links.new(node_tex.outputs['Color'], node_bsdf.inputs['Base Color'])
    links.new(node_bsdf.outputs['BSDF'], node_out.inputs['Surface'])

    # Randomized Rotation and Roughness
    node_mapping.inputs['Rotation'].default_value[2] = random.choice([0, 1.57, 3.14, 4.71])
    node_bsdf.inputs['Roughness'].default_value = random.uniform(0.6, 1.0)
    
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
            bevel.width = 0.0002       # 0.2mm — subtle, matches real LEGO tolerance
            bevel.segments = 2         # 2 segments = smooth highlight, not faceted
            bevel.limit_method = 'ANGLE'  # Only bevel clean edges, protect complex geometry
            bevel.angle_limit = 0.523  # ~30° — avoids artifacts on non-manifold stud mesh
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
                    
                    bsdf.inputs["Roughness"].default_value = 0.15
                    bsdf.inputs["IOR"].default_value = 1.45
                    
                    # Subsurface logic
                    if "Subsurface Weight" in bsdf.inputs:
                        bsdf.inputs["Subsurface Weight"].default_value = 0.02
                    elif "Subsurface" in bsdf.inputs:
                        bsdf.inputs["Subsurface"].default_value = 0.02
                        
                    # Clearcoat
                    if "Coat Weight" in bsdf.inputs:
                        bsdf.inputs["Coat Weight"].default_value = 0.1
                        bsdf.inputs["Coat Roughness"].default_value = 0.05
                    elif "Clearcoat" in bsdf.inputs:
                        bsdf.inputs["Clearcoat"].default_value = 0.1
                        bsdf.inputs["Clearcoat Roughness"].default_value = 0.05
            
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
    import math
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

# ... (existing imports)

# Import LDrawResolver from local module
try:
    from ldraw_resolver import LDrawResolver
except ImportError:
    # If running from Blender, verify sys.path
    pass


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
    
    # Extract new Tiering parameters
    # Fallback for old templates
    global_render_engine = data.get('render_engine', 'CYCLES')
    res_x = data.get('resolution_x', 640)
    res_y = data.get('resolution_y', 640)
    
    setup_render_engine(engine=global_render_engine, resolution=(res_x, res_y))
    setup_compositor()  # Sensor noise + bloom glare post-processing
    cam = setup_camera()
    setup_lighting()
    setup_world_hdri(assets_dir)
    
    # Setup Color Management — AgX for better dynamic range on white pieces
    # AgX preserves internal stud shadows even on overexposed whites
    try:
        color_set = False
        for transform in ['AgX', 'Filmic', 'Standard']:
            try:
                bpy.context.scene.view_settings.view_transform = transform
                color_set = True
                print(f"  🎨 Color Transform: {transform} active.")
                break
            except TypeError:
                continue
        
        # Balanced exposure: not too dark, not too bright
        bpy.context.scene.view_settings.exposure = -0.15
        bpy.context.scene.view_settings.gamma = 1.0
        bpy.context.scene.sequencer_colorspace_settings.name = 'sRGB'
    except Exception as e:
        print(f"Warning: Could not set color management: {e}")

    # Create Ground Plane (20x20cm — matches iPhone 16 FOV at 70cm)
    bpy.ops.mesh.primitive_plane_add(size=0.2, location=(0, 0, 0))
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
    
    # Read render mode from config: 'images_mix' or 'ref_pieza'
    render_mode = data.get('render_mode', 'images_mix')
    PARTS_PER_IMAGE = data.get('parts_per_image', 20)  # For images_mix mode
    
    # 1. Setup Camera (Fixed Zenithal Centered)
    # Surface is 20x20cm centered at 0,0.
    cam.location = (0, 0, 0.70) 
    cam.rotation_euler = (0, 0, 0) 
    cam.data.sensor_width = 36.0 # Ensure standard full-frame sensor
    cam.data.lens = 126.0 # 126mm to capture 20x20cm area at 70cm

    # Initialize Resolver
    resolver = LDrawResolver(ldraw_path_base)
    print(f"🔧 Render mode: {render_mode} | Parts per image: {PARTS_PER_IMAGE}")    


    # Iterate pieces to setup meshes and render engine
    # --- BEVEL LOD (Level of Detail) ---
    def apply_bevel_lod(obj, render_res=800, camera_fov_m=0.20):
        """Reduce bevel segments on small objects to speed up rendering."""
        try:
            bpy.context.view_layer.update()
            dims = obj.dimensions  # World-space size in metres
            screen_fraction = max(dims.x, dims.y) / camera_fov_m
            pixel_size = screen_fraction * render_res
            for mod in obj.modifiers:
                if mod.type == 'BEVEL':
                    if pixel_size < 80:
                        mod.segments = 1  # Very small piece (<80px)
                    elif pixel_size < 150:
                        mod.segments = 2  # Medium piece (80-150px)
                    # Else: keep original segments for large pieces
            # Also recurse into children
            for child in obj.children:
                apply_bevel_lod(child, render_res, camera_fov_m)
        except Exception as e:
            print(f"  ⚠️ Bevel LOD skipped for {obj.name}: {e}")

    # --- GLOBAL RENDER SETUP ---
    # Setup once for the whole run to avoid driver overhead in loops
    first_pc = pieces_config[0] if pieces_config else {'engine': 'CYCLES', 'res': 800}
    setup_render_engine(engine=first_pc.get('engine', 'CYCLES'), resolution=(first_pc.get('res', 800), first_pc.get('res', 800)))
    
    for i, pc in enumerate(pieces_config):
        part = pc['part']
        ldraw_id = part['ldraw_id']
        color_id = part.get('color_id', None)
        color_name = part.get('color_name', None)
        num_images = pc['imgs']
        engine = pc['engine']
        res = pc['res']
        
        color_info = f" | Color: {color_id} ({color_name or 'n/a'})" if color_id is not None else ""
        print(f"\n🚀 Loading part {ldraw_id}{color_info} | Tier: {pc['tier']}")
        
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
            
    # Function to get the max dimension in XY plane to prevent clipping out of the 20x20cm surface
    def get_max_xy_radius(obj):
        corners = get_hierarchy_corners(obj)
        if not corners: return 0.02 # Default 2cm
        max_r = 0
        for c in corners:
            # We care about distance from object center in local XY
            r = math.sqrt(c.x**2 + c.y**2)
            if r > max_r: max_r = r
        return max_r

    # ═══════════════════════════════════════════════════════════
    # MODE: ref_pieza — Single piece centered, 360° rotation
    # ═══════════════════════════════════════════════════════════
    if render_mode == 'ref_pieza':
        print("📸 MODE: ref_pieza — Single piece reference renders")
        if not unique_meshes:
            print("❌ No meshes loaded for ref_pieza mode.")
        else:
            # Use the first (and only) piece
            template = unique_meshes[0]
            template_obj = template['obj']
            class_id = template['id']
            ldraw_id_ref = template_obj.get('ldraw_id', set_id)
            color_id_ref = template.get('color_id', -1)
            color_name_ref = template.get('color_name') or ''
            
            # Place piece at center of the 20x20cm zone
            template_obj.location = (0, 0, 0.001)  # Just above ground
            set_origin_to_center(template_obj)
            
            # 300 images: Full random 3D rotations and ground snapping
            ref_n_images = data.get('ref_num_images', 300)
            
            img_count = 0
            offset_idx = data.get('offset_idx', 0)
            worker_id = data.get('worker_id', '')
            prefix_base = f"w{worker_id}_" if worker_id != '' else ""
            
            # --- PHYSICS STABILITY SETUP ---
            SETTLE_FRAMES = 150
            num_drops = 30
            rots_per_face = max(1, ref_n_images // num_drops)
            
            # Setup Rigid Body for the template
            bpy.context.view_layer.objects.active = template_obj
            if not template_obj.rigid_body:
                bpy.ops.rigidbody.object_add()
            template_obj.rigid_body.type = 'ACTIVE'
            # Convex Hull is faster and accurate enough for stable resting faces
            template_obj.rigid_body.collision_shape = 'CONVEX_HULL'
            template_obj.rigid_body.friction = 1.0
            template_obj.rigid_body.restitution = 0.0
            template_obj.rigid_body.linear_damping = 0.6
            template_obj.rigid_body.angular_damping = 0.6

            scene = bpy.context.scene
            for d in range(num_drops):
                # Reset simulation for each drop
                scene.frame_set(1)
                bpy.ops.ptcache.free_bake_all()
                
                # 1. Random Toss (Drop from 5cm)
                template_obj.location = (0, 0, 0.05)
                template_obj.rotation_euler = (
                    random.uniform(0, 6.28),
                    random.uniform(0, 6.28),
                    random.uniform(0, 6.28)
                )
                
                # 2. Settle Simulation
                for f in range(1, SETTLE_FRAMES + 1):
                    scene.frame_set(f)
                
                # Post-physics snap for ground contact
                snap_to_ground([template_obj])
                
                # 3. Capture N Z-rotations per face
                for r in range(rots_per_face):
                    # Z-axis rotation to capture the same stable face from different angles
                    template_obj.rotation_euler[2] += (6.28 / rots_per_face) + random.uniform(-0.05, 0.05)
                    bpy.context.view_layer.update()
                    
                    # Vary lighting and background every 4 images
                    if img_count % 4 == 0:
                        setup_lighting()
                        setup_world_hdri(assets_dir)
                        ground_obj = bpy.data.objects.get("Plane")
                        if ground_obj:
                            setup_ground_texture(ground_obj, assets_dir)
                
                    # Render
                    if not scene.camera:
                        scene.camera = bpy.data.objects.get("Camera")
                    
                    img_idx = img_count + offset_idx
                    img_prefix = f"{prefix_base}img_{img_idx:04d}"
                    render_path = os.path.join(images_dir, f"{img_prefix}.jpg")
                    scene.render.filepath = render_path
                    
                    if img_count % 20 == 0 or img_count == ref_n_images - 1:
                        print(f"PROGRESS: {img_count + 1}/{ref_n_images} (Physics Stable Snap)")
                    
                    bpy.ops.render.render(write_still=True)
                    
                    # Generate label for centered piece
                    label_path = os.path.join(labels_dir, f"{img_prefix}.txt")
                    camera = scene.camera
                    
                    # Recalculate vertices after position update for labels
                    verts_3d = get_hierarchy_vertices(template_obj)
                    if verts_3d:
                        coords_2d = [world_to_camera_view(scene, camera, coord) for coord in verts_3d]
                        visible_points = []
                        for c in coords_2d:
                            if c.z > 0:
                                cx = max(0.0, min(1.0, c.x))
                                cy = max(0.0, min(1.0, c.y))
                                visible_points.append((cx, cy))
                        
                        hull = get_convex_hull(visible_points) if len(visible_points) >= 3 else []
                        
                        with open(label_path, 'w') as lf:
                            if hull and len(hull) >= 3:
                                pts = []
                                for corner in hull:
                                    px = max(0.0, min(1.0, corner[0]))
                                    py = max(0.0, min(1.0, 1.0 - corner[1]))
                                    pts.extend([f"{px:.6f}", f"{py:.6f}"])
                                lf.write(f"{class_id} {' '.join(pts)}\n")
                    
                    # Metadata
                    meta_path = os.path.join(output_base, "image_meta.jsonl")
                    with open(meta_path, 'a') as mf:
                        mf.write(json.dumps({
                            "img": f"{img_prefix}.jpg",
                            "ids": [ldraw_id_ref],
                            "color_ids": [color_id_ref],
                            "color_names": [color_name_ref]
                        }) + "\n")
                    
                    img_count += 1
            
            print(f"✅ ref_pieza complete: {img_count} images rendered for piece {ldraw_id_ref}")
        
        print("Blender script finished.")
        return  # Exit early for ref_pieza mode
    
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
        
        # Variety Logic: How many distinct types in this image?
        # Target 75% of available types, but cannot exceed the total number of physical pieces planned
        K = min(PARTS_PER_IMAGE, max(1, int(len(available_templates) * 0.75)))
        # If we have very few types (< 5), just use all of them
        if len(available_templates) < 5:
            K = len(available_templates)
            
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
        
            # 5. Generate Labels
            label_path = os.path.join(labels_dir, f"{img_prefix}.txt")
            # Create a separate metadata file for the indexer to map images to parts
            meta_path = os.path.join(output_base, "image_meta.jsonl")
            
            with open(label_path, 'w') as lf:
                # If negative sample, the file remains empty (0 bytes) as required by YOLO
                if not is_empty_background:
                    active_ids_in_image = []
                    active_colors_in_image = []
                    print(f"  🔍 Labeling call: {len(active_pieces)} active pieces.")
                    for obj in active_pieces: # Iterate only over visible/active pieces!
                        # print(f"    - Processing {obj.name} at Z={obj.location.z:.2f}")
                        camera = scene.camera
                        
                        # For OBB, using vertices is much more accurate than the 8 bounding box corners
                        # because the real shape limits the tight OBB fit.
                        verts_3d = get_hierarchy_vertices(obj)
                        if not verts_3d:
                            if not is_empty_background:
                                print(f"  👻 Piece {obj.name} ({obj.get('ldraw_id')}) NO VERTICES | children={len(obj.children)}")
                            continue
                        else:
                            if not is_empty_background:
                                print(f"  📝 Piece {obj.name} ({obj.get('ldraw_id')}) has {len(verts_3d)} verts from {len(obj.children)} immediate children")
                    
                        # Project vertices to 2D
                        coords_2d = [world_to_camera_view(scene, camera, coord) for coord in verts_3d]
                        
                        # Filter visible coords and clamp to 0-1 bounds
                        visible_points = []
                        for c in coords_2d:
                            if c.z > 0: # In front of camera
                                cx, cy = max(0.0, min(1.0, c.x)), max(0.0, min(1.0, c.y))
                                visible_points.append((cx, cy))
                                
                        if not visible_points: 
                            if not is_empty_background:
                                print(f"  👻 Piece {obj.name} ({obj.get('ldraw_id')}) NOT VISIBLE at Z={obj.location.z:.4f}")
                            continue
                        
                        # Calculate Convex Hull for Instance Segmentation (Polygon Mask)
                        hull = get_convex_hull(visible_points)
                        
                        if not hull or len(hull) < 3:
                            if not is_empty_background:
                                print(f"  👻 Piece {obj.name} ({obj.get('ldraw_id')}) INVALID HULL (len={len(hull) if hull else 0})")
                            continue
                        
                        # Check area size to filter microscopic noise
                        xs = [p[0] for p in hull]
                        ys = [p[1] for p in hull]
                        if (max(xs) - min(xs)) < 0.002 or (max(ys) - min(ys)) < 0.002:
                            if not is_empty_background:
                                print(f"  👻 Piece {obj.name} ({obj.get('ldraw_id')}) TOO SMALL (size={(max(xs)-min(xs)):.4f}x{(max(ys)-min(ys)):.4f}) at Z={obj.location.z:.4f}")
                            continue
                            
                        # Write YOLO Segmentation format (class x1 y1 x2 y2 ... xn yn)
                        if 'class_id' in obj:
                            # Note: Blender Y goes up, YOLO Y goes down. We invert Y.
                            # YOLO-Seg expects a continuous sequence of normalized polygon points
                            pts = []
                            for corner in hull:
                                px = max(0.0, min(1.0, corner[0]))
                                py = max(0.0, min(1.0, 1.0 - corner[1])) # Invert Y for YOLO
                                pts.extend([f"{px:.6f}", f"{py:.6f}"])
                                
                            lf.write(f"{obj['class_id']} {' '.join(pts)}\n")
                            if not is_empty_background:
                                print(f"  ✅ Labeled {obj.name} ({obj.get('ldraw_id')}) at Z={obj.location.z:.4f}")
                            # Track which real part ID, color and color_name is in this image
                            active_ids_in_image.append(obj.get('ldraw_id', 'unknown'))
                            active_colors_in_image.append(obj.get('color_id_lego', -1))
                    
                    # Log real IDs + colors for this image to image_meta.jsonl
                    if active_ids_in_image:
                        active_color_names = [
                            obj.get('color_name_lego', '') for obj in active_pieces
                            if 'class_id' in obj and not obj.hide_render
                        ]
                        with open(meta_path, 'a') as mf:
                            mf.write(json.dumps({
                                "img": f"{img_prefix}.jpg", 
                                "ids": active_ids_in_image,
                                "color_ids": active_colors_in_image,
                                "color_names": active_color_names
                            }) + "\n")
            
            # Restore Z position if it was a negative sample
            if is_empty_background:
                for obj in active_pieces:
                    obj.location.z += 100.0 # Return to ground level
                bpy.context.view_layer.update()

        # CLEANUP: Delete the dynamically spawned pieces for this drop
        def select_hierarchy(o):
            o.select_set(True)
            for c in o.children:
                select_hierarchy(c)
                
        bpy.ops.object.select_all(action='DESELECT')
        for obj in active_pieces:
            if obj.name in bpy.data.objects:
                select_hierarchy(obj)
        bpy.ops.object.delete()    # Note: data.yaml is now generated by the calling logic (generator.py) 
    # to correctly handle merged datasets across multiple workers.
    print("Blender script finished.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"\n❌ FATAL ERROR in Blender script:\n{e}")
        traceback.print_exc()
        sys.exit(1)
