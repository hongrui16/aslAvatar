

import os

os.environ['PYOPENGL_PLATFORM'] = 'egl'

import re
import argparse
import glob
import numpy as np
import cv2
import trimesh
import pyrender



# =============================================================================
# Auto camera from mesh bounding box
# =============================================================================

def compute_camera_for_mesh(vertices, img_w=512, img_h=512):
    """
    Compute reasonable camera intrinsics + translation to frame the mesh.

    Returns:
        focal: [fx, fy]
        princpt: [cx, cy]
        cam_trans: (3,) translation to apply to vertices
    """
    # Mesh center and extent
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    center = (vmin + vmax) / 2.0
    extent = (vmax - vmin).max()

    # Place camera looking at mesh center
    # Distance so that mesh fills ~70% of the image
    fov_deg = 50.0
    fov_rad = np.radians(fov_deg)
    distance = (extent / 2.0) / np.tan(fov_rad / 2.0) * 1.1

    focal_px = (img_w / 2.0) / np.tan(fov_rad / 2.0)
    focal = [focal_px, focal_px]
    princpt = [img_w / 2.0, img_h / 2.0]

    # Translation: move mesh so its center is at (0, 0, -distance)
    # OpenGL camera looks along -Z, so mesh must be at negative Z
    cam_trans = np.array([
        -center[0],
        -center[1],
        -distance - center[2],
    ], dtype=np.float32)

    return focal, princpt, cam_trans


# ─── Renderer ─────────────────────────────────────────────────────────

def render_mesh(img, vertices, faces, cam_param, alpha=0.8, color = 'light_green'):
    """Render SMPL-X mesh onto an image via pyrender."""
    focal, princpt = cam_param["focal"], cam_param["princpt"]

    camera = pyrender.IntrinsicsCamera(
        fx=focal[0], fy=focal[1],
        cx=princpt[0], cy=princpt[1],
        znear=0.01, zfar=100.0,
    )

    pyrender2opencv = np.array([
        [1.0,  0,  0, 0],
        [0,   -1,  0, 0],
        [0,    0, -1, 0],
        [0,    0,  0, 1],
    ])
    
    if color == "light_pink":
        baseColorFactor=(0.95, 0.6, 0.6, 1.0)
    else:
        baseColorFactor=(0.6, 0.9, 0.65, 1.0),  # light green

    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.1,
        roughnessFactor=0.4,
        alphaMode="OPAQUE",
        emissiveFactor=(0.15, 0.2, 0.15),
        baseColorFactor = baseColorFactor
    )

    body_trimesh = trimesh.Trimesh(vertices, faces, process=False)
    body_mesh = pyrender.Mesh.from_trimesh(body_trimesh, material=material)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    cam_pose = pyrender2opencv @ np.eye(4)

    scene = pyrender.Scene(
        bg_color=[0.0, 0.0, 0.0, 0.0],
        ambient_light=(0.3, 0.3, 0.3),
    )
    scene.add(camera, pose=cam_pose)
    scene.add(light, pose=cam_pose)
    scene.add(body_mesh, "mesh")

    r = pyrender.OffscreenRenderer(
        viewport_width=img.shape[1],
        viewport_height=img.shape[0],
        point_size=1.0,
    )

    color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    color = color.astype(np.float32) / 255.0

    valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis] * alpha
    img_f = img.astype(np.float32) / 255.0
    output = color[:, :, :3] * valid_mask + (1 - valid_mask) * img_f
    return (output * 255).astype(np.uint8)



    