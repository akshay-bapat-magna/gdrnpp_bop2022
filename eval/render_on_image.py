import open3d as o3d
import numpy as np
import json
import pickle
import cv2

import sys
sys.path.insert(0, '..')
from lib.pysixd import renderer


output_folder = "convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_doorlatch/inference_model_final"
pred_file = f"../output/gdrn/doorlatch/{output_folder}/doorlatch_bop_test_pbr/convnext-a6-AugCosyAAEGray-BG05-mlL1-DMask-amodalClipBox-classAware-doorlatch-test_doorlatch_bop_test_pbr_preds.pkl"
with open(pred_file, 'rb') as f:
    preds = pickle.load(f)

gt_file = "../datasets/BOP_DATASETS/doorlatch/test_pbr/000000/scene_gt.json"
with open(gt_file, 'r') as f:
    gts = json.load(f)

K = np.array([1800.0, 0.0, 320.0000000074506, 0.0, 1800.0, 320.0000000074506, 0.0, 0.0, 1.0]).reshape((3,3))
fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
# pinhole = o3d.camera.PinHoleCameraIntrinsic(640, 640, K)


# mesh = o3d.geometry.TriangleMesh()
# mesh.paint_uniform_color([1.0, 0.0, 0.0])
mesh_path = "../datasets/BOP_DATASETS/doorlatch/models/obj_000003.ply"
mesh_pred = o3d.io.read_triangle_mesh(mesh_path, True)
mesh_gt = o3d.io.read_triangle_mesh(mesh_path, True)
# ren = renderer.create_renderer(640, 640, "python", shading="flat", bg_color=(1.0, 1.0, 1.0, 0.0))
# ren.add_object('3', mesh_path)

d = preds['3']
imgid = 0
for imfile, v in d.items():
    rot_pred = v[0]['R']
    trans_pred = v[0]['t']

    rot_gt = np.array(gts[str(imgid)][0]["cam_R_m2c"]).reshape((3,3))
    trans_gt = np.array(gts[str(imgid)][0]["cam_t_m2c"])/1000

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='GDRN Vis', width=900, height=900)
    vis.add_geometry(mesh_pred)
    vis.add_geometry(mesh_gt)
    
    mesh_pred.rotate(np.linalg.inv(rot_pred))
    mesh_pred.translate(trans_pred)
    mesh_gt.rotate(np.linalg.inv(rot_gt))
    mesh_gt.translate(trans_gt)
    mesh_pred.paint_uniform_color([0.8, 0.8, 0.8])
    mesh_gt.paint_uniform_color([0.4, 0.4, 0.4])

    vis.update_geometry(mesh_pred)
    vis.update_geometry(mesh_gt)
    vis.update_renderer()
    vis.run()
    vis.destroy_window()
    del vis

    imgid += 1
    