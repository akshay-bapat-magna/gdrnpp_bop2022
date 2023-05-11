# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 12:59:41 2023

@author: bilahaid
"""

import numpy as np
import pickle
import json
import open3d as o3d
import open3d.visualization.gui as gui
import copy

import sys
sys.path.insert(0, '..')


def text_3d(text, pos, direction=None, degree=0.0, density=10, font='/usr/share/fonts/truetype/freefont/FreeMono.ttf', font_size=16):
    """
    Generate a 3D text point cloud used for visualization.
    :param text: content of the text
    :param pos: 3D xyz position of the text upper left corner
    :param direction: 3D normalized direction of where the text faces
    :param degree: in plane rotation of text
    :param font: Name of the font - change it according to your system
    :param font_size: size of the font
    :return: o3d.geoemtry.PointCloud object
    """
    if direction is None:
        direction = (0., 0., 1.)

    from PIL import Image, ImageFont, ImageDraw
    from pyquaternion import Quaternion

    font_obj = ImageFont.truetype(font, font_size*density)
    font_dim = font_obj.getsize(text)

    img = Image.new('RGB', font_dim, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font_obj, fill=(0, 0, 0))
    img = np.asarray(img)
    img_mask = img[:, :, 0] < 128
    indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T

    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(img[img_mask, :].astype(float) / 255.0)
    pcd.points = o3d.utility.Vector3dVector(indices / 1000.0 / density)

    raxis = np.cross([0.0, 0.0, 1.0], direction)
    if np.linalg.norm(raxis) < 1e-6:
        raxis = (0.0, 0.0, 1.0)
    trans = (Quaternion(axis=raxis, radians=np.arccos(direction[2])) *
             Quaternion(axis=direction, degrees=degree)).transformation_matrix
    trans[0:3, 3] = np.asarray(pos)
    pcd.transform(trans)
    return pcd


def append_text_anno(All_meshes, errors, imfile, imgid, infos, visib_thresh):

    for data in errors[imfile]:    
        text_pos = data['t']*1000
        text_pos[2] += 20
        # breakpoint()
        # if infos[str(imgid)][i_pred]["visib_fract"] > visib_thresh:
        All_meshes.append(text_3d(f"ADD: {data['ADD']*1000:.2f}, RE: {data['RE']:.2f}, TE: {data['TE']*1000:.2f}", text_pos, density=2, font_size=800))


# Load the  mesh.
path = '../datasets/BOP_DATASETS/doorlatch/models/obj_000003.ply'


# reading as triangle mesh
mesh = o3d.io.read_triangle_mesh(path, True)


gt_file = '../datasets/BOP_DATASETS/doorlatch/test_pbr/000000/scene_gt.json'
info_file = '../datasets/BOP_DATASETS/doorlatch/test_pbr/000000/scene_gt_info.json'
errors_file = '../gts_and_errors.pkl'
# pred_file = '../output/gdrn/doorlatch/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_doorlatch/inference_model_0147239/doorlatch_bop_test_pbr/convnext-a6-AugCosyAAEGray-BG05-mlL1-DMask-amodalClipBox-classAware-doorlatch-test_doorlatch_bop_test_pbr_preds.pkl'
pred_file = "../output/gdrn/doorlatch/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_doorlatch/inference_model_final/doorlatch_bop_test_pbr/preds.pkl"
corr_file = "../tracker.npy"

with open(pred_file, 'rb') as f:
    preds = pickle.load(f)
    
with open(gt_file, 'r') as f:
    gts = json.load(f)

with open(info_file, 'r') as f:
    infos = json.load(f)

with open(errors_file, 'rb') as f:
    errors = pickle.load(f)

with open(corr_file, 'rb') as f:
    corr = np.load(f, allow_pickle=True)

d = preds['3']
imgid = 0
visib_thresh = 0.8

for imfile, v in d.items():

    # if "000073" not in imfile:
    #     imgid += 1
    #     continue
    
    meshes_predictions = []
    All_meshes = []

    # Predictions
    for i_pred in range(len(v)):
        # retreive rotation and translation for predictions 
        rot_pred = v[i_pred]['R']
        trans_pred = v[i_pred]['t'] * 1000
        
        # combine rotation and translation in 4x4 homogenous transformation
        T_pred = np.vstack((np.hstack((rot_pred, trans_pred[:, None])), [0, 0, 0 ,1]))

        
        # paint blue color for predictions
        mesh.paint_uniform_color([0, 0, 1])
        
        # copy original mesh and apply prediction transformation
        mesh_pred_un = copy.deepcopy(mesh).transform(T_pred)
        
        # append prediction geometry
        # meshes_predictions.append(mesh_pred)

        
        # rot_gt = np.array(gts[str(imgid)][i_pred]["cam_R_m2c"]).reshape((3,3))
        
        # trans_gt = np.array(gts[str(imgid)][i_pred]["cam_t_m2c"])
        
        # combine rotation and translation in 4x4 homogenous transformation
        # T_gt = np.vstack((np.hstack((rot_gt, trans_gt[:, None])), [0, 0, 0 ,1]))
        
        # paint green color for gts
        # mesh.paint_uniform_color([0, 1, 0])
        
        # copy original mesh and apply prediction transformation
        # mesh_gt = copy.deepcopy(mesh).transform(T_gt)
        
        # append gt geometry
        # meshes_predictions.append(mesh_gt)
        
        # All_meshes.append(mesh_pred_un)
        # if infos[str(imgid)][i_pred]["visib_fract"] > visib_thresh:
        #     All_meshes.append(mesh_gt)

    # Ground truths
    for i in range(len(errors[imfile])):
        mesh.paint_uniform_color([0, 1, 0])
        rot_gt = errors[imfile][i]['R']
        trans_gt = errors[imfile][i]['t']*1000
        T_gt = np.vstack((np.hstack((rot_gt, trans_gt[:, None])), [0, 0, 0 ,1]))
        mesh_gt = copy.deepcopy(mesh).transform(T_gt)
        All_meshes.append(mesh_gt)
    #     text_pos = errors[imfile][i]['t'].copy()
    #     text_pos *= 1000
    #     text_pos[2] += 20
    #     All_meshes.append(text_3d(f"ADD: {errors[imfile][i]['ADD']*1000:.2f}, RE: {errors[imfile][i]['RE']:.2f}, TE: {errors[imfile][i]['TE']*1000:.2f}", text_pos, density=2, font_size=800))

    # GT and Pred correspondences
    for corr_dict in corr.item()[imfile]:
        # breakpoint()
        mesh.paint_uniform_color([0, 1, 0])
        rot_gt = corr_dict["R_gt"]
        trans_gt = corr_dict["T_gt"]*1000
        T_gt = np.vstack((np.hstack((rot_gt, trans_gt[:, None])), [0, 0, 0 ,1]))
        mesh_gt = copy.deepcopy(mesh).transform(T_gt)
        All_meshes.append(mesh_gt)

        mesh.paint_uniform_color([1.0, 0.67, 0])
        rot_pred = corr_dict["R_pred"]
        trans_pred = corr_dict["t_pred"]*1000
        T_pred = np.vstack((np.hstack((rot_pred, trans_pred[:, None])), [0, 0, 0 ,1]))
        mesh_pred = copy.deepcopy(mesh).transform(T_pred)
        All_meshes.append(mesh_pred)
        
    imgid = imgid + 1

    # append_text_anno(All_meshes, errors, imfile, imgid, infos, visib_thresh)
    
    # input(f"Image #{imgid-1}: Press Enter to continue...")
    print(f"Image #{imgid-1}")
    o3d.visualization.draw_geometries(All_meshes)
    