# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 12:59:41 2023

@author: bilahaid
"""

import argparse, re
import numpy as np
import pickle
import json
import open3d as o3d
import copy

import sys
sys.path.insert(0, '..')

# Load the  mesh.
path = '../datasets/BOP_DATASETS/doorlatch/models/obj_000003.ply'


# reading as triangle mesh
mesh = o3d.io.read_triangle_mesh(path, True)


gt_file = '../datasets/BOP_DATASETS/doorlatch/test_pbr/000000_ablation_08_cluttered/scene_gt.json'
pred_file = '../output/gdrn/doorlatch/ablation_08_cluttered_noregionloss_augafter20ep_weightedlossfn/inference_model_final/doorlatch_bop_test_pbr/convnext-a6-AugCosyAAEGray-BG05-mlL1-DMask-amodalClipBox-classAware-doorlatch-test_doorlatch_bop_test_pbr_preds.pkl'


with open(pred_file, 'rb') as f:
    preds = pickle.load(f)
    
with open(gt_file, 'r') as f:
    gts = json.load(f)
    
d = preds['3']
imgid = 0

for imfile, v in d.items():

    # if "000406" not in imfile:
    #     imgid += 1
    #     continue
    
    meshes_predictions = []
    All_meshes = []
    for i_pred in range(len(v)):
        
        
        # retreive rotation and translation for predictions 
        rot_pred = v[i_pred]['R']
        trans_pred = v[i_pred]['t'] * 1000
        
        # combine rotation and translation in 4x4 homogenous transformation
        T_pred = np.vstack((np.hstack((rot_pred, trans_pred[:, None])), [0, 0, 0 ,1]))

        
        # paint blue color for predictions
        mesh.paint_uniform_color([0, 0, 1])
        
        # copy original mesh and apply prediction transformation
        mesh_pred = copy.deepcopy(mesh).transform(T_pred)
        
        # append prediction geometry
        meshes_predictions.append(mesh_pred)

        
        rot_gt = np.array(gts[str(imgid)][i_pred]["cam_R_m2c"]).reshape((3,3))
        
        trans_gt = np.array(gts[str(imgid)][i_pred]["cam_t_m2c"])
        
        # combine rotation and translation in 4x4 homogenous transformation
        T_gt = np.vstack((np.hstack((rot_gt, trans_gt[:, None])), [0, 0, 0 ,1]))
        
        # paint green color for predictions
        mesh.paint_uniform_color([0, 1, 0])
        
        # copy original mesh and apply prediction transformation
        mesh_gt = copy.deepcopy(mesh).transform(T_gt)
        
        # append gt geometry
        meshes_predictions.append(mesh_gt)
        
        All_meshes.append(mesh_pred)
        All_meshes.append(mesh_gt)
        
    imgid = imgid + 1
    
    input(f"Image #{imgid-1}: Press Enter to continue...")
    o3d.visualization.draw_geometries(All_meshes)
    