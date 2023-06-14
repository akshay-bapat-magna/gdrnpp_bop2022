import numpy as np
import json
import pickle

import sys
sys.path.insert(0, '..')
from lib.pysixd import renderer


output_folder = "convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_doorlatch/inference_model_final"
# output_folder = "lb_ablation_08_10k_bs16/inference_"
# pred_file = f"../output/gdrn/doorlatch/{output_folder}/doorlatch_bop_test_pbr/convnext-a6-AugCosyAAEGray-BG05-mlL1-DMask-amodalClipBox-classAware-doorlatch_doorlatch_bop_test_pbr_preds.pkl"
pred_file = f"../output/gdrn/doorlatch/{output_folder}/doorlatch_bop_test_pbr/preds.pkl"
with open(pred_file, 'rb') as f:
    preds = pickle.load(f)

gt_file = "../datasets/BOP_DATASETS/doorlatch/test_pbr/000000/scene_gt.json"
with open(gt_file, 'r') as f:
    gts = json.load(f)

f = open("preds.txt", 'w')
f.close()
f = open("gts.txt", 'w')
f.close()

with open("preds.txt", 'a') as f:
    for imfile, v in preds['3'].items():
        f.write(imfile + "\n\n")
        for p in v:
            tf = np.eye(4)
            tf[:3,:3] = p['R']
            tf[:3, 3] = p['t']*1000
            np.savetxt(f, tf, fmt='%1.5f')
            f.write("\n")

with open("gts.txt", 'a') as f:
    imgid = 0
    for imfile, v in preds['3'].items():
        curr_gt = gts[str(imgid)]
        f.write(imfile + "\n\n")
        
        for i in range(len(curr_gt)):
            rot_gt = np.array(curr_gt[i]["cam_R_m2c"]).reshape((3,3))
            trans_gt = np.array(curr_gt[i]["cam_t_m2c"])
            tf = np.eye(4)
            tf[:3,:3] = rot_gt
            tf[:3, 3] = trans_gt
            np.savetxt(f, tf, fmt='%1.5f')
            f.write("\n")

        imgid += 1