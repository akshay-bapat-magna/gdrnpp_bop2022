import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.append("./")

from lib.pysixd import inout, pose_error, misc


model_path = "datasets/BOP_DATASETS/doorlatch/models_eval/obj_000003.ply"
model_pts = inout.load_ply(model_path)
model_dia = 30.845675415790865

with open("eval/gts_10.json", 'r') as f:
    gts = json.load(f)

with open("eval/preds_10.json", 'r') as f:
    preds = json.load(f)

thresh = 0.02*model_dia
add_sum = 0
add_count = 0
add_all = []
re_all = []
te_all = []
top_one = False

num_scenes = len(gts.keys())
assert num_scenes == len(preds.keys())
for scene in tqdm(gts.keys()):
    num_objs = len(gts[scene])
    assert num_objs == len(preds[scene])
    if top_one:
        gt = np.array(gts[scene])
        pred = np.array(preds[scene])
        gt_R = gt[:3, :3]
        gt_t = gt[:3, 3]
        pred_R = pred[:3, :3]
        pred_t = pred[:3, 3]
        add = pose_error.add(pred_R, pred_t, gt_R, gt_t, model_pts["pts"])
        add_all.append(add)
        if add < thresh:
            add_sum += 1
        add_count += 1
        re_all.append(pose_error.re(pred_R, gt_R))
        te_all.append(pose_error.te(pred_t, gt_t))
    else:
        for obj in range(num_objs):
            gt = np.array(gts[scene][obj])
            pred = np.array(preds[scene][obj])
            gt_R = gt[:3, :3]
            gt_t = gt[:3, 3]
            pred_R = pred[:3, :3]
            pred_t = pred[:3, 3]
            add = pose_error.add(pred_R, pred_t, gt_R, gt_t, model_pts["pts"])
            add_all.append(add)
            if add < thresh:
                add_sum += 1
            add_count += 1

            re_all.append(pose_error.re(pred_R, gt_R))
            te_all.append(pose_error.te(pred_t, gt_t))

print(add_sum/add_count)
fig, (ad, te, re) = plt.subplots(3)
ad.hist(add_all)
ad.set_title("ADD")
re.hist(re_all)
re.set_title("RE")
te.hist(te_all)
te.set_title("TE")
plt.show()