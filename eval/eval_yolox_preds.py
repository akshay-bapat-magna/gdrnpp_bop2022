import numpy as np
import json
import torch
from tqdm import tqdm


def get_predictions(preds_path, conv_to_xyxy=False):
    all_preds = {}
    compensate_offset = True

    preds_raw = torch.load(preds_path)
    # print(preds_raw[0]["instances"][0].keys())
    # print(preds_raw[0]["instances_bop"][0].keys())

    for img in preds_raw:
        all_preds[img['image_id']] = []
        for det in img['instances']:
            bbox = det['bbox']
            if compensate_offset:
                bbox[0]/=1.125
                bbox[1]/=1.125
                bbox[2]/=1.125
                bbox[3]/=1.125
            if conv_to_xyxy:
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
            all_preds[img['image_id']].append([det['category_id'], bbox])
    
    return all_preds


def get_gt(gt_path):
    all_gt = {}
    with open(gt_path, 'r') as f:
        gt_info = json.load(f)
        for img, value in gt_info.items():
            all_gt[int(img)] = []
            for obj in value:
                all_gt[int(img)].append([None, obj['bbox_obj']])
    
    return all_gt


def compute_iou(pred, gt):
    xa = max(pred[0], gt[0])
    xb = min(pred[2], gt[2])
    ya = max(pred[1], gt[1])
    yb = min(pred[3], gt[3])

    inter_area = max(0, xb-xa+1)*max(0, yb-ya+1)
    pred_area = max(0, pred[2]-pred[0]+1)*max(0, pred[3]-pred[1]+1)
    gt_area = max(0, gt[2]-gt[0]+1)*max(0, gt[3]-gt[1]+1)
    union_area = float(pred_area + gt_area - inter_area)
    # print(f"Computing IoU for {pred} and {gt}: {inter_area/union_area}")
    assert union_area > 0

    return inter_area/union_area


def compute_recall(scores, recall_threshold):
    successful = 0
    total = 0
    for img in scores:
        for det in img:
            if det[0] > recall_threshold:
                successful += 1
            total += 1
    
    return successful/total, total


def main():
    gt_path = "/home/advrob/gdrn/gdrnpp_bop2022/datasets/BOP_DATASETS/doorlatch/test_pbr/000000/"
    gt_path += "scene_gt_info.json"

    preds_path = "/home/advrob/gdrn/gdrnpp_bop2022/output/yolox/" +\
    "bop_pbr/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_doorlatch_pbr_doorlatch_bop_test/" +\
    "inference/doorlatch_bop_test_pbr/"
    preds_path += "instances_predictions.pth"

    num_imgs = 1000
    recall_thresholds = [0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5]

    preds = get_predictions(preds_path, True)
    # preds = sorted(preds, key=lambda x : x[1][0])
    gts = get_gt(gt_path)

    scores = []

    for img in tqdm(range(num_imgs), desc="Computing IoU"):
        temp = []
        if len(gts[img]) > len(preds[img]):
            print(img)
        for box in gts[img]:
            max_iou = 0
            max_pred = [None, [0,0,0,0]]
            for pred_box in preds[img]:
                iou = compute_iou(pred_box[1], box[1])
                if iou > max_iou:
                    max_iou = iou
                    max_pred = pred_box
            
            temp.append([max_iou, max_pred[1].copy()])
            if max_iou > 0:
                preds[img].remove(max_pred)
        
        scores.append(temp)
    
    with open("scores.json", "w") as f:
        json.dump(scores, f, indent=2)
    
    for thresh in recall_thresholds:
        recall, total = compute_recall(scores, thresh)
        print(f"Average recall for IoU > {thresh}: {recall}")
    print(f"Recall calculated for a total of {total} object instances.")


if __name__ == "__main__":
    main()