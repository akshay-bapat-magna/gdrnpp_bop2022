import json
import torch
from tqdm import tqdm

# tless_json_path = "tless/test/test_bboxes/yolox_x_640_tless_real_pbr_tless_bop_test.json"
# with open(tless_json_path, 'r') as f:
# 	d = json.load(f)
# 	print(len(d['1/1']))

# doorlatch_original_path = "doorlatch/test/test_bboxes/"
# # filename = "yolox_x_640_doorlatch_real_pbr_doorlatch_bop_test.json"
# # filename = "coco_instances_results_bop.json"
# filename = "coco_instances_results.json"
# doorlatch_path = doorlatch_original_path + filename
# with open(doorlatch_path, 'r') as f:
# 	d = json.load(f)
# 	print(d[0])

# out = {}
# for item in d:

out = {}
doorlatch_original_path = "datasets/BOP_DATASETS/doorlatch/test/test_bboxes/"
preds = torch.load(doorlatch_original_path + "instances_predictions.pth")

compensate_offset = True

for item in tqdm(preds):
	key = f"0/{item['image_id']}"
	scores = [[], [], [], [], []]	# For one detection per obj: [0,0,0,0,0]
	bboxes = [[], [], [], [], []]

	# THIS BLOCK SAVES JUST ONE DETECTION (BEST) PER OBJ CATEGORY
	# for box in item['instances']:
	# 	if scores[box['category_id']] < box['score']:
	# 		scores[box['category_id']] = box['score']
	# 		bboxes[box['category_id']] = box['bbox']

	# THIS BLOCK SAVES ALL PREDICTIONS WITH SCORE > THRESH
	thresh = 0.6
	for box in item['instances']:
		if box['score'] > thresh:
			scores[box['category_id']].append(box['score'])
			bboxes[box['category_id']].append(box['bbox'])

	val = []
	for i in range(5):
		if len(bboxes[i]) > 0:	# If there are any detections of this category
			for j in range(len(bboxes[i])):
				temp = {}
				temp['bbox_est'] = bboxes[i][j]

				# Change format to xyxy as the entire pipeline uses this format
				# temp['bbox_est'][2] += temp['bbox_est'][0]
				# temp['bbox_est'][3] += temp['bbox_est'][1]

				if compensate_offset:
					temp['bbox_est'][0]/=1.125
					temp['bbox_est'][1]/=1.125
					temp['bbox_est'][2]/=1.125
					temp['bbox_est'][3]/=1.125

				temp['obj_id'] = i+1
				temp['score'] = scores[i][j]
				temp['time'] = 0.0
				val.append(temp)

	out[key] = val


output = "yolox_x_640_doorlatch_real_pbr_doorlatch_bop_test.json"
doorlatch_path = doorlatch_original_path + output
with open(doorlatch_path, 'w') as f:
	json.dump(out, f, indent=2)

output = "yolox_x_640_doorlatch_pbr_doorlatch_bop_test.json"
doorlatch_path = doorlatch_original_path + output
with open(doorlatch_path, 'w') as f:
	json.dump(out, f, indent=2)