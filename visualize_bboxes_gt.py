import json
from collections import defaultdict
import cv2
import sys

from det.yolox.utils.visualize import vis


def serializeResults(results):
	out = defaultdict(list)

	for r in results:
		img = r["image_id"]
		del r["image_id"]
		del r["scene_id"]
		out[img].append(r)

	return out


def getData(res, idx, filter_score=False):
	boxes = []
	scores = []
	cls_ids = []
	filter_score = defaultdict(float)
	for det in res[idx]:
		if filter_score:
			pass
		else:
			boxes.append(det["bbox"])
			scores.append(det["score"])
			cls_ids.append(det["category_id"])

	return boxes, scores, cls_ids


if __name__ == "__main__":
	scene = int(sys.argv[1])

	# resultsFile = f"./datasets/BOP_DATASETS/doorlatch/train_pbr/{scene:06d}/"
	resultsFile = "./datasets/BOP_DATASETS/doorlatch/test_pbr/000000/"
	resultsFile += "scene_gt_info.json"

	# imagePath = f"./datasets/BOP_DATASETS/doorlatch/train_pbr/{scene:06d}/rgb/"
	imagePath = "./datasets/BOP_DATASETS/doorlatch/test_pbr/000000/rgb/"
	# imagePath = "./datasets/BOP_DATASETS/itodd/test/test/000001/gray/"

	predsFile = "scores.json"
	with open(predsFile, 'r') as f:
		preds = json.load(f)

	with open(resultsFile, 'r') as f:
		results = json.load(f)

	# with open(resultsFile2, 'r') as f:
	# 	results2 = json.load(f)

	# res_serialized = serializeResults(results)
	# res_serialized2 = serializeResults(results2)
	# bboxes = [[403.0, 269.5, 423.0, 408.5],
	# 			[-2.0, 309.25, 1111.0, 260.75],
	# 			[0.0625, 756.5, 53.4375, 95.0],
	# 			[0.125, 676.5, 406.875, 286.0],
	# 			[732.0, 703.0, 536.0, 257.0],
	# 			[762.5, 0.5, 509.5, 307.25],
	# 			[750.0, 0.125, 505.0, 308.375],
	# 			[328.0, 301.0, 786.0, 292.5],
	# 			[64.125, 653.0, 342.875, 308.0]]

	# bboxes = [[354.0, 378.5, 200.0, 200.0],
	# 		[407.5, 564.0, 196.0, 200.0],
	# 		[481.0, 196.0, 197.0, 194.5],
	# 		[625.5, 369.75, 195.0, 198.25],
	# 		[578.0, 549.0, 190.0, 192.0],
	# 		[632.0, 287.5, 192.0, 198.5],
	# 		[676.0, 485.75, 194.0, 200.25],
	# 		[409.25, 277.0, 197.75, 202.5],
	# 		[506.0, 449.0, 176.0, 182.0]]

	for imgid in range(1000):
		img = cv2.imread(f"{imagePath}{imgid:06d}.jpg")
		data = results[str(imgid)]
		p = preds[int(imgid)]

		for det in data:
			p1 = (det["bbox_obj"][0], det["bbox_obj"][1])
			p2 = (det["bbox_obj"][2], det["bbox_obj"][3])
			img = cv2.rectangle(img, p1, p2, (0,0,200), 2)
		
		for det in p:
			p1 = (int(det[1][0]), int(det[1][1]))
			p2 = (int(det[1][2]), int(det[1][3]))
			img = cv2.rectangle(img, p1, p2, (0,200,0), 1)
			cv2.putText(img, str(det[0]), (p1[0], p1[1]-5), \
				cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,200,0), 1)

		# for b in bboxes:
		# 	p1 = (int(b[0]), int(b[1]))
		# 	p2 = (int(b[0]+b[2]), int(b[1]+b[3]))
		# 	p3 = (int(b[2]), int(b[3]))
		# 	img = cv2.rectangle(img, p1, p2, (0,0,200), 1)			

		cv2.imshow(f"Image {imgid}", img)
		if cv2.waitKey(0) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break
		cv2.destroyAllWindows()
		# cv2.imwrite(f"detection_{imgid}.jpg", img)