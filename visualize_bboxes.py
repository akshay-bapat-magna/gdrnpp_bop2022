import json
from collections import defaultdict
import cv2

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
	# resultsFile = "./output/yolox/bop_pbr/lb_cluttered_distractors_25k/inference/doorlatch_bop_test_pbr/"
	resultsFile = "./output/yolox/bop_pbr/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_doorlatch_pbr_doorlatch_bop_test/inference/doorlatch_bop_test_pbr/"
	# resultsFile = "/home/advrob/gdrn/gdrnpp_bop2022/output/yolox/bop_pbr/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_tless_real_pbr_tless_bop_test/inference/tless_bop_test_primesense/"
	resultsFile += "coco_bop.json"
	# resultsFile2 = "./output/yolox/bop_pbr/30epochs_25aug_1121/inference/doorlatch_bop_test_pbr/"
	# resultsFile2 += "coco_bop.json"
	imagePath = "./datasets/BOP_DATASETS/doorlatch/test_pbr/000000/rgb/"
	# imagePath = "./datasets/BOP_DATASETS/tless/test_primesense/000001/rgb/"

	with open(resultsFile, 'r') as f:
		results = json.load(f)

	# with open(resultsFile2, 'r') as f:
	# 	results2 = json.load(f)

	res_serialized = serializeResults(results)
	# res_serialized2 = serializeResults(results2)

	for imgid in range(1000):
		img = cv2.imread(f"{imagePath}{imgid:06d}.jpg")
		boxes, scores, cls_ids = getData(res_serialized, imgid)
		# boxes2, scores2, cls_ids2 = getData(res_serialized2, imgid)
		class_names = ["Dummy", "SB", "MB", "LB", "BSC", "SP"]
		# class_names = [str(i) for i in range(31)]
		thresh = 0.4

		img1 = vis(img, boxes, scores, cls_ids, thresh, class_names)
		# img2 = vis(img.copy(), boxes2, scores2, cls_ids2, thresh, class_names)
		cv2.imshow(f"Old model {imgid}", img1)
		# cv2.imshow(f"New model {imgid}", img2)
		# cv2.moveWindow(f"New model {imgid}", 2800, 150)
		if cv2.waitKey(0) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break
		cv2.destroyAllWindows()
		# cv2.imwrite(f"detection_{imgid}.jpg", img1)