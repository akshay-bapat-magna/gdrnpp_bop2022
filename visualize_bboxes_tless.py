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


def getData(res, id):
	boxes = []
	scores = []
	cls_ids = []
	for det in res[id]:
		boxes.append(det["bbox"])
		scores.append(det["score"])
		cls_ids.append(det["category_id"])

	return boxes, scores, cls_ids


if __name__ == "__main__":
	rf_dict = {
		"doorlatch": "./output/yolox/bop_pbr/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_doorlatch_pbr_doorlatch_bop_test/inference/doorlatch_bop_test_pbr/",
		"tless": "./output/yolox/bop_pbr/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_tless_pbr_tless_bop_test/inference/tless_bop_test_primesense/",
	}
	dataset = "tless"
	resultsFile = rf_dict[dataset]
	resultsFile += "coco_bop.json"
	imagePath = f"./datasets/BOP_DATASETS/{dataset}/test_primesense/000001/rgb/"

	with open(resultsFile, 'r') as f:
		results = json.load(f)

	res_serialized = serializeResults(results)

	for imgid in res_serialized.keys():
		img = cv2.imread(f"{imagePath}{imgid:06d}.png")
		boxes, scores, cls_ids = getData(res_serialized, imgid)
		# class_names = ["Dummy", "SB", "MB", "LB", "BSC", "SP"]
		class_names = list(range(50))
		thresh = 0.9

		img = vis(img, boxes, scores, cls_ids, thresh, class_names)
		cv2.imshow(f"Image {imgid}", img)
		if cv2.waitKey(0) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break
		cv2.destroyAllWindows()