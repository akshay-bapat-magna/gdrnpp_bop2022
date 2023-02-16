import json
from tqdm import tqdm

with open("./datasets/BOP_DATASETS/test_pbr/000000/scene_gt.json", 'r') as f:
	gts = json.load(f)

with open("./datasets/BOP_DATASETS/test_targets_ablation.json", "w") as f:
	d = []
	for scene in tqdm(range(1)):
		for img in range(1000):
			for obj in range(3,4):
				temp = {}
				temp["im_id"] = img
				temp["inst_count"] = len(gts[str(img)])
				temp["obj_id"] = obj
				temp["scene_id"] = scene
				d.append(temp.copy())

	json.dump(d, f, indent=4)