import json
from tqdm import tqdm

# key = "tless"
# foldername = "test_primesense"
key = "doorlatch"
foldername = "test_pbr"

with open(f"./datasets/BOP_DATASETS/{key}/{foldername}/000000/scene_gt.json", 'r') as f:
	gts = json.load(f)

with open(f"./datasets/BOP_DATASETS/{key}/test_targets_ablation.json", "w") as f:
	d = []
	for scene in tqdm(range(0,1)):
		for img in range(1000):
			for obj in range(3,4):
				temp = {}
				temp["im_id"] = img
				temp["inst_count"] = len(gts[str(img)])
				temp["obj_id"] = obj
				temp["scene_id"] = scene
				d.append(temp.copy())

	json.dump(d, f, indent=4)