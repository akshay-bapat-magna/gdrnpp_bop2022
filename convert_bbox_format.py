import json
from tqdm import tqdm
from datetime import datetime


num_scenes = 50
filename = "scene_gt_info.json"
datasetPath = "datasets/BOP_DATASETS/doorlatch/train_pbr/"
now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

for scene in tqdm(range(num_scenes)):
	json_path = f"{datasetPath}{scene:06d}/{filename}"
	with open(json_path, 'r') as f:
		data = json.load(f)

	# key = each image
	for key, value in data.items():

		# each list entry corresponds to one object dict
		for obj in value:
			obj["bbox_obj"][2] += obj["bbox_obj"][0]
			obj["bbox_obj"][3] += obj["bbox_obj"][1]
			obj["bbox_visib"][2] += obj["bbox_visib"][0]
			obj["bbox_visib"][3] += obj["bbox_visib"][1]

	with open(json_path, 'w') as f:
		json.dump(data, f, indent=1)
	
	with open(f"{datasetPath}{scene:06d}/update_log.txt", 'a') as f:
		f.write(f"\nscene_gt.info updated on {now}")