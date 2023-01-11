import json
from tqdm import tqdm


num_scenes = 1
filename = "scene_gt_info.json"

for scene in tqdm(range(num_scenes)):
	json_path = f"{scene:06d}/{filename}"
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