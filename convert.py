import json


fin = "yolox_x_640_augCozyAAEhsv_ranger_30_epochs_doorlatch_pbr_doorlatch_bop_test/inference/doorlatch_bop_test_pbr/coco_instances_results_bop.json"
fout = "yolox_x_640_augCozyAAEhsv_ranger_30_epochs_doorlatch_pbr_doorlatch_bop_test/inference/doorlatch_bop_test_pbr/coco_bop.json"

with open(fin, 'r') as f:
	data = json.load(f)

with open(fout, 'w') as f:
	json.dump(data, f, indent=2)