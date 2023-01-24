import json


# fin = "output/yolox/bop_pbr/lb_cluttered_distractors_25k/inference/doorlatch_bop_test_pbr/coco_instances_results_bop.json"
# fout = "output/yolox/bop_pbr/lb_cluttered_distractors_25k/inference/doorlatch_bop_test_pbr/coco_bop.json"

fin = "output/yolox/bop_pbr/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_doorlatch_pbr_doorlatch_bop_test/inference/doorlatch_bop_test_pbr/coco_instances_results_bop.json"
fout = "output/yolox/bop_pbr/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_doorlatch_pbr_doorlatch_bop_test/inference/doorlatch_bop_test_pbr/coco_bop.json"

# fin = "/home/advrob/gdrn/gdrnpp_bop2022/output/yolox/bop_pbr/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_tless_real_pbr_tless_bop_test/inference/tless_bop_test_primesense/coco_instances_results_bop.json"
# fout = "/home/advrob/gdrn/gdrnpp_bop2022/output/yolox/bop_pbr/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_tless_real_pbr_tless_bop_test/inference/tless_bop_test_primesense/coco_bop.json"

with open(fin, 'r') as f:
	data = json.load(f)

with open(fout, 'w') as f:
	json.dump(data, f, indent=2)