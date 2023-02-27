import cv2
import json
import os

dataset = "doorlatch"	# doorlatch or tless

if dataset == "tless":
	json_path = "./datasets/BOP_DATASETS/tless/test/test_bboxes/"
	file_path = json_path + "yolox_x_640_tless_real_pbr_tless_bop_test.json"
	images_path = "./datasets/BOP_DATASETS/tless/test_primesense/000001/rgb/"
	scene = 1

elif dataset == "doorlatch":
	json_path = "./datasets/BOP_DATASETS/doorlatch/test/test_bboxes/"
	file_path = json_path + "yolox_x_640_doorlatch_real_pbr_doorlatch_bop_test.json"
	images_path = "./datasets/BOP_DATASETS/doorlatch/test_pbr/000000/rgb/"
	scene = 0

with open(file_path, 'r') as f:
	data = json.load(f)

for image_file in sorted(os.listdir(images_path)):
	img = cv2.imread(images_path+image_file)
	img_id = int(str(image_file).split(".")[0])
	boxes = [d["bbox_est"] for d in data[f"{scene}/{img_id}"]]
	
	for box in boxes:
		p1 = (int(box[0]), int(box[1]))
		p2 = (p1[0]+int(box[2]), p1[1]+int(box[3]))
		img = cv2.rectangle(img, p1, p2, (0,200,0), 2)
	if img_id >= 157:
		cv2.imshow(f"Image {img_id}", img)
		if cv2.waitKey(0) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break
		cv2.destroyAllWindows()
