import os
import cv2

# Define the directories containing the RGB images and masks
rgb_dir = "./datasets/BOP_DATASETS/doorlatch/train_pbr/000001/rgb"
mask_dir = "./datasets/BOP_DATASETS/doorlatch/train_pbr/000001/mask_visib"

# Get a list of all the RGB images and masks
rgb_files = sorted(os.listdir(rgb_dir))
mask_files = sorted(os.listdir(mask_dir))

# Initialize variables to keep track of the current image and mask
current_image_index = 0
current_mask_index = 0

# Loop through the images and masks
while True:
    # Get the current mask
    current_mask = cv2.imread(os.path.join(mask_dir, mask_files[current_mask_index]), cv2.IMREAD_GRAYSCALE)
    # Convert the mask to 3 channels
    current_mask = cv2.merge((current_mask, current_mask, current_mask))

    # Extract the image index from the mask file name
    image_index = int(mask_files[current_mask_index].split("_")[0])

    # Get the corresponding RGB image
    current_image = cv2.imread(os.path.join(rgb_dir, f"{image_index:06}.jpg"))

    # Overlay the mask on the image with alpha blending
    blended = cv2.addWeighted(current_image, 0.7, current_mask, 0.3, 0)

    # Show the blended image
    cv2.imshow("Blended Image", blended)

    # Wait for user input
    key = cv2.waitKey(0)

    # Check if the user pressed the right arrow key to go to the next mask
    if key == 83: # key code for right arrow
        current_mask_index += 1
        if current_mask_index >= len(mask_files):
            current_mask_index = 0
    # Check if the user pressed the left arrow key to go to the previous mask
    elif key == 81: # key code for left arrow
        current_mask_index -= 1
        if current_mask_index < 0:
            current_mask_index = len(mask_files) - 1
    # Check if the user pressed the down arrow key to go to the previous image
    elif key == 84: # key code for down arrow
        current_image_index -= 1
        if current_image_index < 0:
            current_image_index = len(rgb_files) - 1
        current_mask_index = 0
        while int(mask_files[current_mask_index].split("_")[0]) != current_image_index:
            current_mask_index += 1
    # Check if the user pressed the up arrow key to go to the next image
    elif key == 82: # key code for up arrow
        current_image_index += 1
        if current_image_index >= len(rgb_files):
            current_image_index = 0
        current_mask_index = 0
        while int(mask_files[current_mask_index].split("_")[0]) != current_image_index:
            current_mask_index += 1
    # Check if the user pressed the 'q' key to quit
    elif key == ord('q'):
        break

# Close all windows
cv2.destroyAllWindows()