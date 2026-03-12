import matplotlib.pyplot as plt
from src.data_preprocess.dataloader import get_pdl1_dataset
from src.parser.parser import parse_jepa_args
import cv2
import numpy as np

args = parse_jepa_args()

args.batch_size = 1

train_loader, test_loader = get_pdl1_dataset(args.dataset_input, args.annotation_path, args.reverse_transform)

def denormalize(img, mean, std):
    img = img * np.array(std) + np.array(mean)
    return np.clip(img, 0, 1)

mean = [0.4914, 0.4822, 0.4465]
std = [0.247, 0.243, 0.261]

count = 0
for (images, annotations) in train_loader:
    if count >= 3:
        break

    image = images[0]
    bboxes = annotations[0]["boxes"]

    img = image.permute(1, 2, 0).cpu().numpy()
    img = denormalize(img, mean, std)
    img = np.ascontiguousarray(img)

    for box in bboxes:
        # box is [x, y, w, h]
        x_min = int(box[0])
        y_min = int(box[1])
        x_max = int(box[0] + box[2])
        y_max = int(box[1] + box[3])
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (1, 0, 0), 1)

    plt.figure(figsize=(8,8))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Image {count+1}: BBoxes: {len(bboxes)}")
    plt.show()

    count += 1

## run the command below to plot a random image with scaled bounding boxes
## py visualize_cell_detection.py --dataset_input __input_path__ --annotation_path __annotation_json_path__ --batch_size 1 --image_size 640 --reverse_transform y
