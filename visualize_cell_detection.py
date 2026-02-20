import matplotlib.pyplot as plt
from src.IJEPA.transform.datatransform import get_pdl1_dataset
from src.parser.parser import parse_jepa_args
import cv2

args = parse_jepa_args()

train_loader, test_loader = get_pdl1_dataset(args.dataset_input, args.annotation_path, args.reverse_transform)

import matplotlib.pyplot as plt
import cv2
import numpy as np

for (images, annotations) in train_loader:

    image = images[0]
    bboxes = annotations["boxes"][0]
    labels = annotations["labels"][0]

    final_bbox = []
    for i, label in enumerate(labels):
        if label.item() == 2: ## only show cancer cells
            final_bbox.append(bboxes[i])

    img = image.permute(1, 2, 0).cpu().numpy()
    img = np.ascontiguousarray(img)

    for box in final_bbox:
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 1)

    plt.figure(figsize=(8,8))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"tumor bboxes: {len(final_bbox)}")
    plt.show()

    break

## run the command below to plot a random image with scaled bounding boxes
## py visualize_cell_detection.py --dataset_input __input_path__ --annotation_path __annotation_json_path__ --batch_size 1 --image_size 640 --reverse_transform y
