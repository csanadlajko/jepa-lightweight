from src.obj_detection.faster_rcnn import faster_rcnn_model, collate_fn
from src.IJEPA.transform.healthcare.lung_cancer_dataprocess import PDL1Dataset
from src.parser.parser import parse_jepa_args
from torch.utils.data.dataloader import DataLoader
import torch
from torchvision import transforms

args = parse_jepa_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

test_transform = transforms.Compose([
    transforms.ToTensor()
])

pdl_ds = PDL1Dataset(
    img_dir=args.dataset_input, 
    annotation_file=args.annotation_path,
    transforms=test_transform
)

loader = DataLoader(pdl_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

faster_rcnn_model.train()
optimizer = torch.optim.AdamW(faster_rcnn_model.parameters(), lr=args.lr)

for epoch in range(args.epochs):
    total_loss = 0.0

    for img, target in loader:
        images = [image.to(device) for image in img]
        targets = [{k: v.to(device) for k, v in t.items()} for t in target]

        loss_dict = faster_rcnn_model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    print(f"avg in epoch: {epoch+1} is: {total_loss / epoch+1}" )