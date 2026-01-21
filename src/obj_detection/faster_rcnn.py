import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

## testing on pretrained
faster_rcnn_model = torchvision.models.detection.faster_rcnn.fasterrcnn_resnet50_fpn(weights=None, num_classes=4)

in_features = faster_rcnn_model.roi_heads.box_predictor.cls_score.in_features
faster_rcnn_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 4)

def collate_fn(batch):
    return tuple(zip(*batch))