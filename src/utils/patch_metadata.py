import torch
from typing import Any

## ntc: non tumor cell
## ni (no information): no class in the current patch
## npdl1: pdl1 negative tumor cell
## ppdl1: pdl1 positive tumor cell

class PatchProcesser(object):

    PATCH_DATA_MAP: dict[str, int] = {
        "ni": 0, # no class
        "3": 1, # non tumor
        "1": 2, # negative pd1
        "2": 3, # positive pdl1
        "1_3": 4, # negative pdl1 and non tumor cells
        "2_3": 5, # positive pdl1 and non tumor cells
        "1_2_3": 6, # all cells from all 3 classes
        "1_2": 7 # negative and positive pdl1 cells
    }

    def __init__(self, patch_size):
        self.patch_size = patch_size

    def _get_intersections(self, image_bboxes: list[list[int]], label_list: list[int], top_left_patch: tuple[int, int], bottom_right_patch: tuple[int, int]) -> int:
        assert len(image_bboxes) == len(label_list)
        jepa_classes = []
        px1, py1, px2, py2 = top_left_patch[0], top_left_patch[1], bottom_right_patch[0], bottom_right_patch[1]
        for i, bbox in enumerate(image_bboxes):
            bx1, by1, bx2, by2 = bbox[0], bbox[1], bbox[2], bbox[3]
            # left-bottom-right-top coordinate inspection for intersection
            if (bx2 > px1) and (by1 < py2) and (bx1 < px2) and (by2 > py1):
                # if all the requirements are met above, the bbbox falls inside the current patch
                if label_list[i] not in jepa_classes:
                    jepa_classes.append(label_list[i])

        jepa_classes = sorted(jepa_classes)
        jepa_classes = [str(item) for item in jepa_classes]

        # return ground truth class integer (only one)
        if len(jepa_classes) != 0:
            jepa_classlist_string = "_".join(jepa_classes)
            # weirdly small amount of gt classes ??
            return self.PATCH_DATA_MAP[jepa_classlist_string]
        return self.PATCH_DATA_MAP["ni"]
    
    def __call__(self, x: torch.Tensor, batch_bbox: list[dict[str, Any]]) -> torch.Tensor:
        B, C, H, W = x.shape

        batch_patch_metadata = []

        # for every image in a batch
        for b in range(B):
            # for every patch in a row
            bboxes: list = batch_bbox[b]["boxes"]
            labels: list = batch_bbox[b]["labels"]
            image_patch_metadata = []
            for y_cor in range(0, H, self.patch_size):
                # for ever patch in a column
                for x_cor in range(0, W, self.patch_size):
                    top_left: tuple[int, int] = (x_cor, y_cor)
                    bottom_right: tuple[int, int] = (x_cor+self.patch_size, y_cor+self.patch_size)
                    class_sum: int = self._get_intersections(
                        bboxes,
                        labels,
                        top_left,
                        bottom_right
                    )
                    image_patch_metadata.append(class_sum)
            batch_patch_metadata.append(image_patch_metadata)

        # a tensor containing the class information for every patch in every image in the batch
        # shape [B, N] where N is the corresponding class to every patch
        batch_md_tensor = torch.tensor(batch_patch_metadata, device=x.device, dtype=torch.long)
        assert batch_md_tensor.shape[0] == B
        return batch_md_tensor