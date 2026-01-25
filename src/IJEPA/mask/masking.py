import math
from multiprocessing import Value
import torch
from src.parser.parser import parse_jepa_args
from typing import Any

args = parse_jepa_args()

DEBUG = True if args.debug == "y" else False

device = "cuda" if torch.cuda.is_available() else "cpu"

class CellMask(object):

    """
    Mask a cell by the following logic: \n
        - Check patch index of (x, y) by bounding box,
        - Check patch index of (x, y - h) by bounding box,
        - Check patch index of (x + w, y) by bounding box,
        - Check patch index of (x + w, y - h) by bounding box.\n
    Afterwards mask these patches as they will give the local representations.
    """

    def __init__(
            self,
            bbox_list: list[list[Any]], ## list of boundig boxes per image [x, y, w, h]
            input_size=(args.image_size, args.image_size),
            patch_size=args.patch_size,
            ctx_mask_scale=(0.2, 0.8)
        ):
        if isinstance(input_size, int):
            input_size = (input_size, input_size)

        self.height = input_size[0] // patch_size
        self.width = input_size[1] // patch_size
        self.patch_size = patch_size
        self.ctx_mask_scale = ctx_mask_scale
        self.bbox_list = bbox_list
        self.num_pathes = self.height * self.width

    def _get_patch_indices_by_coordinates(
            self, 
            bbox: list[list[int]] ## list of bboxes -> in case of target its about 20% og the total size
        ) -> list[int]:
        
        """
        Returns the flattened patch indices where the bounding box overlaps.
        """

        patch_indices = set()

        for box in bbox:
            x, y, w, h = box ## x, y top left; w, h width-height

            ## transform from regular coordinates to patch coordinates
            top_left_patch_index = (x // self.patch_size, y // self.patch_size)
            top_right_patch_index = ((x+w)//self.patch_size, y // self.patch_size)
            bottom_left_patch_index = (x // self.patch_size, (y+h) // self.patch_size)
            bottom_right_patch_index = ((x+w) // self.patch_size, (y+h) // self.patch_size)

            ## transform from patch coordinates to flattened patch index
            flattened_top_left = (self.width * top_left_patch_index[1]) + top_left_patch_index[0]
            flattened_top_right = (self.width * top_right_patch_index[1]) + top_right_patch_index[0]
            flattened_bottom_left = (self.width * bottom_left_patch_index[1]) + bottom_left_patch_index[0]
            flattened_bottom_right = (self.width * bottom_right_patch_index[1]) + bottom_right_patch_index[0]

            # for y in range(fla)

            ## using set because duplication is not allowed
            patch_indices.update((flattened_top_left, flattened_top_right, flattened_bottom_left, flattened_bottom_right))

        return list(patch_indices)
    
    def __call__(self, batch):
        pass


class Mask(object):
    
    def __init__(
        self,
        input_size=(args.image_size, args.image_size),
        patch_size=args.patch_size,
        nctx=1,
        ntarg=args.num_target,
        targ_mask_scale=(0.05, 0.1),
        ctx_mask_scale=(0.2, 0.8),
        aspect_ratio=(0.75, 1.5),
        min_keep=4,
        max_tries=200
    ):
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.height = input_size[0] // patch_size
        self.width = input_size[1] // patch_size
        self.patch_size = patch_size
        self.nctx = nctx
        self.ntarg = ntarg
        self.targ_mask_scale = targ_mask_scale
        self.ctx_mask_scale = ctx_mask_scale
        self.aspect_ratio = aspect_ratio
        self.min_keep = min_keep
        self.max_tries = max_tries
        self.iteration_counter = Value('i', -1)
        
    def step(self):
        """
        Increments ``i`` for every image, resulting in different context and target blocks.
        """
        i = self.iteration_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v
        
    def _sample_block_size(self, generator, scale, ascpect_ratio_scale):
        u = torch.rand(1, generator=generator).item()
        
        # block size scale
        min_s, max_s = scale 
        
        # ratio of block size / total patch number
        frac = min_s + u * (max_s - min_s)
        
        # number of patches in the block
        num_patches = int(self.height*self.width * frac)
        
        # provide the minimum amount of patches per block
        num_patches = max(num_patches, self.min_keep)
        
        v = torch.rand(1, generator=generator).item()
        min_asp_ratio, max_asp_ratio = ascpect_ratio_scale
        aspect_ratio = min_asp_ratio + v * (max_asp_ratio - min_asp_ratio)
        
        h = int(round(math.sqrt((num_patches * aspect_ratio))))
        w = int(round(math.sqrt((num_patches / aspect_ratio))))
        h = min(h, self.height - 1)
        w = min(w, self.width - 1)
        h = max(h, 1)
        w = max(w, 1)
        return h, w
    
    def _place_block_without_overlap(self, h, w, occ):
        H, W = occ.shape
        tries = 0
        found_target = 0
        total_target = h*w
        while found_target <= total_target or tries > self.max_tries:
            top = torch.randint(0, H - h + 1, (1,)).item()
            left = torch.randint(0, W- w + 1, (1,)).item()
            cut_region = occ[top:top+h, left:left+w]

            if torch.count_nonzero(cut_region) == 0:
                mask = torch.zeros((H, W), dtype=torch.int32)
                mask[top:top+h, left:left+w] = 1
                occ[top:top+h, left:left+w] = 1
                idx = torch.nonzero(mask.flatten(), as_tuple=False).squeeze()
                found_target = idx.numel()

                if idx.numel() == total_target:
                    return idx, occ
            
            tries += 1
        
        idx = torch.randperm(H*W)[:max(self.min_keep, h*w // 2)]
        
        return idx, occ
    
    def __call__(self, batch, id_only=True):
        B = len(batch)
        if isinstance(batch, torch.Tensor):
            collated_batch = batch
        else: collated_batch = torch.utils.data.default_collate(batch)
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        
        ctx_h, ctx_w = self._sample_block_size(g, self.ctx_mask_scale, (1., 1.))
        target_h, target_w = self._sample_block_size(g, self.targ_mask_scale, self.aspect_ratio)
        
        all_mask_ctx, all_mask_target = [], []
        
        for _ in range(B):
            occ = torch.zeros((self.height, self.width), dtype=torch.int32)
            
            target_mask = []
            for _ in range(self.ntarg):
                idx, occ = self._place_block_without_overlap(target_h, target_w, occ)
                idx = idx.to(device)
                target_mask.append(idx)
            
            free = (occ == 0).to(torch.int32)
            
            tries = 0
            cmask = None
            
            while tries < self.max_tries:
                top = torch.randint(0, self.height - ctx_h + 1, (1,)).item()
                left = torch.randint(0, self.width - ctx_w + 1, (1,)).item()
                region = free[top:top+ctx_h, left:left+ctx_w]
                if torch.all(region == 1):
                    cmask2d = torch.zeros((self.height, self.width), dtype=torch.int32)
                    cmask2d[top:top+ctx_h, left:left+ctx_w] = 1
                    cmask = torch.nonzero(cmask2d.flatten(), as_tuple=False).squeeze()
                    break
                tries += 1
            
            free_idx = torch.nonzero(free.flatten(), as_tuple=False).view(-1)
            target_size = ctx_h * ctx_w + 1
            
            if cmask is None:
                if free_idx.numel() == 0:
                    cmask = torch.randperm(self.height * self.width)[:target_size]
                else:
                    perm = torch.randperm(free_idx.numel())
                    take = min(target_size, free_idx.numel())
                    cmask = free_idx[perm[:take]]
                    
            current_size = cmask.numel() if cmask is not None and cmask.dim() > 0 else 0
            needed = target_size - current_size
            
            while needed > 0 and free_idx.numel() > 0:
                if cmask is not None and cmask.numel() > 0:
                    free_idx = free_idx[~torch.isin(free_idx, cmask)]
                
                if free_idx.numel() == 0:
                    break
                
                perm = torch.randperm(free_idx.numel())
                take = min(needed, free_idx.numel())
                additional = free_idx[perm[:take]]
                
                if cmask is None:
                    cmask = additional
                else:
                    cmask = torch.cat([cmask, additional])
                
                needed -= take
                    
            all_mask_target.append(target_mask)
            cmask = cmask.to(device)
            all_mask_ctx.append(cmask)
            
        if id_only:
            return all_mask_ctx, all_mask_target
        
        masked_ctx_batch = batch.clone()
        masked_target_batch = batch.clone()
        
        for i in range(B):
            ctx_idx = all_mask_ctx[i][0] + 1
            targ_id_list = all_mask_target[i]
            
            target_idx = torch.cat([idx for idx in targ_id_list]) + 1
            masked_ctx_batch[i, target_idx, :] = 0  # Mask target tokens
            
            masked_target_batch[i, ctx_idx, :] = 0  # Mask context tokens
            
        return collated_batch, masked_ctx_batch, masked_target_batch
    
def apply_mask(x, mask_indices: list[torch.Tensor], predictor=False):
    if isinstance(mask_indices, list):
        all_masked_tokens = []
        for i, mask_idx in enumerate(mask_indices):
            if isinstance(mask_idx, list):
                # enter when selecting target blocks
                all_idx = torch.cat(mask_idx).to(device)
                conc=False # dont concat when using teacher, only use N patches for masking
            else:
                # enter when selecting context blocks
                all_idx = mask_idx.to(device)
                conc=True # concat when selecting context blocks for vit and vit predictor
            if all_idx.numel() > 0:
                if not predictor:
                    patch_idx = all_idx+1 ## shift indices to right because cls
                else:
                    patch_idx = all_idx ## dont shift when using predictor -> not predicting cls token
                ## concat cls when working with context blocks otherwise only shift indices to right
                indices = torch.cat([torch.tensor([0], device=device), patch_idx]) if conc==True else patch_idx
                masked_tokens = x[i:i+1].index_select(1, indices) ## needed for cls token
                all_masked_tokens.append(masked_tokens)
        return torch.cat(all_masked_tokens, dim=0).to(device)
    else:
        return x.index_select(1, mask_indices)
