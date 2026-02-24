import torch
from ..utils.masking import apply_mask
from ..utils.ema import _ema_update
import torch.nn.functional as F

def train_pdl1(teacher_mod, 
          student_mod, 
          loader, 
          optim_student, 
          optim_predictor, 
          predictor, 
          momentum, 
          ijepa_loss,
          device,
          cell_mask,
          multimodal=True,
          cell_percentage=20):
    
    teacher_mod.eval()
    student_mod.train()
    predictor.train()

    total_loss = 0.0
    num_batches = 0

    for (images, annotation) in loader:
        ## annotation -> len(batch_size) list containing annotations for each image
        images = images.to(device)

        ## acquire context and target cell masks for current batch (patch indices)
        ## size: len(batch)
        mask_meta = cell_mask(cell_percentage, annotation)

        target_mask_indices = mask_meta["target_patch_indices"]
        context_mask_indices = mask_meta["context_patch_indices"]
        target_patch_labels = mask_meta["target_patch_labels"]
        context_patch_labels = mask_meta["context_patch_labels"]


        with torch.no_grad():
            ## create teacher tokens for full image
            teacher_tokens = teacher_mod(images, cls=False)
            teacher_tokens = F.layer_norm(teacher_tokens, (teacher_tokens.size(-1),))
            teacher_target_tokens = apply_mask(teacher_tokens, target_mask_indices, predictor=True)


        ## create context student tokens
        student_tokens = student_mod(images, masks=context_mask_indices, cls=False, cell_mask=True)

        ## label handling should be the following:
        ## every target patch / block should receive multimodality
        ## but that would not be computationally efficient at all (in case of more targets it takes a lot of time)


        predicted_target_tokens = predictor(
            student_tokens, 
            context_mask_indices, 
            target_mask_indices, 
            target_patch_labels,
            multimodal, 
            return_cls_only=False,
            cell_mask=True
        )
        
        optim_student.zero_grad()
        optim_predictor.zero_grad()
        
        loss_curr = ijepa_loss(predicted_target_tokens, teacher_target_tokens)
            
        loss_curr.backward()
        
        optim_student.step()
        optim_predictor.step()
        
        _ema_update(teacher_mod, student_mod, momentum)
        
        total_loss += loss_curr.item()
        num_batches += 1
    
    print("=== Epoch ended ===")
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    print(f"avg. training loss this epoch: {avg_loss:.4f}")