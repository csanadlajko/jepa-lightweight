import torch
from ..utils.masking import apply_mask
from ..utils.ema import _ema_update
import torch.nn.functional as F
from typing import Any

def train_pdl1(
        teacher_mod, 
        student_mod, 
        loader, 
        optim_student, 
        optim_predictor, 
        predictor, 
        momentum, 
        ijepa_loss,
        device,
        cell_mask,
        cell_percentage=20
    ):
    
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
            teacher_target_tokens = apply_mask(teacher_tokens, target_mask_indices, predictor=True, use_padding=True)


        ## create context student tokens
        student_tokens = student_mod(images, masks=context_mask_indices, cls=False, cell_mask=True)

        predicted_target_tokens = predictor(
            student_tokens, 
            context_mask_indices, 
            target_mask_indices, 
            target_patch_labels,
            multimodal=False, 
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

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    print(f"avg. training loss this epoch: {avg_loss:.4f}")

def train_cell_predictor(
        student_mod, 
        loader,
        optim_predictor,
        cell_predictor,
        device,
        cell_mask,
        patch_processer,
        loss_fn,
        cell_percentage=20
    ):
    student_mod.eval()
    cell_predictor.train()

    total_loss = 0.0
    num_batches = 0
    running_acc = 0.0

    for (images, annotation) in loader:
        ## annotation -> len(batch_size) list containing annotations for each image
        images = images.to(device)

        ## acquire context and target cell masks for current batch (patch indices)
        ## size: len(batch)
        mask_meta: list[dict[str, Any]] = cell_mask(cell_percentage, annotation)
        patch_meta: torch.Tensor = patch_processer(images, annotation)

        target_mask_indices: list[list[torch.Tensor]] = mask_meta["target_patch_indices"]
        context_mask_indices = mask_meta["context_patch_indices"]

        student_tokens = student_mod(images, masks=context_mask_indices, cell_mask=True)

        ## average loss of all predicted target values
        loss_all, accuracy = cell_predictor(
            student_tokens,
            target_mask_indices,
            patch_meta,
            loss_fn
        )

        optim_predictor.zero_grad()

        loss_all.backward()

        optim_predictor.step()

        total_loss += loss_all.item()
        num_batches += 1
        running_acc += accuracy

    avg_loss = total_loss / num_batches
    avg_acc = running_acc / num_batches
    print(f"=== running accuracy this cell prediction epoch: {avg_acc:.2f}% ===")
    print(f"=== avg loss this cell prediction epoch: {avg_loss:.4f} ===")

def eval_cell_predictor(
    student_model,
    cell_predictor,
    test_loader,
    device,
    cell_mask,
    patch_processer,
    loss_fn,
    cell_percentage
    ):
    student_model.eval()
    cell_predictor.eval()

    for (images, annotation) in test_loader:
        images = images.to(device)

        mask_meta: list[dict[str, Any]] = cell_mask(cell_percentage, annotation)
        patch_meta: torch.Tensor = patch_processer(images, annotation)

        target_mask_indices: list[list[torch.Tensor]] = mask_meta["target_patch_indices"]
        context_mask_indices = mask_meta["context_patch_indices"]

        with torch.no_grad():
            student_tokens = student_model(images, masks=context_mask_indices, cell_mask=True)

            ## average loss of all predicted target values
            loss_all, accuracy = cell_predictor(
                student_tokens,
                target_mask_indices,
                patch_meta,
                loss_fn
            )

        print(f"=== Running eval accuracy: {accuracy:.2f}% ===")
        print(f"=== Running eval loss: {(sum(loss_all) / len(loss_all)):.4f} ===")