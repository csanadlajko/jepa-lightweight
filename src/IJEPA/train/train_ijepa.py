from src.IJEPA.mask.masking import Mask
from src.IJEPA.mask.masking import apply_mask
import torch
import torch.nn.functional as F
import datetime
import matplotlib.pyplot as plt
from torchviz import make_dot

run_identifier: str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

device = "cuda" if torch.cuda.is_available() else "cpu"

mask = Mask()

@torch.no_grad()
def _ema_update(teacher_mod, student_mod, momentum=0.996):
    for t_param, s_param in zip(teacher_mod.parameters(), student_mod.parameters()):
        t_param.data.mul_(momentum).add_(s_param.data, alpha=1.0 - momentum)

def train(teacher_mod, 
          student_mod, 
          loader, 
          optim_student, 
          optim_predictor, 
          predictor, 
          momentum, 
          ijepa_loss,
          multimodal=True,
          debug="n"):
    teacher_mod.eval()
    student_mod.train()
    predictor.train()

    print("---STARTING EPOCH---")
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        
        context_masks, target_masks = mask(images) # only indices -> exclude cls
        
        with torch.no_grad():
            teacher_tokens = teacher_mod(images)
            teacher_tokens = F.layer_norm(teacher_tokens, (teacher_tokens.size(-1),))
            teacher_target_tokens = apply_mask(teacher_tokens, target_masks)

        student_tokens = student_mod(images, masks=context_masks)

        # if debug == "y":
        #     make_dot(student_tokens, params=dict(student_mod.named_parameters())).render(filename="model_vis", directory="results", format="png")

        predicted_target_tokens = predictor(student_tokens, context_masks, target_masks, labels, multimodal, return_cls_only=False)
        
        optim_student.zero_grad()
        optim_predictor.zero_grad()
        
        loss_curr = ijepa_loss(predicted_target_tokens, teacher_target_tokens)
            
        loss_curr.backward()
        
        optim_student.step()
        optim_predictor.step()
        
        _ema_update(teacher_mod, student_mod, momentum)
        
        total_loss += loss_curr.item() 
        num_batches += 1

    print("---EPOCH ENDED---")
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    print(f"Average training loss: {avg_loss:.4f}")

    return avg_loss

def train_cls(student_model, 
              train_dataset, 
              predictor, 
              optim_cls, 
              cls_loss, 
              multimodal=False):
    student_model.eval() # freeze trained student model
    predictor.train()
    
    print("---STARTING CLS TRAINING---")
    total_loss = 0.0
    num_batches = 0
    correct_predictions = 0
    total_predictions = 0

    for name, param in student_model.named_parameters():
        if "cls_fc" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    for batch_idx, (images, labels) in enumerate(train_dataset):
        images = images.to(device)
        labels = labels.to(device)

        ctx_mask, trgt_masks = mask(images)
        
        # student mask embeddings from trained model
        student_enc = student_model(images, masks=ctx_mask)

        # cls tokens for the predicted full image -> not training cls from context embeddings
        pred_classes = predictor(student_enc, ctx_mask, trgt_masks, labels, multimodal, return_cls_only=True)

        optim_cls.zero_grad()

        loss = cls_loss(pred_classes, labels)
        loss.backward()
        optim_cls.step()

        _, predicted = torch.max(pred_classes, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 500 == 0:
            current_acc = correct_predictions / total_predictions
            print(f"CLS Loss at batch {batch_idx}: {loss.item():.4f}, Accuracy: {current_acc:.4f}")

    current_acc = correct_predictions / total_predictions
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    print(f"Average CLS training loss: {avg_loss:.4f}")
    print("---CLS TRAINING ENDED---")

    return avg_loss, current_acc * 100

def eval_cls(model, 
            test_dataset,
            predictor,
            multimodal=False):
    """
    Evaluate the model using CLS token classification
    """
    model.eval()
    predictor.eval()
    total_correct = 0
    total_samples = 0
    
    print("---STARTING CLS EVALUATION---")
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_dataset):
            images = images.to(device)
            labels = labels.to(device)

            ctx_masks, target_masks = mask(images)
            
            student_embeddings = model(images, masks=ctx_masks)

            pred_classes = predictor(student_embeddings, ctx_masks, target_masks, labels, multimodal, return_cls_only=True)

            _, predicted = torch.max(pred_classes, 1)
            
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
            if batch_idx % 50 == 0:
                current_acc = total_correct / total_samples
                print(f"Batch {batch_idx}, Current accuracy: {current_acc:.4f}")
            
    
    final_accuracy = total_correct / total_samples
    print(f"---CLS EVALUATION ENDED---")
    print(f"Final CLS accuracy: {final_accuracy:.4f}")
    return final_accuracy

def show_loss_per_epoch(jepa_loss_epoch_list: list[int], cls_loss_per_epoch: list[int], run_id: str, result_folder: str):
    epoch_list = range(1, len(jepa_loss_epoch_list) + 1)
    plt.figure(figsize=(8,5))
    plt.plot(epoch_list, jepa_loss_epoch_list, label="MSE loss per JEPA epoch")
    plt.plot(epoch_list, cls_loss_per_epoch, label="CE loss per CLS epochs")
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Loss over epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{result_folder}/jepa_loss_plot_{run_id}.png', dpi=500)
    plt.show()

def show_cls_data_per_epoch(accuracy_per_epoch: list[int], run_id: str, result_folder: str):
    epoch_list = range(1, len(accuracy_per_epoch) + 1)
    plt.figure(figsize=(8,5))
    plt.plot(epoch_list, accuracy_per_epoch, label="Accuracy per epoch")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy per epoch (%)')
    plt.title("CLS accuracy per epoch (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{result_folder}/cls_accuracy_plot_{run_id}.png', dpi=300)
    plt.show()
