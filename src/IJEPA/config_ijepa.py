import torch
import torch.nn as nn

def get_model_config(student_model, predictor, learning_rate, epochs):

    optim_student = torch.optim.AdamW(
        student_model.parameters(),
        lr=learning_rate,
        weight_decay=0.05
    )

    optim_predictor = torch.optim.AdamW(
        predictor.parameters(),
        lr=learning_rate,
        weight_decay=0.05
    )

    optim_cls = torch.optim.AdamW(
        predictor.parameters(),
        lr=learning_rate,
        weight_decay=0.05
    )

    student_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim_student,
        epochs
    )

    return {
        "optim_student": optim_student,
        "optim_cls": optim_cls,
        "optim_predictor": optim_predictor,
        "ijepa_loss": nn.MSELoss(),
        "cls_loss": nn.CrossEntropyLoss(),
        "student_scheduler": student_scheduler
    }