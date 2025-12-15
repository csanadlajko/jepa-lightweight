import torch
import torch.nn as nn

def get_model_config(student_model, predictor, learning_rate, epochs):

    optim_student = torch.optim.Adam(
        student_model.parameters(),
        lr=learning_rate
    )

    optim_predictor = torch.optim.Adam(
        predictor.parameters(),
        lr=learning_rate
    )

    optim_cls = torch.optim.Adam(
        predictor.parameters(),
        lr=learning_rate
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