from src.parser.parser import parse_jepa_args
from src.models.vit import VisionTransformer
from src.models.predictor import ViTPredictor
from src.train.train_ijepa import (
    train, # main JEPA training loop for creating the representational space
    train_cls, # CLS token supervised training loop
    show_cls_data_per_epoch, # cls accuracy plot
    show_loss_per_epoch, # cls loss plot
    eval_cls # cls evalutaion on the test dataset
)
from src.train.train_pdl1_mm_jepa import (
    train_pdl1 ## train JEPA model on PDL1 cell dataset
)
from src.data_preprocess.dataloader import load_dataset
from src.utils.config_ijepa import get_model_config, init_weights
from src.utils.masking import Mask, CellMask
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import datetime
import torch

args = parse_jepa_args()
device = "cuda" if torch.cuda.is_available() else "cpu"

mask = Mask(device=device)
cell_mask = CellMask(device=device)

if __name__ == "__main__":
    jepa_loss_per_epoch = []
    accuracy_per_epoch = []
    cls_loss_per_epoch = []

    run_identifier: str = datetime.datetime.now().strftime("%Y-%m-%dT%H%M%SZ")
    result_folder: str = args.result_folder

    datasets = load_dataset(args.dataset, args.dataset_input, args.reverse_transform)

    if "error" in datasets:
        raise FileNotFoundError(datasets["error"])

    train_loader, test_loader = datasets["train_loader"], datasets["test_loader"]

    teacher_model = VisionTransformer(
        img_size=args.image_size,
        patch_size=args.patch_size,
        in_chans=args.channels,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        depth=args.depth,
        mlp_dim=args.mlp_dim,
        drop_rate=args.teacher_dropout,
        num_classes=args.num_classes
    ).to(device)

    student_model = VisionTransformer(
        img_size=args.image_size,
        patch_size=args.patch_size,
        in_chans=args.channels,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        depth=args.depth,
        mlp_dim=args.mlp_dim,
        drop_rate=args.student_dropout,
        num_classes=args.num_classes
    ).to(device)

    text_encoder = AutoModelForCausalLM.from_pretrained(args.sentence_encoder).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.sentence_encoder)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    predictor = ViTPredictor(
        num_patches=teacher_model.patch_embed.num_patches,
        embed_dim=args.embed_dim,
        device=device,
        pred_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        num_classes=args.num_classes
    ).to(device)

    teacher_model.apply(init_weights)
    student_model.apply(init_weights)
    predictor.apply(init_weights)

    model_config = get_model_config(
        student_model,
        predictor,
        args.lr,
        args.epochs
    )

    student_scheduler = model_config["student_scheduler"]
    
    print(f"total number of parameters approx.: {sum(p.numel() for p in student_model.parameters()) + sum(p.numel() for p in teacher_model.parameters()) + sum(p.numel() for p in predictor.parameters())}")
    print(f"Which from are trainable parameters: {sum(p.numel() for p in student_model.parameters() if p.requires_grad) + sum(p.numel() for p in teacher_model.parameters() if p.requires_grad) + sum(p.numel() for p in predictor.parameters() if p.requires_grad)}")

    if args.multimodal_run == "y":
        print(f"\nStarting training in MULTIMODAL mode for {args.epochs} epoch(s)!")
    else:
        print(f"\nStarting training normal I-JEPA mode for {args.epochs} epoch(s)!")
    
    for epoch in range(args.epochs):

        print(f"\n=== EPOCH {epoch + 1}/{args.epochs} ===")

        if args.dataset != "pdl1":
            ## train JEPA on regular classification tasks with cls token
            loss_epoch = train(
                teacher_mod=teacher_model, 
                student_mod=student_model, 
                loader=train_loader, 
                optim_student=model_config["optim_student"],
                optim_predictor=model_config["optim_predictor"],
                predictor=predictor,
                momentum=args.momentum,
                ijepa_loss=model_config["ijepa_loss"],
                device=device,
                mask=mask,
                multimodal=args.multimodal_run,
                debug=args.debug
            )
        else:
            ## train JEPA on PDL1 dataset with local representation classification
            print("setting up pdl1 training...")
            text_encoder.eval()
            loss_epoch = train_pdl1(
                teacher_mod=teacher_model, 
                student_mod=student_model, 
                loader=train_loader, 
                optim_student=model_config["optim_student"],
                optim_predictor=model_config["optim_predictor"],
                predictor=predictor,
                momentum=args.momentum,
                ijepa_loss=model_config["ijepa_loss"],
                multimodal=args.multimodal_run,
                cell_percentage=args.cell_percentage,
                device=device,
                cell_mask=cell_mask
            )

        student_scheduler.step()
        jepa_loss_per_epoch.append(loss_epoch)
    
    if args.debug == "y":
        torch.save(student_model.state_dict(), f"{result_folder}/trained_student_jepa_{run_identifier}.pth")
        torch.save(teacher_model.state_dict(), f"{result_folder}/teacher_model_jepa_{run_identifier}.pth")
        torch.save(predictor.state_dict(), f"{result_folder}/trained_predictor_jepa_{run_identifier}.pth")


    for epoch in range(args.epochs):
        print(f"\n=== CLS EPOCH {epoch+1}/{args.epochs} ===")
        cls_loss_at_epoch, accuracy_epoch = train_cls(
            student_model=student_model, 
            train_dataset=train_loader, 
            predictor=predictor,
            optim_cls=model_config["optim_cls"],
            cls_loss=model_config["cls_loss"],
            multimodal=False ## false in every case to prevent leakage !!
        )
        accuracy_per_epoch.append(accuracy_epoch)
        cls_loss_per_epoch.append(cls_loss_at_epoch)
    
    if args.debug == "y":
        torch.save(student_model.state_dict(), f"{result_folder}/trained_student_cls_{run_identifier}.pth")
        torch.save(teacher_model.state_dict(), f"{result_folder}/teacher_model_cls_{run_identifier}.pth")
        torch.save(predictor.state_dict(), f"{result_folder}/trained_predictor_cls_{run_identifier}.pth")

    print("\n=== FINAL EVALUATION ===")
    
    cls_acc = eval_cls(
        model=student_model, 
        test_dataset=test_loader, 
        predictor=predictor, 
        multimodal=False ## false in every case to prevent leakage !!
    )

    show_loss_per_epoch(jepa_loss_per_epoch, cls_loss_per_epoch, run_identifier, result_folder)
    show_cls_data_per_epoch(accuracy_per_epoch, run_identifier, result_folder)
    print(f"-- CLS token classification accuracy: {cls_acc:.4f}")