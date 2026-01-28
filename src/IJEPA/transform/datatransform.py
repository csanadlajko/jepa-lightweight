from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch
from src.IJEPA.transform.healthcare.mri_dataprocess import MRIImageDataset
from src.IJEPA.transform.healthcare.lung_cancer_dataprocess import LungCancerDataset, PDL1Dataset
from src.IJEPA.transform.cifar10dot1 import CIFAR10dot1Dataset
from src.parser.parser import parse_jepa_args

args = parse_jepa_args()

transform = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.RandomInvert(0.3),
    transforms.RandomHorizontalFlip(p=0.6),
    # transforms.RandomRotation(degrees=180),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                     std=[0.247, 0.243, 0.261])
])


test_transform = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
])

def get_pdl1_dataset(input_dir: str, annotation_file_path: str, reverse: str = "n"):
    if reverse=="y":
        full_dataset_train = PDL1Dataset(input_dir, annotation_file_path, test_transform)
        full_dataset_test = PDL1Dataset(input_dir, annotation_file_path, transform)
    else:
        full_dataset_train = PDL1Dataset(input_dir, annotation_file_path, transform)
        full_dataset_test = PDL1Dataset(input_dir, annotation_file_path, test_transform)
    
    train_size: int = int(len(full_dataset_train)*0.8)
    test_size: int = len(full_dataset_train) - train_size

    train_data, test_indices = random_split(full_dataset_train, [train_size, test_size])

    test_data = torch.utils.data.Subset(full_dataset_test, test_indices.indices)

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=PDL1Dataset.collate_fn
    )
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=PDL1Dataset.collate_fn
    )
    return train_loader, test_loader

def get_lung_cancer_dataset(input_dir: str, reverse: str = "n"):
    if reverse=="y":
        full_dataset_train = LungCancerDataset(input_dir, test_transform)
        full_dataset_test = LungCancerDataset(input_dir, transform)
    else:
        full_dataset_train = LungCancerDataset(input_dir, transform)
        full_dataset_test = LungCancerDataset(input_dir, test_transform)
    
    train_size: int = int(len(full_dataset_train)*0.8)
    test_size: int = len(full_dataset_train) - train_size

    train_data, test_indices = random_split(full_dataset_train, [train_size, test_size])

    test_data = torch.utils.data.Subset(full_dataset_test, test_indices.indices)

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        shuffle=False
    )
    return train_loader, test_loader

def get_cifarten_dataset(reverse: str = "n"):
    if reverse=="y":
        train_data = datasets.CIFAR10("data", train=True, transform=test_transform, download=True)
        test_data = datasets.CIFAR10("data", transform=transform, train=False, download=True)
    else:
        train_data = datasets.CIFAR10("data", train=True, transform=transform, download=True)
        test_data = datasets.CIFAR10("data", transform=test_transform, train=False, download=True)

    ## batchify data

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        shuffle=False
    )
    return train_loader, test_loader

## LOAD MRI DATASET

def get_mri_dataset(input_folder: str, reverse: str = "n"):
    # inverse transform for measuting generalization
    if reverse=="y":
        full_dataset_train = MRIImageDataset(input_folder, transform=transform)
        full_dataset_test = MRIImageDataset(input_folder, transform=test_transform)
    else:
        full_dataset_train = MRIImageDataset(input_folder, transform=test_transform)
        full_dataset_test = MRIImageDataset(input_folder, transform=transforms)

    train_size = int(0.8 * len(full_dataset_train))
    test_size = len(full_dataset_train) - train_size

    mri_train_dset, test_indices = random_split(full_dataset_train, [train_size, test_size])
    mri_test_dset = torch.utils.data.Subset(full_dataset_test, test_indices.indices)

    mri_train_loader = DataLoader(
        dataset=mri_train_dset,
        batch_size=args.batch_size,
        shuffle=True
    )

    mri_test_loader = DataLoader(
        dataset=mri_test_dset,
        batch_size=args.batch_size,
        shuffle=False
    )
    return mri_train_loader, mri_test_loader

## LOAD CIFAR10.1 DATASET

def get_cifar_tendotone_dataset(input_folder: str):
    cifar10dot1 = CIFAR10dot1Dataset("CIFAR10dot1/cifar10.1_v6_data.npy", "CIFAR10dot1/cifar10.1_v6_labels.npy", transform=test_transform)
    cifar101_test_loader = DataLoader(
        dataset=cifar10dot1,
        batch_size=args.batch_size,
    shuffle=False
    )
    return cifar101_test_loader
