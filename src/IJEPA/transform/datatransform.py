from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import json
from src.IJEPA.transform.mri_dataprocess import MRIImageDataset
from src.IJEPA.transform.cifar10dot1 import CIFAR10dot1Dataset

file = open("././parameters.json")
all_params: dict[str, int] = json.load(file)

parameters = all_params["ijepa"]
mm_params = all_params["multimodal"]


transform = transforms.Compose([
    transforms.Resize((parameters["IMAGE_SIZE"], parameters["IMAGE_SIZE"])),
    transforms.RandomInvert(0.3),
    transforms.RandomHorizontalFlip(p=0.6),
    # transforms.RandomRotation(degrees=180),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                     std=[0.247, 0.243, 0.261])
])


test_transform = transforms.Compose([
    transforms.Resize((parameters["IMAGE_SIZE"], parameters["IMAGE_SIZE"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
])

def get_cifarten_dataset():
    train_data = datasets.CIFAR10("data", train=True, transform=transform, download=True)
    test_data = datasets.CIFAR10("data", transform=test_transform, train=False, download=True)

    ## batchify data

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=parameters["BATCH_SIZE"],
        shuffle=True
    )
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=parameters["BATCH_SIZE"],
        shuffle=False
    )
    return train_loader, test_loader

## LOAD MRI DATASET

def get_mri_dataset(input_folder: str):
    # inverse transform for measuting generalization
    full_dataset_train = MRIImageDataset(input_folder, transform=transform)
    full_dataset_test = MRIImageDataset(input_folder, transform=test_transform)

    train_size = int(0.8 * len(full_dataset_train))
    test_size = len(full_dataset_train) - train_size

    mri_train_dset, _ = random_split(full_dataset_train, [train_size, test_size])
    _, mri_test_dset = random_split(full_dataset_test, [train_size, test_size])

    mri_train_loader = DataLoader(
        dataset=mri_train_dset,
        batch_size=parameters["BATCH_SIZE"],
        shuffle=True
    )

    mri_test_loader = DataLoader(
        dataset=mri_test_dset,
        batch_size=parameters["BATCH_SIZE"],
        shuffle=False
    )
    return mri_train_loader, mri_test_loader

## LOAD CIFAR10.1 DATASET

def get_cifar_tendotone_dataset(input_folder):
    cifar10dot1 = CIFAR10dot1Dataset("CIFAR10dot1/cifar10.1_v6_data.npy", "CIFAR10dot1/cifar10.1_v6_labels.npy", transform=test_transform)
    cifar101_test_loader = DataLoader(
        dataset=cifar10dot1,
        batch_size=parameters["BATCH_SIZE"],
    shuffle=False
    )
    return cifar101_test_loader
