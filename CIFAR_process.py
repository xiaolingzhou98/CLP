import copy
import numpy as np
import data.cifar as cifar
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def uniform_corruption(corruption_ratio, num_classes):
    eye = np.eye(num_classes)
    noise = np.full((num_classes, num_classes), 1/num_classes)
    corruption_matrix = eye * (1 - corruption_ratio) + noise * corruption_ratio
    return corruption_matrix


def flip1_corruption(corruption_ratio, num_classes):
    corruption_matrix = np.eye(num_classes) * (1 - corruption_ratio)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        corruption_matrix[i][np.random.choice(row_indices[row_indices != i])] = corruption_ratio
    return corruption_matrix


def flip2_corruption(corruption_ratio, num_classes):
    corruption_matrix = np.eye(num_classes) * (1 - corruption_ratio)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        corruption_matrix[i][np.random.choice(row_indices[row_indices != i], 2, replace=False)] = corruption_ratio / 2
    return corruption_matrix


def build_dataloader(
        seed=1,
        dataset='cifar10',
        num_meta_total=1000,
        imbalanced_factor=None,
        corruption_type=None,
        corruption_ratio=0.,
        batch_size=100,
):
    noisy_indices = None
    np.random.seed(seed)
    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
    )

    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    dataset_list = {
        'cifar10': cifar.CIFAR10,
        'cifar100': cifar.CIFAR100,
    }

    corruption_list = {
        'uniform': uniform_corruption,
        'flip1': flip1_corruption,
        'flip2': flip2_corruption,
    }

    if dataset == 'cifar10':
        
        train_dataset = dataset_list[dataset](root='', train=True, download=False, transform=train_transforms)
        test_dataset = dataset_list[dataset](root='', train=False, transform=test_transforms)
    
    if dataset == 'cifar100':
        train_dataset = dataset_list[dataset](root='', train=True, download=False, transform=train_transforms)
        test_dataset = dataset_list[dataset](root='', train=False, transform=test_transforms)
        


    num_classes = len(train_dataset.classes)
    num_meta = int(num_meta_total / num_classes)

    index_to_meta = []
    index_to_train = []

    if imbalanced_factor is not None:
        imbalanced_num_list = []
        sample_num = int((len(train_dataset.targets) - num_meta_total) / num_classes)
        for class_index in range(num_classes):
            imbalanced_num = sample_num / (imbalanced_factor ** (class_index / (num_classes - 1)))
            imbalanced_num_list.append(int(imbalanced_num))
        np.random.shuffle(imbalanced_num_list)
        print(imbalanced_num_list)
    else:
        imbalanced_num_list = None

    for class_index in range(num_classes):
        index_to_class = [index for index, label in enumerate(train_dataset.targets) if label == class_index]
        np.random.shuffle(index_to_class)
        index_to_meta.extend(index_to_class[:num_meta])
        index_to_class_for_train = index_to_class[num_meta:]

        if imbalanced_num_list is not None:
            index_to_class_for_train = index_to_class_for_train[:imbalanced_num_list[class_index]]

        index_to_train.extend(index_to_class_for_train)

    meta_dataset = copy.deepcopy(train_dataset)
    train_dataset.data = train_dataset.data[index_to_train]
    train_dataset.targets = list(np.array(train_dataset.targets)[index_to_train])
    meta_dataset.data = meta_dataset.data[index_to_meta]
    meta_dataset.targets = list(np.array(meta_dataset.targets)[index_to_meta])

    if corruption_type is not None:
        corruption_matrix = corruption_list[corruption_type](corruption_ratio, num_classes)
        print(corruption_matrix)
        noisy_indices = []
        for index in range(len(train_dataset.targets)):
            p = corruption_matrix[train_dataset.targets[index]]
            new_label = np.random.choice(num_classes, p=p)
            if new_label != train_dataset.targets[index]:  # Check if noise is introduced
                noisy_indices.append(index)
            train_dataset.targets[index] = new_label
            
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True,num_workers=4)
    meta_dataloader = DataLoader(meta_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True,num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, pin_memory=True,num_workers=4)

    return train_dataloader, meta_dataloader, test_dataloader, imbalanced_num_list, noisy_indices