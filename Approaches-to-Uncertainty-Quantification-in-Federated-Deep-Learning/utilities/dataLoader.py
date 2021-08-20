import torchvision
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


def load_mnist_datasets():
    transform = transforms.Compose([
        # you can add other transformations in this list
        transforms.ToTensor()
    ])
    MNIST_train = torchvision.datasets.MNIST("../../data", train=True, transform=transform, target_transform=None, download=True)
    MNIST_test = torchvision.datasets.MNIST("../../data", train=False, transform=transform, target_transform=None, download=True)
    return MNIST_train, MNIST_test


def load_fmnist_datasets():
    transform = transforms.Compose([
        # you can add other transformations in this list
        transforms.ToTensor()
    ])
    FMNIST_train = torchvision.datasets.FashionMNIST("../../data", train=True, transform=transform, target_transform=None, download=True)
    FMNIST_test = torchvision.datasets.FashionMNIST("../../data", train=False, transform=transform, target_transform=None, download=True)
    return FMNIST_train, FMNIST_test


def load_kmnist_datasets():
    transform = transforms.Compose([
        # you can add other transformations in this list
        transforms.ToTensor()
    ])
    KMNIST_train = torchvision.datasets.KMNIST("../../data", train=True, transform=transform, target_transform=None, download=True)
    KMNIST_test = torchvision.datasets.KMNIST("../../data", train=False, transform=transform, target_transform=None, download=True)
    return KMNIST_train, KMNIST_test


def load_cifar_10():
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='../../data', train=True,
                                        download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='../../data', train=False,
                                           download=True, transform=transform)
    return trainset, testset


def load_svhn():
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.SVHN(root='../../data', split='train',
                                        download=True, transform=transform)
    testset = torchvision.datasets.SVHN(root='../../data', split='test',
                                           download=True, transform=transform)
    return trainset, testset


def load_data_and_datasets(amount, dataset="MNIST"):
    if dataset=="MNIST":
        train, test = load_mnist_datasets()
        return create_subsets(train, amount)
    elif dataset == "FMNIST":
        train, test = load_fmnist_datasets()
        return create_subsets(train, amount)
    elif dataset == "KMNIST":
        train, test = load_kmnist_datasets()
        return create_subsets(train, amount)
    elif dataset == "CIFAR10":
        train, test = load_cifar_10()
        return create_subsets(train, amount)
    elif dataset == "SVHN":
        train, test = load_svhn()
        return create_subsets(train, amount)
    else:
        print("Dataset not supported yet!")


def create_subsets(dataset, amount):
    print("amount:" + str(amount))
    size_dataset = len(dataset.data)
    size_per_dataset = int(size_dataset/amount)
    all_datasets = []
    for i in range(amount):
        index = list(range(i*size_per_dataset, (i+1)*size_per_dataset, 1))
        trainset = torch.utils.data.Subset(dataset, index)
        all_datasets.append(trainset)
    print("Created " + str(amount) + " Datasets with " + str(len(all_datasets[0])) + " images each")
    return all_datasets


def load_test_set(dataset="MNIST"):
    if dataset == "MNIST" :
        train, test = load_mnist_datasets()
    elif dataset == "FMNIST":
        train, test = load_fmnist_datasets()
    elif dataset == "KMNIST":
        train, test = load_kmnist_datasets()
    elif dataset == "CIFAR10":
        train, test = load_cifar_10()
    elif dataset == "SVHN":
        train, test = load_svhn()
    else:
        print("Dataset not supported yet!")
        test = None
    return test

