from typing import List, Union      # Union 类型可以用于表示参数或函数返回值等多种情况下可能的不同类型

import os
import copy
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
from lacuna import Lacuna10, Lacuna100, Small_Lacuna10, Small_Binary_Lacuna10, Small_Lacuna5,Small_Lacuna6
from Small_CIFAR10 import Small_CIFAR10, Small_Binary_CIFAR10, Small_CIFAR5, Small_CIFAR6
from Small_MNIST import Small_MNIST, Small_Binary_MNIST
from TinyImageNet import TinyImageNet_pretrain, TinyImageNet_finetune, TinyImageNet_finetune5
from IPython import embed   # 将IPython嵌入到Python代码的命令空间中

def manual_seed(seed):
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)     # 设置CPU生成随机数的种子
    torch.backends.cudnn.deterministic = True     # True表示每次返回的卷积算法将是确定的
    torch.backends.cudnn.benchmark = False        # True在每次输入时，会使得cuDNN来衡量自己库里面的多个卷积算法的速度，然后选择其中最快的那个卷积算法。输入大小若变化，则会重新配置，减缓训练速度

_DATASETS = {}

def _add_dataset(dataset_fn):   # 将数据集加入到_DATASETS中
    _DATASETS[dataset_fn.__name__] = dataset_fn
    return dataset_fn

def _get_mnist_transforms(augment=True):
    transform_augment = transforms.Compose([
        transforms.Pad(padding=2),     # 边界各填充2个像素，默认填充为黑色
        transforms.RandomCrop(32, padding=4),   # 图像修正为（32,32），边界填充4个像素，变为（40,40）
        transforms.ToTensor(),         # 将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1] 注意事项：归一化至[0-1]是直接除以255
    ])
    transform_test = transforms.Compose([
        transforms.Pad(padding=2),
        transforms.ToTensor()
    ])
    transform_train = transform_augment if augment else transform_test

    return transform_train, transform_test

def _get_lacuna_transforms(augment=True):
    transform_augment = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.Resize(size=(32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.382, 0.420, 0.502), (0.276, 0.279, 0.302)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(size=(32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.382, 0.420, 0.502), (0.276, 0.279, 0.302)),
    ])
    transform_train = transform_augment if augment else transform_test

    return transform_train, transform_test

def _get_cifar_transforms(augment=True):
    transform_augment = transforms.Compose([
        transforms.Pad(padding=4, fill=(125,123,113)),      # 默认RGB三通道分别填充（125,123,113）的Pad
        transforms.RandomCrop(32, padding=0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_train = transform_augment if augment else transform_test

    return transform_train, transform_test

def _get_imagenet_transforms(augment=True):
    transform_augment = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.Resize(size=(32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(size=(32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transform_train = transform_augment if augment else transform_test

    return transform_train, transform_test

def _get_mix_transforms(augment=True):    # 参数与之前_get_cifar_transforms一致
    transform_augment = transforms.Compose([
        transforms.Resize(size=(32, 32)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_train = transform_augment if augment else transform_test

    return transform_train, transform_test

@_add_dataset   
def cifar10(root, augment=False):         # 加载全部cifar10训练集和测试集
    transform_train, transform_test = _get_cifar_transforms(augment=augment)   # 若为False，则transform_train 与 transform_test 相同
    train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    test_set  = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    return train_set, test_set

@_add_dataset   
def small_cifar5(root, augment=False):    # 加载0-4类的cifar10训练集和测试集
    transform_train, transform_test = _get_cifar_transforms(augment=augment)   # 若为False，则transform_train 与 transform_test 相同
    train_set = Small_CIFAR5(root=root, train=True, transform=transform_train)
    test_set  = Small_CIFAR5(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset   
def small_cifar6(root, augment=False):
    transform_train, transform_test = _get_cifar_transforms(augment=augment)
    train_set = Small_CIFAR6(root=root, train=True, transform=transform_train)
    test_set  = Small_CIFAR6(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset   
def small_cifar10(root, augment=False):
    transform_train, transform_test = _get_cifar_transforms(augment=augment)
    train_set = Small_CIFAR10(root=root, train=True, transform=transform_train)
    test_set  = Small_CIFAR10(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset   
def small_binary_cifar10(root, augment=False):
    transform_train, transform_test = _get_cifar_transforms(augment=augment)
    train_set = Small_Binary_CIFAR10(root=root, train=True, transform=transform_train)
    test_set  = Small_Binary_CIFAR10(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset   
def cifar100(root, augment=False):
    transform_train, transform_test = _get_cifar_transforms(augment=augment)
    train_set = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform_train)
    test_set  = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform_test)
    return train_set, test_set

@_add_dataset
def mnist(root, augment=False):
    transform_train, transform_test = _get_mnist_transforms(augment=augment)
    train_set = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform_test)
    return train_set, test_set

@_add_dataset
def small_mnist(root, augment=False):
    transform_train, transform_test = _get_mnist_transforms(augment=augment)
    train_set = Small_MNIST(root=root, train=True, transform=transform_train)
    test_set = Small_MNIST(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset
def small_binary_mnist(root, augment=False):
    transform_train, transform_test = _get_mnist_transforms(augment=augment)
    train_set = Small_Binary_MNIST(root=root, train=True, transform=transform_train)
    test_set = Small_Binary_MNIST(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset
def lacuna100(root, augment=False):
    transform_train, transform_test = _get_lacuna_transforms(augment=augment)
    train_set = Lacuna100(root=root, train=True, transform=transform_train)
    test_set = Lacuna100(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset
def lacuna10(root, augment=False):
    transform_train, transform_test = _get_lacuna_transforms(augment=augment)
    train_set = Lacuna10(root=root, train=True, transform=transform_train)
    test_set = Lacuna10(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset
def small_lacuna5(root, augment=False):
    transform_train, transform_test = _get_lacuna_transforms(augment=augment)
    train_set = Small_Lacuna5(root=root, train=True, transform=transform_train)
    test_set = Small_Lacuna5(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset
def small_lacuna6(root, augment=False):
    transform_train, transform_test = _get_lacuna_transforms(augment=augment)
    train_set = Small_Lacuna6(root=root, train=True, transform=transform_train)
    test_set = Small_Lacuna6(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset
def small_lacuna10(root, augment=False):
    transform_train, transform_test = _get_lacuna_transforms(augment=augment)
    train_set = Small_Lacuna10(root=root, train=True, transform=transform_train)
    test_set = Small_Lacuna10(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset
def small_binary_lacuna10(root, augment=False):
    transform_train, transform_test = _get_lacuna_transforms(augment=augment)
    train_set = Small_Binary_Lacuna10(root=root, train=True, transform=transform_train)
    test_set = Small_Binary_Lacuna10(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset
def tinyimagenet_pretrain(root, augment=False):
    transform_train, transform_test = _get_imagenet_transforms(augment=augment)
    train_set = TinyImageNet_pretrain(root=root, train=True, transform=transform_train)
    test_set = TinyImageNet_pretrain(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset
def tinyimagenet_finetune(root, augment=False):
    transform_train, transform_test = _get_imagenet_transforms(augment=augment)
    train_set = TinyImageNet_finetune(root=root, train=True, transform=transform_train)
    test_set = TinyImageNet_finetune(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset
def tinyimagenet_finetune5(root, augment=False):
    transform_train, transform_test = _get_imagenet_transforms(augment=augment)
    train_set = TinyImageNet_finetune5(root=root, train=True, transform=transform_train)
    test_set = TinyImageNet_finetune5(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset
def mix10(root, augment=False):
    transform_train, transform_test = _get_mix_transforms(augment=augment)
    lacuna_train_set = Lacuna10(root=root, train=True, transform=transform_train)
    lacuna_test_set = Lacuna10(root=root, train=False, transform=transform_test)
    cifar_train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=False, transform=transform_train)
    cifar_test_set  = torchvision.datasets.CIFAR10(root=root, train=False, download=False, transform=transform_test)
    
    lacuna_train_set.targets = np.array(lacuna_train_set.targets)
    lacuna_test_set.targets = np.array(lacuna_test_set.targets)
    cifar_train_set.targets = np.array(cifar_train_set.targets)
    cifar_test_set.targets = np.array(cifar_test_set.targets)
        
    lacuna_train_set.data = lacuna_train_set.data[:,::2,::2,:]
    lacuna_test_set.data = lacuna_test_set.data[:,::2,::2,:]
    
    classes = np.arange(5)
    for c in classes:
        lacuna_train_class_len = np.sum(lacuna_train_set.targets==c)
        lacuna_train_set.data[lacuna_train_set.targets==c]=cifar_train_set.data[cifar_train_set.targets==c]\
                                                            [:lacuna_train_class_len,:,:,:]
        lacuna_test_class_len = np.sum(lacuna_test_set.targets==c)
        lacuna_test_set.data[lacuna_test_set.targets==c]=cifar_test_set.data[cifar_test_set.targets==c]\
                                                            [:lacuna_test_class_len,:,:,:]
    return lacuna_train_set, lacuna_test_set

@_add_dataset
def mix100(root, augment=False):
    transform_train, transform_test = _get_mix_transforms(augment=augment)
    lacuna_train_set = Lacuna100(root=root, train=True, transform=transform_train)
    lacuna_test_set = Lacuna100(root=root, train=False, transform=transform_test)
    cifar_train_set = torchvision.datasets.CIFAR100(root=root, train=True, download=False, transform=transform_train)
    cifar_test_set  = torchvision.datasets.CIFAR100(root=root, train=False, download=False, transform=transform_test)
    
    lacuna_train_set.targets = np.array(lacuna_train_set.targets)
    lacuna_test_set.targets = np.array(lacuna_test_set.targets)
    cifar_train_set.targets = np.array(cifar_train_set.targets)
    cifar_test_set.targets = np.array(cifar_test_set.targets)
        
    lacuna_train_set.data = lacuna_train_set.data[:,::2,::2,:]     # ::2表示每隔2个取一个
    lacuna_test_set.data = lacuna_test_set.data[:,::2,::2,:]
    
    classes = np.arange(50)    
    for c in classes:     # 将lacuna100数据集中的前50类数据换为cifar100数据集
        lacuna_train_class_len = np.sum(lacuna_train_set.targets==c)       # lacuna_train_set数据集中类别为c的个数
        lacuna_train_set.data[lacuna_train_set.targets==c]=cifar_train_set.data[cifar_train_set.targets==c]\
                                                            [:lacuna_train_class_len,:,:,:]    # 将lacuna_train_set中类别为c的数据值 换为 cifar_train_set中类别为c的数据值
        lacuna_test_class_len = np.sum(lacuna_test_set.targets==c)
        lacuna_test_set.data[lacuna_test_set.targets==c]=cifar_test_set.data[cifar_test_set.targets==c]\
                                                            [:lacuna_test_class_len,:,:,:]     # 将lacuna_test_set中类别为c的数据值 换为 cifar_test_set中类别为c的数据值
    return lacuna_train_set, lacuna_test_set



def replace_indexes(dataset: torch.utils.data.Dataset, indexes: Union[List[int], np.ndarray], seed=0,       # 输入的indexes可能为int类型的List，也可能会是ndarray
                    only_mark: bool = False):
    if not only_mark:
        rng = np.random.RandomState(seed)      # 随机数生成器
        new_indexes = rng.choice(list(set(range(len(dataset))) - set(indexes)), size=len(indexes))       # 从indexes以外的序号中随机抽取  len(indexes)个
        dataset.data[indexes] = dataset.data[new_indexes]       # 用其余数据集替换这indexs个数据
        dataset.targets[indexes] = dataset.targets[new_indexes]
    else:
        # Notice the -1 to make class 0 work
        dataset.targets[indexes] = - dataset.targets[indexes] - 1


def replace_class(dataset: torch.utils.data.Dataset, class_to_replace: List[int], num_indexes_to_replace: int = None,     # class_to_replace为需要替换的类别，num_indexes_to_replace为每一类中需要替换的数量
                  seed: int = 0, only_mark: bool = False):

    indexes = np.array([])
    for itm in class_to_replace:
        indexes = np.concatenate((indexes, np.flatnonzero(np.array(dataset.targets) == itm)))     # 将数据集中标签为itm的位置下标加入到indexes中
    indexes = indexes.astype(int)
    if num_indexes_to_replace is not None:
        assert num_indexes_to_replace <= len(        # 需要替换的数量需要小于数据集中所有需要替换的标签数量
            indexes), f"Want to replace {num_indexes_to_replace} indexes but only {len(indexes)} samples in dataset"
        rng = np.random.RandomState(seed)
        indexes = rng.choice(indexes, size=num_indexes_to_replace, replace=False)     # 随机选出 num_indexes_to_replace 个需要替换的下标
        print(f"Replacing indexes {indexes}")
    replace_indexes(dataset, indexes, seed, only_mark)       # 真正实现替换

def confuse_class(dataset: torch.utils.data.Dataset, class_to_replace: List[int], num_indexes_to_replace: int = None,
                  seed: int = 0, only_mark: bool = False):

    indexes0 = np.flatnonzero(np.array(dataset.targets) == class_to_replace[0])      # 数据集中标签为需要替换的第0类的下标
    indexes0 = indexes0.astype(int)

    indexes1 = np.flatnonzero(np.array(dataset.targets) == class_to_replace[1])      # 数据集中标签为需要替换的第1类的下标
    indexes1 = indexes1.astype(int)
    # 随机取indexes0和indexes1前一半的下标
    np.random.seed(seed)
    np.random.shuffle(indexes0)
    np.random.seed(seed)
    np.random.shuffle(indexes1)

    sub_indexes0 = indexes0[:int(len(indexes0)/2)]
    sub_indexes1 = indexes1[:int(len(indexes1)/2)]
    # 交换两者的标签
    dataset.targets[sub_indexes0] = class_to_replace[1]
    dataset.targets[sub_indexes1] = class_to_replace[0]
    # 合并
    indexes = np.concatenate((sub_indexes0, sub_indexes1))

    replace_indexes(dataset, indexes, seed, only_mark)    # 对该indexes数据集，用其余数据的值和标签替换

def get_loaders(dataset_name, class_to_replace: List[int] = None, num_indexes_to_replace: int = None,
                indexes_to_replace: List[int] = None, confuse_mode: bool = False, seed: int = 1, only_mark: bool = False, 
                root: str = None, batch_size=128, shuffle=True, split: str = 'train',
                **dataset_kwargs):
    '''
    :param dataset_name: Name of dataset to use
    :param class_to_replace: If not None, specifies which class to replace completely or partially
    :param num_indexes_to_replace: If None, all samples from `class_to_replace` are replaced. Else, only replace
                                   `num_indexes_to_replace` samples
    :param indexes_to_replace: If not None, denotes the indexes of samples to replace. Only one of class_to_replace and
                               indexes_to_replace can be specidied.
    :param seed: Random seed to sample the samples to replace and to initialize the data loaders so that they sample
                 always in the same order
    :param root: Root directory to initialize the dataset
    :param batch_size: Batch size of data loader
    :param shuffle: Whether train data should be randomly shuffled when loading (test data are never shuffled)
    :param dataset_kwargs: Extra arguments to pass to the dataset init.
    :return: The train_loader and test_loader
    '''
    manual_seed(seed)
    if root is None:
        root = os.path.expanduser('~/data')
    train_set, test_set = _DATASETS[dataset_name](root, **dataset_kwargs)
    train_set.targets = np.array(train_set.targets)
    test_set.targets = np.array(test_set.targets)
    
    valid_set = copy.deepcopy(train_set)      # 深拷贝，后续改动不影响这次拷贝
    rng = np.random.RandomState(seed)
    
    valid_idx=[]     # 保证了 0-max(train_set.targets) 各个类别中都能抽出 20% 的下标
    for i in range(max(train_set.targets) + 1):
        class_idx = np.where(train_set.targets==i)[0]       # 取得标签为i的数据集下标
        valid_idx.append(rng.choice(class_idx,int(0.2*len(class_idx)),replace=False))     # 从这些下标中抽取20%，加入到valid_idx中
    valid_idx = np.hstack(valid_idx)       # 将np.array([[1,2,3],[2,3,4]])变为[1,2,3,2,3,4]
    
    train_idx = list(set(range(len(train_set)))-set(valid_idx))    # train_idx只包含非valid_idx的下标
    
    train_set_copy = copy.deepcopy(train_set)
    # 将数据集分为 train_set(80%) 和 valid_set(20%)
    train_set.data = train_set_copy.data[train_idx]
    train_set.targets = train_set_copy.targets[train_idx]

    valid_set.data = train_set_copy.data[valid_idx]
    valid_set.targets = train_set_copy.targets[valid_idx]

    print ("confuse mode:",confuse_mode)
    print ("split mode:", split)
    if confuse_mode:          # 如果需要打乱（只能打乱两类）,只改变train_set中需要打乱的两个类的标签
        indexes0 = np.flatnonzero(np.array(train_set.targets) == class_to_replace[0])
        indexes0 = indexes0.astype(int)

        indexes1 = np.flatnonzero(np.array(train_set.targets) == class_to_replace[1])
        indexes1 = indexes1.astype(int)

        #np.random.seed(seed)
        #np.random.shuffle(indexes0)
        #np.random.seed(seed)
        #np.random.shuffle(indexes1)

        #sub_indexes0 = indexes0[:int(num_indexes_to_replace/2)]
        #sub_indexes1 = indexes1[:int(num_indexes_to_replace/2)]

        rng = np.random.RandomState(seed-1)
        sub_indexes0 = rng.choice(indexes0, size=int(num_indexes_to_replace/2), replace=False)
        sub_indexes1 = rng.choice(indexes1, size=int(num_indexes_to_replace/2), replace=False)
        # 互相赋予对方的标签
        train_set.targets[sub_indexes0] = class_to_replace[1]
        train_set.targets[sub_indexes1] = class_to_replace[0]

        indexes = np.concatenate((sub_indexes0, sub_indexes1))    # 将class_to_replace的第0、1类抽取的50%下标组合
        print (indexes)

        if split == "train":
            class_to_replace = None
            indexes_to_replace = None
        elif split == "forget":
            class_to_replace = None
            indexes_to_replace = indexes

        
    if class_to_replace is not None and indexes_to_replace is not None:
        raise ValueError("Only one of `class_to_replace` and `indexes_to_replace` can be specified")
    if class_to_replace is not None: 
        if confuse_mode:
            if len(class_to_replace) != 2:
                raise ValueError("In the confusion mode, the number of classes should be 2")
            confuse_class(train_set, class_to_replace, num_indexes_to_replace=num_indexes_to_replace, seed=seed-1,\
                      only_mark=only_mark)
        else:
            replace_class(train_set, class_to_replace, num_indexes_to_replace=num_indexes_to_replace, seed=seed-1,\
                          only_mark=only_mark)      # 用数据集的其他数据替换 class_to_replace 中的 num_indexes_to_replace 个数据
            if num_indexes_to_replace is None:      # 所有 class_to_replace 中的样本均会被替换
                test_indexes = np.array([])
                for c in class_to_replace:
                    test_indexes = np.concatenate((test_indexes, np.where(test_set.targets == c)[0]))
                test_indexes = test_indexes.astype(int)
                all_indexes = np.indices(test_set.targets.shape)      # 等于np.array([list(range(test_set.targets.shape))])
                not_indices = np.setxor1d(all_indexes, test_indexes)  # 查找两个数组的集合异或，即将all_indexes中test_indexes值删掉
                test_set.data = test_set.data[not_indices]
                test_set.targets = test_set.targets[not_indices]
    elif indexes_to_replace is not None:
        replace_indexes(dataset=train_set, indexes=indexes_to_replace, seed=seed-1, only_mark=only_mark)

    loader_args = {'num_workers': 0, 'pin_memory': False}

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle,            # 占train_set中80%
                                               worker_init_fn=_init_fn if seed is not None else None, **loader_args)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False,              # 占train_set中20%
                                              worker_init_fn=_init_fn if seed is not None else None, **loader_args)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,                # test_set
                                              worker_init_fn=_init_fn if seed is not None else None, **loader_args)


    return train_loader, valid_loader, test_loader

