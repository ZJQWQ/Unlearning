import numpy as np
import os
from PIL import Image
import torchvision
from torchvision.datasets import VisionDataset    # torchvision.datasets包含了目前流行的数据集。VisionDataset共包含四个参数：root、transforms、transform（对图像）、target_transform（对标签）
root = os.path.expanduser('~/data')

np.random.seed(0)

class Small_CIFAR10(VisionDataset):

    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(Small_CIFAR10, self).__init__(root, transform=transform,
                                        target_transform=target_transform)      # 调用父类的__init__函数
        self.train = train                                                      # True
        ds = torchvision.datasets.CIFAR10(root=root, train=True, download=True)     # train为True，表示从训练集创建数据集
        ds.targets=np.array(ds.targets)                                         # 标签
        sub_ds_data_list=[]
        sub_ds_target_list=[]
        for i in range(10):
            if self.train:
                sub_cls_id = np.random.choice(np.where(ds.targets==i)[0],125,replace=False)    # 从每一类中随机选出125个index，且不会重复
            else:
                sub_cls_id = np.random.choice(np.where(ds.targets==i)[0],100,replace=False)
                #np.where(ds.targets==i)[0]
            sub_ds_data_list.append(ds.data[sub_cls_id,:,:,:])     # 将所有数据集添加到sub_ds_data_list列表中
            sub_ds_target_list.append(ds.targets[sub_cls_id])      # 同时也加入标签
        self.data=np.concatenate(sub_ds_data_list)
        self.targets=np.concatenate(sub_ds_target_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)      # 将array转为Image类型
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target              # 返回处理好的图像和标签

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")
    
class Small_CIFAR5(VisionDataset):            # 同Small_CIFAR10，但是抽取0-4标签的数据集

    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(Small_CIFAR5, self).__init__(root, transform=transform,
                                        target_transform=target_transform)
        self.train = train
        ds = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
        ds.targets=np.array(ds.targets)
        sub_ds_data_list=[]
        sub_ds_target_list=[]
        for i in range(5):
            if self.train:
                sub_cls_id = np.random.choice(np.where(ds.targets==i)[0],125,replace=False)
            else:
                sub_cls_id = np.random.choice(np.where(ds.targets==i)[0],100,replace=False)
                #np.where(ds.targets==i)[0]                
            sub_ds_data_list.append(ds.data[sub_cls_id,:,:,:])
            sub_ds_target_list.append(ds.targets[sub_cls_id])
        self.data=np.concatenate(sub_ds_data_list)
        self.targets=np.concatenate(sub_ds_target_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

class Small_CIFAR6(VisionDataset):            # 同Small_CIFAR10，但是抽取0-5标签的数据集

    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(Small_CIFAR6, self).__init__(root, transform=transform,
                                        target_transform=target_transform)
        self.train = train
        ds = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
        ds.targets=np.array(ds.targets)
        sub_ds_data_list=[]
        sub_ds_target_list=[]
        for i in range(6):
            if self.train:
                sub_cls_id = np.random.choice(np.where(ds.targets==i)[0],125,replace=False)
            else:
                sub_cls_id = np.random.choice(np.where(ds.targets==i)[0],100,replace=False)
                #np.where(ds.targets==i)[0]                
            sub_ds_data_list.append(ds.data[sub_cls_id,:,:,:])
            sub_ds_target_list.append(ds.targets[sub_cls_id])
        self.data=np.concatenate(sub_ds_data_list)
        self.targets=np.concatenate(sub_ds_target_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")


class Small_Binary_CIFAR10(VisionDataset):            # 同Small_CIFAR10，但是抽取0-1标签仅两类的数据集

    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(Small_Binary_CIFAR10, self).__init__(root, transform=transform,
                                        target_transform=target_transform)
        self.train = train
        ds = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
        ds.targets=np.array(ds.targets)
        sub_ds_data_list=[]
        sub_ds_target_list=[]
        for i in range(2):
            if self.train:
                sub_cls_id = np.random.choice(np.where(ds.targets==i)[0],250,replace=False)      # 对每一类随机抽取250个数据集
            else:
                sub_cls_id = np.random.choice(np.where(ds.targets==i)[0],250,replace=False)
                #np.where(ds.targets==i)[0]                
            sub_ds_data_list.append(ds.data[sub_cls_id,:,:,:])
            sub_ds_target_list.append(ds.targets[sub_cls_id])
        self.data=np.concatenate(sub_ds_data_list)
        self.targets=np.concatenate(sub_ds_target_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")
