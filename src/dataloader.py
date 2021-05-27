import os

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

def load_mnist(data_path, batch_size, val_portion=0):
    '''
    Args:
        data_path: path to dataset
        batch_size: batch size
        val_portion: portion of validation data
    Returns:
        dataloaders: dict which stores datasets
        classes: class corresponding to label
    '''
    
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, ), (0.5, ))])
    train_val_set = torchvision.datasets.MNIST(root=data_path, 
                                             train=True,
                                             download=True,
                                             transform=transform)
    

    test_set = torchvision.datasets.MNIST(root=data_path, 
                                          train=False, 
                                          download=True, 
                                          transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, 
                                              batch_size=batch_size,
                                              shuffle=False, 
                                              num_workers=os.cpu_count())
    classes = tuple(np.linspace(0, 9, 10, dtype=np.uint8))
    
    
    if val_portion!=0:
        val_size = int(len(train_val_set)*val_portion)
        train_size = len(train_val_set) - val_size

        train_set, val_set = torch.utils.data.random_split(train_val_set, [train_size, val_size])

        train_loader = torch.utils.data.DataLoader(train_set,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=os.cpu_count())

        val_loader = torch.utils.data.DataLoader(val_set,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=os.cpu_count())
        dataloaders = {"train":train_loader, "test":test_loader, "validation":val_loader}

        return dataloaders, classes
    
    
    else:
        train_loader = torch.utils.data.DataLoader(train_val_set,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=os.cpu_count())
        dataloaders = {"train":train_loader, "test":test_loader}

        return dataloaders, classes
    
