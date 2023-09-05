#%%
import numpy as np
import torch
import torch.utils.data
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Current cuda device is', device)

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

train_data = datasets.MNIST(root='./data/',
                            train=True,
                            download=True,
                            transform=transform)

test_data = datasets.MNIST(root='./data/',
                           train=False,
                           download=True,
                           transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

print('Num of train data:', len(train_data))
print('Num of test data:', len(test_data))

it = iter(train_loader)
images, labels = next(it)

print('Shape of train data:', images.shape)
print('Shape of test data', labels.shape)

#%%
fig = plt.figure()
num = 60
for idx in range(1, num+1):
    plt.subplot(6, 10, idx)
    plt.axis('off')
    plt.imshow(images[idx].squeeze().numpy(), cmap='gray_r')
plt.show()

#%%
