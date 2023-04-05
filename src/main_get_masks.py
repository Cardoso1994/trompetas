#!/usr/bin/env python3

import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models.segmentation import deeplabv3_resnet50, \
    DeepLabV3_ResNet50_Weights, deeplabv3_resnet101, DeepLabV3_ResNet101_Weights

IMG_SIZE = (180, 240)
IMG_SIZE = (513, 513)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE)
DS_PATH = '../img/Train/'
MASK_PATH = '../img/Train_mask/'
MODEL = 50
MODEL = 101

torch.manual_seed(42)

train_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE,
                      interpolation=transforms.InterpolationMode.NEAREST),
    transforms.RandomRotation(
        (0, 21), interpolation=transforms.InterpolationMode.NEAREST,
        fill=0),
    transforms.ToTensor()])

train_dataset = ImageFolder(DS_PATH, transform=train_transforms)

batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
                                           # shuffle=True)

# load deeplab model and fix its weights
if MODEL == 50:
    model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
elif MODEL == 101:
    model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
model = model.to(DEVICE)
model.eval()  # set the model  to evaluation mode

for param in model.parameters():
    param.requires_grad = False

for batch, data in enumerate(train_loader):
    imgs, labels = data
    new_batch_size = labels.shape[0]
    filenames = [train_dataset.samples[idx][0].split("/")[-1] for idx in
                 range(batch * new_batch_size, (batch + 1) * new_batch_size)]
    print(filenames)
    imgs = imgs.to(DEVICE)
    # labels = labels.to(DEVICE)
    imgs = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])(imgs)
    output_imgs = model(imgs)
    output_imgs = output_imgs
    output_tensor = output_imgs['out'].cpu()
    pred_index = np.argmax(output_tensor.cpu(), axis=1)
    num_classes = output_tensor.shape[1]
    colormap = cm.get_cmap('inferno', int(num_classes))
    output_tensor = colormap(pred_index)

    for i, filename in enumerate(filenames):
        # Extract the output tensor for the i-th image in the batch
        output_image = output_tensor[i]

        # Convert the output tensor to a PIL image
        output_image = transforms.ToPILImage()((output_image * 255).astype('uint8'))
        output_image = output_image.convert('RGB')
        # plt.imshow(output_image)
        # plt.show()

        class_name = f"{labels[i]}_mask"
        os.makedirs(f"{MASK_PATH}/{class_name}", exist_ok=True)
        new_filename = filename.replace(".jpg", f"_mask_{MODEL}.jpg")

        # Save the output image to disk with the corresponding filename
        output_image.save(f"{MASK_PATH}/{class_name}/{new_filename}")
exit()
