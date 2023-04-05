#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

IMG_SIZE = (180, 240)
IMG_SIZE = (513, 513)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE)

torch.manual_seed(42)

train_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE,
                      interpolation=transforms.InterpolationMode.NEAREST),
    transforms.RandomRotation(
        (0, 21), interpolation=transforms.InterpolationMode.NEAREST,
        fill=0),
    transforms.ToTensor()])

train_dataset = ImageFolder('../img/Train/', transform=train_transforms)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1,
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32,
                                           shuffle=True)

# Create a new dataset with 31 augmented images per original image
# new_dataset = []
# for i in range(len(train_dataset)):
#     original_image, label = train_dataset[i]
#     new_dataset.append((original_image, label))
#     original_image_pil = transforms.ToPILImage()(original_image)
#     for j in range(31):
#         augmented_image = train_transforms(original_image_pil)
#         new_dataset.append((augmented_image, label))
#
# # Create a new DataLoader for the augmented dataset
# dataloader = torch.utils.data.DataLoader(new_dataset, batch_size=32,
#                                          shuffle=True)

print(f"len of original dataset: {len(train_dataset)}")
# print(f"len of augmented dataset: {len(new_dataset)}")

# load deeplab model and fix its weights
model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
model = model.to(DEVICE)
for param in model.parameters():
    param.requires_grad = False


# for data in dataloader:
for data in train_loader:
    imgs, labels = data
    imgs = imgs.to(DEVICE)
    imgs = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])(imgs)

    output_imgs = model(imgs)
    output_tensor = output_imgs['out'][0]
    output_predictions = output_tensor.argmax(dim=0).cpu().numpy()

    # Convert the output predictions to an image
    colors = np.array([(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), (192, 192, 192), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), (128, 64, 128), (0, 192, 128), (128, 192, 128), (64, 64, 0), (192, 64, 0), (64, 192, 0), (192, 192, 0), (64, 64, 128)], dtype=np.uint8)

    # colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    # colors = (colors % 255).numpy().astype("uint8")
    output_image = Image.fromarray(output_predictions.astype(np.uint8)).resize(IMG_SIZE)
    output_image.putpalette(colors)

    # Display the output image
    plt.imshow(output_image)
    plt.show()
exit()



# Loop over the dataloader to train your model
for batch in dataloader:
    # Train your model with the augmented images
    pass
