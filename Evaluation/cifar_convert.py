from PIL import Image
import numpy as np
from tqdm import tqdm
import os

corruptions = ["brightness","contrast","defocus_blur","elastic_transform","fog","frost","gaussian_blur","gaussian_noise","glass_blur","impulse_noise","jpeg_compression","motion_blur","pixelate","saturate","shot_noise","snow","spatter","speckle_noise","zoom_blur"]

# corruptions = ["brightness"]

targets = np.load('/home/ss6712/cifar100c/CIFAR-100-C/labels.npy')

for corruption in corruptions:
    data_cifar = np.load('/home/ss6712/cifar100c/CIFAR-100-C/' + corruption + '.npy')

    for j in tqdm(range(0, 10000)):
        target = targets[j]
        image = Image.fromarray(data_cifar[j])

        directory = '/home/ss6712/cifar100c/final/' + corruption + '/'+ str(targets[j]) + "/"

        if not os.path.exists(directory):
            os.makedirs(directory)

        image.save(directory + corruption + str(j) + ".png","PNG")