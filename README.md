# Robust-Prompt

Visual prompting has recently gained traction in its ability to transfer the learnings of a model on a particular distribution to another out-of-distribution dataset. Moreover, one of the best models evaluated on the ImageNet-C dataset, which aims at testing robustness of image classification, uses augmentation techniques such as  DeepAugment - an image-to-image neural network for performing data augmentation. In this work, we present a novel (to the best of our knowledge) technnique which combines visual prompting and DeepAugment to improve robustness of large language+vision models, specifically - CLIP. Apart from DeepAugment, we also perform composition with techninques like CutOut and CutMix and find that CutMix performs especially well on CIFAR-C, which serves as a measure for testing the model robustness to perturbations.

![image](https://user-images.githubusercontent.com/29446732/208546252-d959a5d1-69b3-456d-93f9-133ea89f2384.png)

# Installation

To run the requirements for running the code, you can clone the repo and install the requirements
`pip install -r requirements.txt`
To install the models run the `download_models.sh` file.

# Structure

* `/DeepAugment` contains scripts for evaluating deep augmented samples for ImageNet andn CIFAR-100.
* `/Evaluation` contains scripts for evaluating models on CIFAR-C dataset 
* `/ImageNetLoader` contains scripts for loading and manipulating ImageNet data
* `/model_checkpoints_version_1` contains checkpoints for our saved models
* `/VisualPrompting` contains scripts for training Visual prompts

# Data

# Data augmentation
### DeepAug:
### CutMix
### CutOut:
  To create the augmented images using cutout, run the following command:
  ```
  python cutout/create_and_save_data.py --dataset cifar100 --data_augmentation --cutout --length 16 --batch_size 1
  ```


# Visual Prompt Tuning

Once the dataset is created concatenated with the augmented data from CIFAR100, we run the prompt tuning stage to train the network to learn the parameters for the prompt. To tune the prompt for the CLiP model, run the following command:
```
python3 visual_prompting/main_clip.py --dataset cifar100 --root ./data --train_folder [TRAIN_FOLDER_PATH_WITH_AUGMENTATIONS] --val_folder [VAL_FOLDER_PATHS_WITH_AUGMENTATIONS] --num_workers 1 --batch_size 64
```

# Evaluation

To test our trained prompts on natural corruptions, we evaluate our model on the CIFAR100-C dataset. 
```
python3 cifar_c.py --dataset cifar100 --root ./data --cifar_c_path [PATH_TO_CIFAR_C] --model_saved_path [SAVED_MODEL_FOLDER/checkpoint.pth.tar] --train_folder [TRAIN_FOLDER] --val_folder [VAL_FOLDER] --num_workers 3 --batch_size 500
```

# Acknowledgments

We would like to thank the work done by the following:

* Hyojin Bahng, Ali Jahanian, Swami Sankaranarayanan, and Phillip Isola. Exploring visual prompts
for adapting large-scale models, 2022. URL https://arxiv.org/abs/2203.17274
* Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common
corruptions and perturbations, 2019. URL https://arxiv.org/abs/1903.12261.

## References:
* https://github.com/hjbahng/visual_prompting
* https://github.com/hendrycks/imagenet-r
* https://github.com/ildoonet/cutmix
* https://towardsdatascience.com/downloading-and-using-the-imagenet-dataset-with-pytorch-f0908437c4be
* https://towardsdatascience.com/pytorch-ignite-classifying-tiny-imagenet-with-efficientnet-e5b1768e5e8f#:~:text=Tiny%20ImageNet%20is%20a%20subset,images%2C%20and%2050%20test%20images


