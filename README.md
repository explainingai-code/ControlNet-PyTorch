ControlNet Implementation in PyTorch
========

This repository implements ControlNet in PyTorch for diffusion models.
As of now, the repo provides code to do the following:
* Training and Inference of Unconditional DDPM on MNIST dataset
* Training and Inference of ControlNet with DDPM on MNIST using canny edges
* Training and Inference of Unconditional Latent Diffusion Model on CelebHQ dataset(resized to 128x128 with latent images being 32x32)
* Training and Inference of ControlNet with Unconditional Latent Diffusion Model on CelebHQ using canny edges


For autoencoder of Latent Diffusion Model, I provide training and inference code for vae.


## ControlNet Tutorial Video


___  


## Sample Output for ControlNet with DDPM on MNIST as an example


## Sample Output for ControlNet with LDM on CelebHQ

___

## Setup
* Create a new conda environment with python 3.10 then run below commands
* ```git clone https://github.com/explainingai-code/ControlNet-PyTorch.git```
* ```cd ControlNet-PyTorch```
* ```pip install -r requirements.txt```
* Download lpips weights from https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/weights/v0.1/vgg.pth and put it in ```models/weights/v0.1/vgg.pth```
___  

## Data Preparation
### Mnist

For setting up the mnist dataset follow - https://github.com/explainingai-code/Pytorch-VAE#data-preparation

Ensure directory structure is following
```
ControlNet-PyTorch
    -> data
        -> mnist
            -> train
                -> images
                    -> *.png
            -> test
                -> images
                    -> *.png
```

### CelebHQ 
For setting up on CelebHQ, simply download the images from the official repo of CelebMASK HQ [here](https://github.com/switchablenorms/CelebAMask-HQ?tab=readme-ov-file).

Ensure directory structure is the following
```
ControlNet-PyTorch
    -> data
        -> CelebAMask-HQ
            -> CelebA-HQ-img
                -> *.jpg

```
---
## Configuration
 Allows you to play with different components of ddpm and autoencoder training
* ```config/mnist.yaml``` - Config for MNIST dataset
* ```config/celebhq.yaml``` - Configuration used for celebhq dataset

Relevant configuration parameters

Most parameters are self-explanatory but below I mention couple which are specific to this repo.
* ```autoencoder_acc_steps``` : For accumulating gradients if image size is too large for larger batch sizes
* ```save_latents``` : Enable this to save the latents , during inference of autoencoder. That way ddpm training will be faster

___  
## Training
The repo provides training and inference for Mnist(Unconditional DDPM) and CelebHQ (Unconditional LDM) and ControlNet with both these variations using canny edges.

For working on your own dataset:
* Create your own config and have the path in config point to images (look at `celebhq.yaml` for guidance)
* Create your own dataset class which will just collect all the filenames and return the image and its hint in its getitem method. Look at `mnist_dataset.py` or `celeb_dataset.py` for guidance 

Once the config and dataset is setup:
* For training and inference of Unconditional DDPM follow [this section](#training-unconditional-ddpm)
* For training and inference of ControlNet with Unconditional DDPM follow [this section](#training-controlnet-for-unconditional-ddpm)
* Train the auto encoder on your dataset using [this section](#training-autoencoder-for-ldm)
* For training and inference of Unconditional LDM follow [this section](#training-unconditional-ldm)
* For training and inference of ControlNet with Unconditional LDM follow [this section](#training-controlnet-for-unconditional-ldm)



## Training Unconditional DDPM
* For training ddpm on mnist,ensure the right path is mentioned in `mnist.yaml`
* For training ddpm on your own dataset 
  * Create your own config and have the path point to images (look at celebhq.yaml for guidance)
  * Create your own dataset class, similar to celeb_dataset.py
* Call the desired dataset class in training file
* For training DDPM run ```python -m tools.train_ddpm --config config/mnist.yaml``` for training ddpm with the desire config file
* For inference run ```python -m tools.sample_ddpm --config config/mnist.yaml``` for generating samples with right config file.

## Training ControlNet for Unconditional DDPM
* For training controlnet, ensure the right path is mentioned in `mnist.yaml`
* For training controlnet with ddpm on your own dataset 
  * Create your own config and have the path point to images (look at celebhq.yaml for guidance)
  * Create your own dataset class, similar to celeb_dataset.py
* Call the desired dataset class in training file
* Ensure ```return_hints``` is passed as True in the dataset class initialization
* For training controlnet run ```python -m tools.train_ddpm_controlnet --config config/mnist.yaml``` for training controlnet ddpm with the desire config file
* For inference run ```python -m tools.sample_ddpm_controlnet --config config/mnist.yaml``` for generating ddpm samples using canny hints with right config file.


## Training AutoEncoder for LDM
* For training autoencoder on celebhq,ensure the right path is mentioned in `celebhq.yaml`
* For training autoencoder on your own dataset 
  * Create your own config and have the path point to images (look at celebhq.yaml for guidance)
  * Create your own dataset class, similar to celeb_dataset.py
* Call the desired dataset class in training file
* For training autoencoder run ```python -m tools.train_vae --config config/celebhq.yaml``` for training autoencoder with the desire config file
* For inference make sure ```save_latent``` is `True` in the config
* For inference run ```python -m tools.infer_vae --config config/celebhq.yaml``` for generating reconstructions and saving latents with right config file.


## Training Unconditional LDM
Train the autoencoder first and setup dataset accordingly.

For training unconditional LDM ensure the right dataset is used in `train_ldm_vae.py`
* ```python -m tools.train_ldm_vae --config config/celebhq.yaml``` for training unconditional ldm using right config
* ```python -m tools.sample_ldm_vae --config config/celebhq.yaml``` for generating images using trained ldm


## Training ControlNet for Unconditional LDM
* For training controlnet with celebhq, ensure the right path is mentioned in `celebhq.yaml`
* For training controlnet with ldm on your own dataset 
  * Create your own config and have the path point to images (look at celebhq.yaml for guidance)
  * Create your own dataset class, similar to celeb_dataset.py
* Ensure Autoencoder and LDM have already been trained
* Call the desired dataset class in training file
* Ensure ```return_hints``` is passed as True in the dataset class initialization
* Ensure ```down_sample_factor``` is correctly computed in the model initialization  
* For training controlnet run ```python -m tools.train_ldm_controlnet --config config/celebhq.yaml``` for training controlnet ldm with the desire config file
* For inference with controlnet run ```python -m tools.sample_ldm_controlnet --config config/celebhq.yaml``` for generating ldm samples using canny hints with right config file.




## Training Conditional LDM
For training conditional models we need two changes:
* Dataset classes must provide the additional conditional inputs(see below)
* Config must be changed with additional conditioning config added

Specifically the dataset `getitem` will return the following:
* `image_tensor` for unconditional training
* tuple of `(image_tensor,  cond_input )` for conditional training where cond_input is a dictionary consisting of keys ```{class/text/image}```

### Training Class Conditional LDM
The repo provides class conditional latent diffusion model training code for mnist dataset, so one
can use that to follow the same for their own dataset

* Use `mnist_class_cond.yaml` config file as a guide to create your class conditional config file.
Specifically following new keys need to be modified according to your dataset within `ldm_params`.
* ```  
  condition_config:
    condition_types: ['class']
    class_condition_config :
      num_classes : <number of classes: 10 for mnist>
      cond_drop_prob : <probability of dropping class labels>
  ```
* Create a dataset class similar to mnist where the getitem method now returns a tuple of image_tensor and dictionary of conditional_inputs.
* For class, conditional input will ONLY be the integer class
* ```
    (image_tensor, {
                    'class' : {0/1/.../num_classes}
                    })

For training class conditional LDM map the dataset to the right class in `train_ddpm_cond` and run the below commands using desired config
* ```python -m tools.train_ddpm_cond --config config/mnist_class_cond.yaml``` for training class conditional on mnist 
* ```python -m tools.sample_ddpm_class_cond --config config/mnist.yaml``` for generating images using class conditional trained ddpm

### Training Text Conditional LDM
The repo provides text conditional latent diffusion model training code for celebhq dataset, so one
can use that to follow the same for their own dataset

* Use `celebhq_text_cond.yaml` config file as a guide to create your config file.
Specifically following new keys need to be modified according to your dataset within `ldm_params`.
* ```  
    condition_config:
        condition_types: [ 'text' ]
        text_condition_config:
            text_embed_model: 'clip' or 'bert'
            text_embed_dim: 512 or 768
            cond_drop_prob: 0.1
  ```
* Create a dataset class similar to celebhq where the getitem method now returns a tuple of image_tensor and dictionary of conditional_inputs.
* For text, conditional input will ONLY be the caption
* ```
    (image_tensor, {
                    'text' : 'a sample caption for image_tensor'
                    })

For training text conditional LDM map the dataset to the right class in `train_ddpm_cond` and run the below commands using desired config
* ```python -m tools.train_ddpm_cond --config config/celebhq_text_cond.yaml``` for training text conditioned ldm on celebhq 
* ```python -m tools.sample_ddpm_text_cond --config config/celebhq_text_cond.yaml``` for generating images using text conditional trained ddpm

### Training Text and Mask Conditional LDM
The repo provides text and mask conditional latent diffusion model training code for celebhq dataset, so one
can use that to follow the same for their own dataset and can even use that train a mask only conditional ldm

* Use `celebhq_text_image_cond.yaml` config file as a guide to create your config file.
Specifically following new keys need to be modified according to your dataset within `ldm_params`.
* ```  
    condition_config:
        condition_types: [ 'text', 'image' ]
        text_condition_config:
            text_embed_model: 'clip' or 'bert
            text_embed_dim: 512 or 768
            cond_drop_prob: 0.1
        image_condition_config:
           image_condition_input_channels: 18
           image_condition_output_channels: 3
           image_condition_h : 512 
           image_condition_w : 512
           cond_drop_prob: 0.1
  ```
* Create a dataset class similar to celebhq where the getitem method now returns a tuple of image_tensor and dictionary of conditional_inputs.
* For text and mask, conditional input will be caption and mask image
* ```
    (image_tensor, {
                    'text' : 'a sample caption for image_tensor',
                    'image' : NUM_CLASSES x MASK_H x MASK_W
                    })

For training text unconditional LDM map the dataset to the right class in `train_ddpm_cond` and run the below commands using desired config
* ```python -m tools.train_ddpm_cond --config config/celebhq_text_image_cond.yaml``` for training text and mask conditioned ldm on celebhq 
* ```python -m tools.sample_ddpm_text_image_cond --config config/celebhq_text_image_cond.yaml``` for generating images using text and mask conditional trained ddpm


## Output 
Outputs will be saved according to the configuration present in yaml files.

For every run a folder of ```task_name``` key in config will be created

During training of autoencoder the following output will be saved 
* Latest Autoencoder and discriminator checkpoint in ```task_name``` directory
* Sample reconstructions in ```task_name/vqvae_autoencoder_samples```

During inference of autoencoder the following output will be saved
* Reconstructions for random images in  ```task_name```
* Latents will be save in ```task_name/vqvae_latent_dir_name``` if mentioned in config

During training and inference of ddpm following output will be saved
* During training of unconditional or conditional DDPM we will save the latest checkpoint in ```task_name``` directory
* During sampling, unconditional sampled image grid for all timesteps in ```task_name/samples/*.png``` . The final decoded generated image will be `x0_0.png`. Images from `x0_999.png` to `x0_1.png` will be latent image predictions of denoising process from T=999 to T=1. Generated Image is at T=0
* During sampling, class conditionally sampled image grid for all timesteps in ```task_name/cond_class_samples/*.png``` . The final decoded generated image will be `x0_0.png`.  Images from `x0_999.png` to `x0_1.png` will be latent image predictions of denoising process from T=999 to T=1. Generated Image is at T=0
* During sampling, text only conditionally sampled image grid for all timesteps in ```task_name/cond_text_samples/*.png``` . The final decoded generated image will be `x0_0.png` . Images from `x0_999.png` to `x0_1.png` will be latent image predictions of denoising process from T=999 to T=1. Generated Image is at T=0
* During sampling, image only conditionally sampled image grid for all timesteps in ```task_name/cond_text_image_samples/*.png``` . The final decoded generated image will be `x0_0.png`. Images from `x0_999.png` to `x0_1.png` will be latent image predictions of denoising process from T=999 to T=1. Generated Image is at T=0




