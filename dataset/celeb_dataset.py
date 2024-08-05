import glob
import os
import cv2
import torchvision
import numpy as np
from PIL import Image
from utils.diffusion_utils import load_latents
from tqdm import tqdm
from torch.utils.data.dataset import Dataset


class CelebDataset(Dataset):
    r"""
    Celeb dataset will by default centre crop and resize the images.
    This can be replaced by any other dataset. As long as all the images
    are under one directory.
    """

    def __init__(self, split, im_path, im_size=256, im_channels=3, im_ext='jpg',
                 use_latents=False, latent_path=None, return_hint=False):#, condition_config=None):
        self.split = split
        self.im_size = im_size
        self.im_channels = im_channels
        self.im_ext = im_ext
        self.im_path = im_path
        self.latent_maps = None
        self.use_latents = False
        self.return_hints = return_hint
        # self.condition_types = [] if condition_config is None else condition_config['condition_types']

        # self.idx_to_cls_map = {}
        # self.cls_to_idx_map = {}

        # if 'image' in self.condition_types:
        #     self.mask_channels = condition_config['image_condition_config']['image_condition_input_channels']
        #     self.mask_h = condition_config['image_condition_config']['image_condition_h']
        #     self.mask_w = condition_config['image_condition_config']['image_condition_w']

        #self.images, self.texts, self.masks = self.load_images(im_path)
        self.images = self.load_images(im_path)

        # Whether to load images or to load latents
        if use_latents and latent_path is not None:
            latent_maps = load_latents(latent_path)
            if len(latent_maps) == len(self.images):
                self.use_latents = True
                self.latent_maps = latent_maps
                print('Found {} latents'.format(len(self.latent_maps)))
            else:
                print('Latents not found')

    def load_images(self, im_path):
        r"""
        Gets all images from the path specified
        and stacks them all up
        """
        assert os.path.exists(im_path), "images path {} does not exist".format(im_path)
        ims = []
        fnames = glob.glob(os.path.join(im_path, 'CelebA-HQ-img/*.{}'.format('png')))
        fnames += glob.glob(os.path.join(im_path, 'CelebA-HQ-img/*.{}'.format('jpg')))
        fnames += glob.glob(os.path.join(im_path, 'CelebA-HQ-img/*.{}'.format('jpeg')))
        texts = []
        masks = []

        # if 'image' in self.condition_types:
        #     label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth',
        #                   'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
        #     self.idx_to_cls_map = {idx: label_list[idx] for idx in range(len(label_list))}
        #     self.cls_to_idx_map = {label_list[idx]: idx for idx in range(len(label_list))}

        for fname in tqdm(fnames):
            ims.append(fname)

            # if 'text' in self.condition_types:
            #     im_name = os.path.split(fname)[1].split('.')[0]
            #     captions_im = []
            #     with open(os.path.join(im_path, 'celeba-caption/{}.txt'.format(im_name))) as f:
            #         for line in f.readlines():
            #             captions_im.append(line.strip())
            #     texts.append(captions_im)

            # if 'image' in self.condition_types:
            #     im_name = int(os.path.split(fname)[1].split('.')[0])
            #     masks.append(os.path.join(im_path, 'CelebAMask-HQ-mask', '{}.png'.format(im_name)))
        # if 'text' in self.condition_types:
        #     assert len(texts) == len(ims), "Condition Type Text but could not find captions for all images"
        # if 'image' in self.condition_types:
        #     assert len(masks) == len(ims), "Condition Type Image but could not find masks for all images"
        print('Found {} images'.format(len(ims)))
        #print('Found {} masks'.format(len(masks)))
        #print('Found {} captions'.format(len(texts)))
        return ims#, texts, masks

    # def get_mask(self, index):
    #     r"""
    #     Method to get the mask of WxH
    #     for given index and convert it into
    #     Classes x W x H mask image
    #     :param index:
    #     :return:
    #     """
    #     mask_im = Image.open(self.masks[index])
    #     mask_im = np.array(mask_im)
    #     im_base = np.zeros((self.mask_h, self.mask_w, self.mask_channels))
    #     for orig_idx in range(len(self.idx_to_cls_map)):
    #         im_base[mask_im == (orig_idx + 1), orig_idx] = 1
    #     mask = torch.from_numpy(im_base).permute(2, 0, 1).float()
    #     return mask

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        ######## Set Conditioning Info ########
        # cond_inputs = {}
        # if 'text' in self.condition_types:
        #     cond_inputs['text'] = random.sample(self.texts[index], k=1)[0]
        # if 'image' in self.condition_types:
        #     mask = self.get_mask(index)
        #     cond_inputs['image'] = mask
        #######################################
        # im = Image.open(self.images[index])
        # im.save('original_image.png')
        # canny_image = np.array(im)
        # print(self.images[index])
        # low_threshold = 100
        # high_threshold = 200
        # import cv2
        # canny_image = cv2.Canny(canny_image, low_threshold, high_threshold)
        # canny_image = canny_image[:, :, None]
        # canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
        # canny_image = 255 - canny_image
        # canny_image = Image.fromarray(canny_image)
        # canny_image.save('canny_image.png')
        # print(list(self.latent_maps.keys())[0])
        # print(self.images[index] in self.latent_maps)
        # print(self.images[index].replace('../', '') in self.latent_maps)
        # latent = self.latent_maps[self.images[index].replace('../', '')]
        # latent = torch.clamp(latent, -1., 1.)
        # latent = (latent + 1) / 2
        # latent = torchvision.transforms.ToPILImage()(latent[0:-1, :, :])
        # latent.save('latent_image.png')
        # exit(0)

        if self.use_latents:
            latent = self.latent_maps[self.images[index]]
            if self.return_hints:
                canny_image = Image.open(self.images[index])
                canny_image = np.array(canny_image)
                canny_image = cv2.Canny(canny_image, 100, 200)
                canny_image = canny_image[:, :, None]
                canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
                canny_image_tensor = torchvision.transforms.ToTensor()(canny_image)
                return latent, canny_image_tensor
            else:
                return latent

        else:
            im = Image.open(self.images[index])
            im_tensor = torchvision.transforms.Compose([
                torchvision.transforms.Resize(self.im_size),
                torchvision.transforms.CenterCrop(self.im_size),
                torchvision.transforms.ToTensor(),
            ])(im)
            im.close()

            # Convert input to -1 to 1 range.
            im_tensor = (2 * im_tensor) - 1

            if self.return_hints:
                canny_image = Image.open(self.images[index])
                canny_image = np.array(canny_image)
                canny_image = cv2.Canny(canny_image, 100, 200)
                canny_image = canny_image[:, :, None]
                canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
                canny_image_tensor = torchvision.transforms.ToTensor()(canny_image)
                return im_tensor, canny_image_tensor
            else:
                return im_tensor


if __name__ == '__main__':
    mnist = CelebDataset('train', im_path='../data/CelebAMask-HQ',
                         use_latents=True, latent_path='../celebhq/vae_latents')
    x = mnist[1800]
