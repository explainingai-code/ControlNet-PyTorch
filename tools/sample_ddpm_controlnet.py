import torch
import torchvision
import argparse
import yaml
import os
import random
from torchvision.utils import make_grid
from tqdm import tqdm
from dataset.mnist_dataset import MnistDataset
from models.controlnet import ControlNet
from scheduler.linear_noise_scheduler import LinearNoiseScheduler


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using mps')


def sample(model, scheduler, train_config, model_config, diffusion_config, dataset):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """
    xt = torch.randn((train_config['num_samples'],
                      model_config['im_channels'],
                      model_config['im_size'],
                      model_config['im_size'])).to(device)

    # Get random hints for the desired number of samples
    hints = []
    for idx in range(train_config['num_samples']):
        hint_idx = random.randint(0, len(dataset))
        hints.append(dataset[hint_idx][1].unsqueeze(0).to(device))
    hints = torch.cat(hints, dim=0).to(device)

    # Save the hints
    hints_grid = make_grid(hints, nrow=train_config['num_grid_rows'])
    hints_img = torchvision.transforms.ToPILImage()(hints_grid)
    hints_img.save(os.path.join(train_config['task_name'], 'hint.png'))

    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
        # Get prediction of noise
        noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device), hints)

        # Prediction from original model
        # noise_pred = model.trained_unet(xt, torch.as_tensor(i).unsqueeze(0).to(device))

        # Use scheduler to get x0 and xt-1
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))
        
        # Save x0
        ims = torch.clamp(xt, -1., 1.).detach().cpu()
        ims = (ims + 1) / 2
        grid = make_grid(ims, nrow=train_config['num_grid_rows'])
        img = torchvision.transforms.ToPILImage()(grid)
        if not os.path.exists(os.path.join(train_config['task_name'], 'samples_controlnet')):
            os.mkdir(os.path.join(train_config['task_name'], 'samples_controlnet'))
        img.save(os.path.join(train_config['task_name'], 'samples_controlnet', 'x0_{}.png'.format(i)))
        img.close()


def infer(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################
    
    diffusion_config = config['diffusion_params']
    model_config = config['model_params']
    train_config = config['train_params']
    dataset_config = config['dataset_params']

    # Change to require hints
    mnist_canny = MnistDataset('test', im_path=dataset_config['im_test_path'], return_hints=True)

    # Load model with checkpoint
    model = ControlNet(model_config,
                       model_ckpt=os.path.join(train_config['task_name'], train_config['ddpm_ckpt_name']),
                       device=device).to(device)

    assert os.path.exists(os.path.join(train_config['task_name'],
                                       train_config['controlnet_ckpt_name'])), "Train ControlNet first"
    model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                  train_config['controlnet_ckpt_name']),
                                     map_location=device))
    model.eval()
    print('Loaded ControlNet checkpoint')
    
    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    with torch.no_grad():
        sample(model, scheduler, train_config, model_config, diffusion_config, mnist_canny)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for controlnet ddpm image generation')
    parser.add_argument('--config', dest='config_path',
                        default='config/mnist.yaml', type=str)
    args = parser.parse_args()
    infer(args)
