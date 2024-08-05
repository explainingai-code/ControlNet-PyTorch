import torch
import yaml
import argparse
import os
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from dataset.mnist_dataset import MnistDataset
from torch.utils.data import DataLoader
from models.controlnet import ControlNet
from scheduler.linear_noise_scheduler import LinearNoiseScheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using mps')


def train(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################
    
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']
    
    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    
    # Create the dataset
    mnist = MnistDataset('train',
                         im_path=dataset_config['im_path'],
                         return_hints=True)
    mnist_loader = DataLoader(mnist, batch_size=train_config['batch_size'], shuffle=True)

    # Load model with checkpoint
    model = ControlNet(model_config,
                       model_locked=True,
                       model_ckpt=os.path.join(train_config['task_name'],
                                               train_config['ddpm_ckpt_name']),
                       device=device).to(device)
    model.train()
    
    # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
    
    # Load checkpoint if found
    if os.path.exists(os.path.join(train_config['task_name'],
                                   train_config['controlnet_ckpt_name'])):
        print('Loading checkpoint as found one')
        model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                      train_config['controlnet_ckpt_name']),
                                         map_location=device))

    # Specify training parameters
    num_epochs = train_config['controlnet_epochs']
    optimizer = Adam(model.get_params(), lr=train_config['controlnet_lr'])
    criterion = torch.nn.MSELoss()
    
    # Run training
    steps = 0
    for epoch_idx in range(num_epochs):
        losses = []
        for im, hint in tqdm(mnist_loader):
            optimizer.zero_grad()

            im = im.float().to(device)
            hint = hint.float().to(device)

            # Sample random noise
            noise = torch.randn_like(im).to(device)
            
            # Sample timestep
            t = torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],)).to(device)
            
            # Add noise to images according to timestep
            noisy_im = scheduler.add_noise(im, noise, t)

            # Additionally start passing the hint
            noise_pred = model(noisy_im, t, hint)

            loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            steps += 1
        print('Finished epoch:{} | Loss : {:.4f}'.format(
            epoch_idx + 1,
            np.mean(losses),
        ))
        torch.save(model.state_dict(), os.path.join(train_config['task_name'],
                                                    train_config['controlnet_ckpt_name']))
    
    print('Done Training ...')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for controlnet ddpm training')
    parser.add_argument('--config', dest='config_path',
                        default='config/mnist.yaml', type=str)
    args = parser.parse_args()
    train(args)
