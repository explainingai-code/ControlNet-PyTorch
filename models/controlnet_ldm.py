import torch
import torch.nn as nn
from models.unet_cond_base import Unet
from models.blocks import get_time_embedding


def make_zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


class ControlNet(nn.Module):
    r"""
    Control Net Module for DDPM
    """
    def __init__(self, im_channels,
                 model_config,
                 model_locked=True,
                 model_ckpt=None,
                 device=None,
                 down_sample_factor=32):
        super().__init__()
        # Trained DDPM
        self.model_locked = model_locked
        self.trained_unet = Unet(im_channels, model_config)

        # Load weights for the trained model
        if model_ckpt is not None and device is not None:
            print('Loading Trained Diffusion Model')
            self.trained_unet.load_state_dict(torch.load(model_ckpt,
                                                         map_location=device), strict=True)

        # ControlNet Copy of Trained DDPM
        # use_up = False removes the upblocks(decoder layers) from DDPM Unet
        self.control_unet = Unet(im_channels, model_config, use_up=False)
        # Load same weights as the trained model
        if model_ckpt is not None and device is not None:
            print('Loading Control Diffusion Model')
            self.control_unet.load_state_dict(torch.load(model_ckpt,
                                                         map_location=device), strict=False)

        # Hint Block for ControlNet
        # Stack of Conv activation and zero convolution at the end
        base_hint_channel = 16
        curr_down_sample_factor = down_sample_factor
        hint_layers = [nn.Sequential(
                nn.Conv2d(model_config['hint_channels'],
                          base_hint_channel,
                          kernel_size=3,
                          padding=(1, 1)),
                nn.SiLU())]
        while curr_down_sample_factor > 1:
            hint_layers.append(nn.Sequential(
                nn.Conv2d(base_hint_channel,
                          base_hint_channel*2,
                          kernel_size=3,
                          padding=(1, 1),
                          stride=2),
                nn.SiLU(),
                nn.Conv2d(base_hint_channel*2,
                          base_hint_channel*2,
                          kernel_size=3,
                          padding=(1, 1))
            ))
            base_hint_channel = base_hint_channel * 2
            curr_down_sample_factor = curr_down_sample_factor / 2
        hint_layers.append(nn.Sequential(
            nn.Conv2d(base_hint_channel,
                      self.trained_unet.down_channels[0],
                      kernel_size=3,
                      padding=(1, 1)),
            nn.SiLU(),
            make_zero_module(nn.Conv2d(self.trained_unet.down_channels[0],
                                       self.trained_unet.down_channels[0],
                                       kernel_size=1,
                                       padding=0))
        ))
        self.control_unet_hint_block = nn.Sequential(*hint_layers)

        # Zero Convolution Module for Downblocks(encoder Layers)
        self.control_unet_down_zero_convs = nn.ModuleList([
            make_zero_module(nn.Conv2d(self.trained_unet.down_channels[i],
                                       self.trained_unet.down_channels[i],
                                       kernel_size=1,
                                       padding=0))
            for i in range(len(self.trained_unet.down_channels)-1)
        ])

        # Zero Convolution Module for MidBlocks
        self.control_unet_mid_zero_convs = nn.ModuleList([
            make_zero_module(nn.Conv2d(self.trained_unet.mid_channels[i],
                                       self.trained_unet.mid_channels[i],
                                       kernel_size=1,
                                       padding=0))
            for i in range(1, len(self.trained_unet.mid_channels))
        ])

    def get_params(self):
        # Add all ControlNet parameters
        # First is our copy of unet
        params = list(self.control_unet.parameters())

        # Add parameters of hint Blocks & Zero convolutions for down/mid blocks
        params += list(self.control_unet_hint_block.parameters())
        params += list(self.control_unet_down_zero_convs.parameters())
        params += list(self.control_unet_mid_zero_convs.parameters())

        # If we desire to not have the decoder layers locked, then add
        # them as well
        if not self.model_locked:
            params += list(self.trained_unet.ups.parameters())
            params += list(self.trained_unet.norm_out.parameters())
            params += list(self.trained_unet.conv_out.parameters())
        return params

    def forward(self, x, t, hint):
        # Time embedding and timestep projection layers of trained unet
        trained_unet_t_emb = get_time_embedding(torch.as_tensor(t).long(),
                                                self.trained_unet.t_emb_dim)
        trained_unet_t_emb = self.trained_unet.t_proj(trained_unet_t_emb)

        # Get all downblocks output of trained unet first
        trained_unet_down_outs = []
        with torch.no_grad():
            train_unet_out = self.trained_unet.conv_in(x)
            for idx, down in enumerate(self.trained_unet.downs):
                trained_unet_down_outs.append(train_unet_out)
                train_unet_out = down(train_unet_out, trained_unet_t_emb)

        # ControlNet Layers start here #
        # Time embedding and timestep projection layers of controlnet's copy of unet
        control_unet_t_emb = get_time_embedding(torch.as_tensor(t).long(),
                                                self.control_unet.t_emb_dim)
        control_unet_t_emb = self.control_unet.t_proj(control_unet_t_emb)

        # Hint block of controlnet's copy of unet
        control_unet_hint_out = self.control_unet_hint_block(hint)

        # Call conv_in layer for controlnet's copy of unet
        # and add hint blocks output to it
        control_unet_out = self.control_unet.conv_in(x)
        control_unet_out += control_unet_hint_out

        # Get all downblocks output for controlnet's copy of unet
        control_unet_down_outs = []
        for idx, down in enumerate(self.control_unet.downs):
            # Save the control nets copy output after passing it through zero conv layers
            control_unet_down_outs.append(self.control_unet_down_zero_convs[idx](control_unet_out))
            control_unet_out = down(control_unet_out, control_unet_t_emb)

        for idx in range(len(self.control_unet.mids)):
            # Get midblock output of controlnets copy of unet
            control_unet_out = self.control_unet.mids[idx](control_unet_out, control_unet_t_emb)

            # Get midblock output of trained unet
            train_unet_out = self.trained_unet.mids[idx](train_unet_out, trained_unet_t_emb)

            # Add midblock output of controlnets copy of unet to that of trained unet
            # but after passing them through zero conv layers
            train_unet_out += self.control_unet_mid_zero_convs[idx](control_unet_out)

        # Call upblocks of trained unet
        for up in self.trained_unet.ups:
            # Get downblocks output from both trained unet and controlnets copy of unet
            trained_unet_down_out = trained_unet_down_outs.pop()
            control_unet_down_out = control_unet_down_outs.pop()

            # Add these together and pass this as downblock input to upblock
            train_unet_out = up(train_unet_out,
                                control_unet_down_out + trained_unet_down_out,
                                trained_unet_t_emb)

        # Call output layers of trained unet
        train_unet_out = self.trained_unet.norm_out(train_unet_out)
        train_unet_out = nn.SiLU()(train_unet_out)
        train_unet_out = self.trained_unet.conv_out(train_unet_out)
        # out B x C x H x W
        return train_unet_out






