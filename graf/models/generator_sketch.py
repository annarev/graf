import numpy as np
import torch
from torch import nn
from ..utils import sample_on_sphere, look_at, to_sphere
from ..transforms import FullRaySampler
from submodules.nerf_pytorch.run_nerf_mod import render, run_network            # import conditional render
from functools import partial
from . import discriminator_input

def GetEncoderModel(z_dim):
    """Build a model that encodes a sketch feature vector to a latent
    vector that encodes car's shape and appearance.

    We encode from a 512 feature vector that we get from the Resnet-18 model
    pre-trained with SimCLR.
    """
    encoder_layers = [
        nn.Linear(512, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, z_dim),
    ]
    return nn.Sequential(*encoder_layers)


class Generator(object):
    def __init__(self, H, W, z_dim, focal, radius, ray_sampler, render_kwargs_train,
                 render_kwargs_test, parameters, named_parameters,
                 range_u=(0,1), range_v=(0.01,0.49),
                 chunk=None, device='cuda', orthographic=False,
                 encode_from_sketch=True):
        self.device = device
        self.H = int(H)
        self.W = int(W)
        self.z_dim = z_dim
        self.focal = focal
        self.radius = radius
        self.range_u = range_u
        self.range_v = range_v
        self.chunk = chunk
        coords = torch.from_numpy(np.stack(np.meshgrid(np.arange(H), np.arange(W), indexing='ij'), -1))
        self.coords = coords.view(-1, 2)

        self.ray_sampler = ray_sampler
        self.val_ray_sampler = FullRaySampler(orthographic=orthographic)
        self.render_kwargs_train = render_kwargs_train
        self.render_kwargs_test = render_kwargs_test
        self.initial_raw_noise_std = self.render_kwargs_train['raw_noise_std']
        self._parameters = parameters
        self._named_parameters = named_parameters
        self.encode_from_sketch = encode_from_sketch
        self.encoder_net = GetEncoderModel(z_dim)
        
        self.module_dict = {'generator': self.render_kwargs_train['network_fn'],
                'encoder': self.encoder_net}
        for name, module in [('generator_fine', self.render_kwargs_train['network_fine'])]:
            if module is not None:
                self.module_dict[name] = module
        
        for k, v in self.module_dict.items():
            if k in ['generator', 'generator_fine']:
                continue       # parameters already included
            self._parameters += list(v.parameters())
            self._named_parameters += list(v.named_parameters())
        
        self.parameters = lambda: self._parameters           # save as function to enable calling model.parameters()
        self.named_parameters = lambda: self._named_parameters           # save as function to enable calling model.named_parameters()
        self.use_test_kwargs = False

        self.render = partial(render, H=self.H, W=self.W, focal=self.focal, chunk=self.chunk)
        

    # z is a tensor of input sketches
    def __call__(self, z, y=None, rays=None):
        bs = z.shape[0]
        if rays is None:
            rays = torch.cat([self.sample_rays() for _ in range(bs)], dim=1)

        render_kwargs = self.render_kwargs_test if self.use_test_kwargs else self.render_kwargs_train
        render_kwargs = dict(render_kwargs)        # copy
    
        # in the case of a variable radius
        # we need to adjust near and far plane for the rays
        # so they stay within the bounds defined wrt. maximal radius
        # otherwise each camera samples within its own near/far plane (relative to this camera's radius)
        # instead of the absolute value (relative to maximum camera radius)
        if isinstance(self.radius, tuple):
            assert self.radius[1] - self.radius[0] <= render_kwargs['near'], 'Your smallest radius lies behind your near plane!'
    
            rays_radius = rays[0].norm(dim=-1)
            shift = (self.radius[1] - rays_radius).view(-1, 1).float()      # reshape s.t. shape matches required shape in run_nerf
            render_kwargs['near'] = render_kwargs['near'] - shift
            render_kwargs['far'] = render_kwargs['far'] - shift
            assert (render_kwargs['near'] >= 0).all() and (render_kwargs['far'] >= 0).all(), \
                (rays_radius.min(), rays_radius.max(), shift.min(), shift.max())
            

        if self.encode_from_sketch:
          # z is a vector of sketch features. We need to encode it to latent vector
          # first.
          render_kwargs['features'] = self.encoder_net.forward(z)
        else:
          render_kwargs['features'] = z

        rgb, disp, acc, extras = render(self.H, self.W, self.focal, chunk=self.chunk, rays=rays,
                                        **render_kwargs)

        rays_to_output = lambda x: x.view(len(x), -1) * 2 - 1      # (BxN_samples)xC

        output_rgb = rays_to_output(rgb)
        
        if self.use_test_kwargs:               # return all outputs
            return output_rgb, \
                   rays_to_output(disp), \
                   rays_to_output(acc), extras

        # During training, return both the generated image and the input sketch features
        # so that both get passed to the discriminator.
        return discriminator_input.DiscriminatorInput(output_rgb, z)

    def decrease_nerf_noise(self, it):
        end_it = 5000
        if it < end_it:
            noise_std = self.initial_raw_noise_std - self.initial_raw_noise_std/end_it * it
            self.render_kwargs_train['raw_noise_std'] = noise_std

    def sample_pose(self):
        # sample location on unit sphere
        loc = sample_on_sphere(self.range_u, self.range_v)
        
        # sample radius if necessary
        radius = self.radius
        if isinstance(radius, tuple):
            radius = np.random.uniform(*radius)

        loc = loc * radius
        R = look_at(loc)[0]

        RT = np.concatenate([R, loc.reshape(3, 1)], axis=1)
        RT = torch.Tensor(RT.astype(np.float32))
        return RT

    def sample_rays(self):
        pose = self.sample_pose()
        sampler = self.val_ray_sampler if self.use_test_kwargs else self.ray_sampler
        batch_rays, _, _ = sampler(self.H, self.W, self.focal, pose)
        return batch_rays

    def to(self, device):
        self.render_kwargs_train['network_fn'].to(device)
        self.encoder_net.to(device)
        if self.render_kwargs_train['network_fine'] is not None:
            self.render_kwargs_train['network_fine'].to(device)
        self.device = device
        return self

    def train(self):
        self.use_test_kwargs = False
        self.encoder_net.train()
        self.render_kwargs_train['network_fn'].train()
        if self.render_kwargs_train['network_fine'] is not None:
            self.render_kwargs_train['network_fine'].train()

    def eval(self):
        self.use_test_kwargs = True
        self.encoder_net.eval()
        self.render_kwargs_train['network_fn'].eval()
        if self.render_kwargs_train['network_fine'] is not None:
            self.render_kwargs_train['network_fine'].eval()
