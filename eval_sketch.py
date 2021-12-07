"""Binary that evaluates a GRAF model conditioned on an input sketch-like image.

The code in this file is based on eval.py and modified to condition on
a sketch.
"""
import argparse
import os
from os import path
import numpy as np
import time
import copy
import csv
import cv2
import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor')
from torchvision.utils import save_image

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib
matplotlib.use('Agg')

import sys
sys.path.append('submodules')        # needed to make imports work in GAN_stability

from graf.gan_training import Evaluator as Evaluator
from graf.config import get_data, build_models, update_config, get_render_poses
from graf.utils import count_trainable_parameters, to_phi, to_theta, get_nsamples_with_sketch
from graf.transforms import ImgToPatch
from graf.models import simclr
from graf.models import sketch_feature_extractor

from submodules.GAN_stability.gan_training.metrics import FIDEvaluator
from submodules.GAN_stability.gan_training.checkpoints import CheckpointIO
from submodules.GAN_stability.gan_training.distributions import get_ydist, get_zdist
from submodules.GAN_stability.gan_training.config import (
    load_config,
)

from external.colmap.filter_points import filter_ply

def compute_fid_kid(test_loader, evaluator, eval_dir, sketch_feature_extr):
    num_test_examples = 498
    n_samples = num_test_examples
    real, sketch, small_sketch = get_nsamples_with_sketch(test_loader, n_samples) 
    sketch_features = sketch_feature_extr.get_features(small_sketch.to(device))

    samples, _, _ = evaluator.create_samples(sketch_features)
    samples = samples.clamp_(-1, 1) 

    # Compute FID and KID
    fid_cache_file = os.path.join(out_dir, 'fid_cache_train.npz')
    kid_cache_file = os.path.join(out_dir, 'kid_cache_train.npz')

    target_examples = torch.split(real, batch_size, dim=0)
    generated_examples = torch.split(samples, batch_size, dim=0)
    inception_eval = FIDEvaluator(
          device=device,
          batch_size=batch_size,
          resize=True,
          n_samples=len(target_examples) * batch_size,
          n_samples_fake=len(generated_examples) * batch_size,
    )
    
    inception_eval.initialize_target(
            target_examples, cache_file=fid_cache_file, act_cache_file=kid_cache_file)
    
    fid, (kids, vars) = inception_eval.get_fid_kid(generated_examples)
    kid = np.mean(kids)

    filename = 'fid_kid.csv'
    outpath = os.path.join(eval_dir, filename)
    with open(outpath, mode='w') as csv_file:
        fieldnames = ['fid', 'kid']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow({'fid': fid, 'kid': kid})

    print('Saved FID ({:.1f}) and KIDx100 ({:.2f}) to {}.'.format(fid, kid*100, outpath))


def save_samples(test_loader, evaluator, eval_dir):
    n_samples = 1
    real, sketch, small_sketch = get_nsamples_with_sketch(test_loader, n_samples) 
    sketch_features = sketch_feature_extr.get_features(small_sketch.to(device))
    real = real / 2  + 0.5  # Change to 0 - 1 range
    sketch = sketch / 2 + 0.5
    samples, _, _ = evaluator.create_samples(sketch_features)
    samples = samples.clamp_(-1, 1) / 2 + 0.5
    target_examples = torch.split(real, 1, dim=0)
    generated_examples = torch.split(samples, 1, dim=0)
    sketch_examples = torch.split(sketch, 1, dim=0)

    out_dir = os.path.join(eval_dir, 'examples')
    if not os.path.exists(out_dir):
      os.makedirs(out_dir)
    print('Writing %i images to: %s' % (len(target_examples), out_dir))

    for i in range(len(target_examples)):
      target_path = os.path.join(out_dir, 'target%i.png' % i)
      sketches_path = os.path.join(out_dir, 'sketch%i.png' % i)
      generated_path = os.path.join(out_dir, 'generated%i.png' % i)

      save_image(target_examples[i], target_path);
      save_image(sketch_examples[i], sketches_path);
      save_image(generated_examples[i], generated_path);


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate the GRAF model conditioned on a sketch.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--fid_kid', action='store_true', help='Evaluate FID and KID.')
    parser.add_argument('--save_samples', action='store_true', help='Store sample images.')

    args, unknown = parser.parse_known_args()
    config = load_config(args.config, 'configs/default.yaml')
    config['data']['fov'] = float(config['data']['fov'])
    config = update_config(config, unknown)

    num_test_examples = 498
    batch_size = num_test_examples / 6
    out_dir = os.path.join(config['training']['outdir'], config['expname'])
    checkpoint_dir = path.join(out_dir, 'chkpts')
    eval_dir = os.path.join(out_dir, 'eval')
    os.makedirs(eval_dir, exist_ok=True)
    fid_kid = int(args.fid_kid)

    config['training']['nworkers'] = 0

    checkpoint_io = CheckpointIO(
        checkpoint_dir=checkpoint_dir
    )

    device = torch.device("cuda:0")

    # Dataset
    config['data']['datadir'] = config['data']['test_datadir']

    test_dataset, hwfr, render_poses = get_data(config)

    config['data']['hwfr'] = hwfr         # add for building generator
    
    test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=config['training']['nworkers'],
            shuffle=False, pin_memory=False, sampler=None, drop_last=False   # enable shuffle for fid/kid computation
    )

    # Create models
    generator, _ = build_models(config, disc=False)
    print('Generator params: %d' % count_trainable_parameters(generator))

    # Put models on gpu if needed
    generator = generator.to(device)

    # Register modules to checkpoint
    checkpoint_io.register_modules(
        **generator.module_dict  # treat NeRF specially
    )

    # Distributions
    ydist = get_ydist(1, device=device)         # Dummy to keep GAN training structure in tact
    y = torch.zeros(batch_size)                 # Dummy to keep GAN training structure in tact
    
    # Evaluator
    zdist = None  # We use sketches instead of sampling z
    evaluator = Evaluator(fid_kid, generator, zdist, ydist,
                          batch_size=batch_size, device=device)

    # Load checkpoint
    print('Loading %s' % os.path.join(checkpoint_dir, model_file))
    model_file = 'model_best.pt'
    load_dict = checkpoint_io.load(model_file)
    it = load_dict.get('it', -1)
    epoch_idx = load_dict.get('epoch_idx', -1)
    print('Loaded epoch_idx = %i' % epoch_idx)

    sketch_feature_extr = sketch_feature_extractor.SketchFeatureExtractor(
            config['data']['resnet18_checkpoint'], device)

    # Evaluation loop
    if args.fid_kid:
        compute_fid_kid(test_loader, evaluator, eval_dir, sketch_feature_extr)
    if args.save_samples:
        save_samples(test_loader, evaluator, eval_dir, sketch_feature_extr)

