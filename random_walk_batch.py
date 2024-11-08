#!/usr/bin/env python

import os
import sys
import numpy as np
import torch
import legacy
import dnnlib
from time import perf_counter # measures execution time of random walk
import click
from PIL import Image
import subprocess



# based off of projector.py from this github repository https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/projector.py

# function to generate random walk in latent space
def generate_random_walk(
    G, # load styleGAN2
    w_avg: np.ndarray, # used as starting point of random walk
    num_steps: int = 1000,
    step_size: float = 0.01, # change step size to 0.01 set of images that smoothly transition -> play until you get an obvious transition of images (limit on vector norm - if you go outside numpy.norm then tell it to stay within that ball -> put a guardian on it, generate a video of it (ffpeg)) inSIGHTFACE (all we want is detection)
    max_norm: float = None, # this will be computed using w_avg
    device=None
):

    if device is None:
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # start from the average latent vector
    w_current = torch.tensor(w_avg, dtype=torch.float32, device=device) # current latent vector -> intialized as w_avg
    w_out = torch.zeros([num_steps] + list(w_current.shape[1:]), dtype=torch.float32, device=device) # used to store the generated latent vectors

    # calculate norm of w_avg
    w_avg_norm = torch.norm(torch.tensor(w_avg, dtype=torch.float32, device=device))

    # if max_norm is not provided, set it to a multiple of w_avg's norm
    if max_norm is None:
        max_norm = w_avg_norm * 1.5

    if max_norm is not None:
        print(f"w_avg norm: {w_avg_norm:.4f}, setting max_norm to: {max_norm:.4f}")
    else:
        print(f"w_avg norm: {w_avg_norm:.4f}, max_norm not set.")

    for step in range(num_steps):
        # apply random step to the current latent vector
        w_random_step = torch.randn_like(w_current) * step_size # generates a random perturbation from a normal distribution & scales by step size
        w_current = w_current + w_random_step # update the average/current latent vector using the random step
        # Check if the norm of w_current exceeds max_norm
        if torch.norm(w_current) > max_norm:
            w_current = w_current * (max_norm / torch.norm(w_current))  # Scale back to the boundary

        w_out[step] = w_current.detach()  # Store the current latent vector

    return w_out # returns the latent vectors generated

# save the generated latent vectors to .npz files
def save_latent_vectors_as_npz(w_out, outdir):
    os.makedirs(outdir, exist_ok=True)
    # Save the latent vectors at each step as a .npz file
    #for step in range(w_out.shape[0]):
        #np.savez(f'{outdir}/latent_vector_step_{step}.npz', w=w_out[step].cpu().numpy())
    np.savez(f'{outdir}/latent_vectors.npz',w_out.cpu().numpy())

# generate images from latent vectors
def synthesize_write_batch(G,batch,base,outdir):
    batch = torch.stack(batch).squeeze()   # whee!
    with torch.no_grad(): img_batch = G.synthesis(batch, noise_mode='const')
    for i,img in enumerate(img_batch):
        # scale image to [0,255] and convert it to h,w,c indexing
        img = ((img.clamp(-1, 1) + 1) * (255/2)).permute(1,2,0).cpu().numpy().astype(np.uint8)
        img_pil = Image.fromarray(img)
        img_pil.save(f'{outdir}/image_step_{base+i}.png')
        print(f'Saved image for step {base+i}.')
    return

def generate_images_from_latent_vectors(G, latent_vectors, outdir, device, batch_size):
    os.makedirs(outdir, exist_ok=True)
    batch = []
    for step, latent_vector in enumerate(latent_vectors):
        w_tensor = torch.tensor(latent_vector, dtype=torch.float32, device=device)
        w_tensor = w_tensor.unsqueeze(0).repeat([1, G.num_ws, 1])  # repeat w for each layer
        batch.append(w_tensor)
        if (len(batch) == batch_size):
            synthesize_write_batch(G,batch,step-batch_size+1,outdir)
            batch = []
    if len(batch) > 0:
        synthesize_write_batch(G,batch,step-len(batch)+1,outdir)


# these are all command line interface options (implemented in original projector.py script)
# allows users to define StyleGAN2 network, output directory, number of random steps, and the step size
@click.command() # defines the entry point for the command line interface
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True) # path to pretrained network
@click.option('--outdir', 'outdir', help='Where to save the output latent vectors', required=True, metavar='DIR') # directory where output latent vectors will be saved
@click.option('--num-steps', help='Number of random walk steps', type=int, default=1000, show_default=True) # num of random steps
@click.option('--step-size', help='Size of each random step in latent space', type=float, default=0.1, show_default=True) # step size
@click.option('--seed', help='Random seed', type=int, default=303, show_default=True) # seed for reproducability
@click.option('--batch-size', help='Render batch size', type=int, default=10, show_default=True) # batch_size
def run_random_walk(
    network_pkl: str, # network file
    outdir: str, # output directory
    num_steps: int, # number of random steps
    step_size: float, # random step size
    seed: int, # random seed
    batch_size
):
    """Generate a series of latent vectors using a random walk in the latent space of a pretrained network."""
    # random seeds for both NumPy and pyTorch - for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f'Loading networks from "{network_pkl}"...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # check if CUDA is available
    # load the pretrained StyleGAN2
    with dnnlib.util.open_url(network_pkl,cache_dir='.cache') as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device)

    # compute the average latent vector (w_avg) --> allows us to calculate a starting point in latent space
    w_avg_samples = 10000 # 10,000 random latent vectors are generated
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)
    w_avg = np.mean(w_samples, axis=0, keepdims=True)

    # generate latent vectors via random walk
    print(f"Generating {num_steps} latent vectors using random walk...")
    start_time = perf_counter() # measures how long the random walk generation took
    w_out = generate_random_walk(G, w_avg, num_steps, step_size, max_norm=None, device=device) # takes the generator G, the average latent vector, the number of steps, the step size, and the device
    print(f'Elapsed: {(perf_counter()-start_time):.1f} s')

    # save the latent vectors as .npz files
    latent_outdir = os.path.join(outdir, 'latents')
    save_latent_vectors_as_npz(w_out, outdir)

    # convert latent vectors to images
    print("Generating images from latent vectors...")
    generate_images_from_latent_vectors(G, w_out.cpu().numpy(), os.path.join(outdir, 'images'), device, batch_size)

    print("Done.")

if __name__ == "__main__":
    run_random_walk() # pylint: disable=no-value-for-parameter
