#!/usr/bin/env python

import os
import numpy as np
import torch
import legacy
import dnnlib
from PIL import Image
import click
import pandas as pd
from datetime import datetime
import pytz

# ------------------------------
# Sample from hypersphere in Z-space
# ------------------------------
def sample_z_hypersphere(z_start, num_samples=10, radius=0.5):
    samples = []
    for _ in range(num_samples):
        # pick a random direction
        direction = np.random.randn(*z_start.shape).astype(np.float32)
        # normalize it to a unit length
        direction /= np.linalg.norm(direction)
        # step from z_start along radius in that direction
        sample = z_start + radius * direction
        samples.append(sample)
    return samples

# ------------------------------
# Save image from latent vector
# ------------------------------
def synthesize_images(G, w, outdir, identity_id, step_id, radius, device):
    # ensure tensor
    w_tensor = torch.from_numpy(w).to(device).to(torch.float32)

    # G.mapping already returns shape [N, num_ws, w_dim]
    img = G.synthesis(w_tensor, noise_mode='const')

    # convert to properly formatted numpy image
    img = (img.clamp(-1, 1) + 1) * (255 / 2)
    img = img.permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()[0]

    os.makedirs(outdir, exist_ok=True)
    image_path = os.path.join(
        outdir,
        f"id{identity_id:03d}_r{radius:.3f}_step{step_id:03d}.png"
    )
    Image.fromarray(img, 'RGB').save(image_path)
    return image_path

# ------------------------------
# command line arguments
# ------------------------------
@click.command()
@click.option('--network', 'network_pkl', required=True, help='Path to pretrained StyleGAN2 network (.pkl)')
@click.option('--outdir', 'outdir', required=True, help='Directory to save generated images + latents')
@click.option('--num-identities', type=int, default=5, help='Number of synthetic identities')
@click.option('--num-steps', type=int, default=20, help='Number of samples per identity per radius')
@click.option('--radii', multiple=True, type=float,
              default=[0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0],
              help='List of radii to sweep')
@click.option('--seed', type=int, default=303, help='Random seed for reproducibility')
def main(network_pkl, outdir, num_identities, num_steps, radii, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    est = pytz.timezone('America/New_York')
    timestamp = datetime.now(est).strftime('%Y_%m_%d_%H_%M')
    outdir = os.path.join(outdir, timestamp)
    os.makedirs(outdir, exist_ok=True)

    # load the network (FRGC or ffhq)
    print(f"Loading network from {network_pkl}...")
    with dnnlib.util.open_url(network_pkl, cache_dir='.cache') as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device).eval()
        G = G.to(torch.float32)
        G.synthesis.float()

    manifest = []

    for identity_id in range(num_identities):
        # anchor latent in Z
        z_base = np.random.randn(1, G.z_dim).astype(np.float32)

        for radius in radii:
            z_samples = sample_z_hypersphere(
                z_base, num_samples=num_steps, radius=radius
            )

            for step_id, z_sample in enumerate(z_samples):
                # map Z onto W
                z_tensor = torch.from_numpy(z_sample).to(device)
                w_tensor = G.mapping(z_tensor, None)  # [1, num_ws, w_dim]
                w = w_tensor.cpu().numpy()

                # save Z and W probes
                probe_dir = os.path.join(outdir, "latents")
                os.makedirs(probe_dir, exist_ok=True)

                z_path = os.path.join(
                    probe_dir,
                    f"id{identity_id:03d}_r{radius:.3f}_step{step_id:03d}_z.npy"
                )
                w_path = os.path.join(
                    probe_dir,
                    f"id{identity_id:03d}_r{radius:.3f}_step{step_id:03d}_w.npy"
                )
                np.save(z_path, z_sample)
                np.save(w_path, w)

                # synthesize the image and save it
                image_path = synthesize_images(
                    G, w, outdir, identity_id, step_id, radius, device
                )

                manifest.append({
                    "identity": identity_id,
                    "step": step_id,
                    "radius": radius,
                    "image_file": image_path,
                    "z_file": z_path,
                    "w_file": w_path,
                })

            print(f"Generated identity {identity_id} at radius={radius} ({num_steps} images)")

    # save manifest
    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(os.path.join(outdir, 'probes_manifest.csv'), index=False)
    print(f"Saved manifest CSV with {len(manifest)} images and latent vectors.")

if __name__ == "__main__":
    main()




