"""Alyssa Riter - StyleGAN2-ada-demo.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1lerS3Kd7lEjnQ-weOVU-DrlCZpAtMxj7
"""

# Commented out IPython magic to ensure Python compatibility.
import os
# %pip install ninja

# clone the repo
if not os.path.exists('stylegan2-ada-pytorch'):
    !git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git

# change the current working directory
os.chdir('stylegan2-ada-pytorch')

"""The script below will generate a zillion lines of error messages, but the results will be generated."""

!python generate.py --outdir=out --seeds=36-70     --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl

# project an existing image of a face in the latent space of the GAN, and then recreate the image based on its latent vector representation

# projecting images to latent space -> find closest latent vector in the GAN's latent space to reproduce the image
!python projector.py --outdir=out --target=/content/stylegan2-ada-pytorch/mytargetimg.png \
   --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl

# to use random walk for rendering the vector
!python random_walk.py --outdir=out \
    --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl

# render the resulting latent vector
!python generate.py --outdir=out --projected-w=out/projected_w.npz \
    --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
