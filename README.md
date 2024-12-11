# random-walk
Repository for undergraduate research at The University of Notre Dame regarding face generation and its presence in vector spaces.

# StyleGAN2 Demo
This was iterated upon from the work found in this github repository: https://github.com/NVlabs/stylegan2-ada-pytorch  
  colab notebook: https://colab.research.google.com/drive/1lerS3Kd7lEjnQ-weOVU-DrlCZpAtMxj7

# InsightFace Embeddings
colab notebook: https://colab.research.google.com/drive/17Q9YKvppBBReLsaPjW-eGGH-IH4bqrLs

# Instructions For Running Face Generation
StyleGAN2-ada part:
  after cloning the styleGAN2 github repo:
  cd stylegan2-ada-pytorch
  docker build -t sg2ada:latest .
  bash docker_run.sh python playground/random_walk_batch.py --outdir=out --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl --batch-size=50
  Can do the same thing for non-batch script

# Instructions For Running Face Detector
InsightFace:
  python3.9 -m venv insightface
  cd insightface
  source bin/activate to activate the env in your shell
  pip install insightface
  pip install onnxruntime-gpu
  mkdir playground
  cd playground
