# random-walk
Repository for undergraduate research at The University of Notre Dame regarding face generation and its presence in vector spaces.

# StyleGAN2 Demo
This was iterated upon from the work found in this github repository: https://github.com/NVlabs/stylegan2-ada-pytorch  
  colab notebook: https://colab.research.google.com/drive/1lerS3Kd7lEjnQ-weOVU-DrlCZpAtMxj7

# InsightFace Embeddings
colab notebook: https://colab.research.google.com/drive/17Q9YKvppBBReLsaPjW-eGGH-IH4bqrLs

# Instructions For Running Face Generation
StyleGAN2-ada part:<br />
  after cloning the styleGAN2 github repo:<br />
  cd stylegan2-ada-pytorch<br />
  docker build -t sg2ada:latest .<br />
  bash docker_run.sh python playground/random_walk_batch.py --outdir=out --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl --batch-size=50<br />
  Can do the same thing for non-batch script<br />

# Instructions For Running Face Detector
InsightFace:<br />
  python3.9 -m venv insightface<br />
  cd insightface<br />
  source bin/activate to activate the env in your shell<br />
  pip install insightface<br />
  pip install onnxruntime-gpu<br />
  mkdir playground<br />
  cd playground<br />

  # CSV files contains the image_step_*.png, confidence scores, vector embeddings
