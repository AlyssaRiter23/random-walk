# random-walk
Repository for undergraduate research at The University of Notre Dame regarding face generation and its presence in vector spaces.

# Important Notes - InsightFace 
1. The scripts named 2025_*.csv represent the image step, confidence score, and vector embedding for the output of 500 images using the random_walk_batch.py script with varying seeds using the FRGC network.
2. The figures named confidence_scores_dotplot_2025_*.png represent the confidence scores at each image step for a single seed from the same output mentioned above.
3. The figures named confidence_scores_histogram_2025_* represent a histogram of those confidence scores mentioned above.
4. The scripts named outfile*_embed.csv contain the image step, confidence scores, and embeddings for 1000 steps using the random_walk_batch.py script with varying seeds with the original FFHQ network.
5. The figures named confidence_scores_histogram_embed*.png, and confidence_scores_plt*.png represent the confidence scores and detections for those 1000 images mentioned in step 4.
   
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

  # CSV files information
  Contains the image_step_*.png, confidence scores, vector embeddings <br />
