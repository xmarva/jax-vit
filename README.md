# jax-lung-cancer-classifier

Classifying chest CT scans into four categories: Adenocarcinoma, Large cell carcinoma, Squamous cell carcinoma, and normal tissue. I use both PyTorch and JAX/Flax to compare performance and quality metrics.

## Dataset

The project uses a chest cancer CT scan dataset containing approximately 1000 images across four classes. Images are in jpg/png format with a 70/20/10 train/test/validation split. Each image is classified into one of three cancer types or normal tissue.

Dataset: https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images