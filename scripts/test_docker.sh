echo "Testing Docker image..."
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"