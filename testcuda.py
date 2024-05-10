import torch

# Check if PyTorch is installed
print("PyTorch version:", torch.__version__)

# Check CUDA availability
cuda_available = torch.cuda.is_available()
print("CUDA Available:", cuda_available)

# Print CUDA version and device details if available
if cuda_available:
    print("CUDA Version:", torch.version.cuda)
    print("CUDA Devices:")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No CUDA compatible devices found.")
