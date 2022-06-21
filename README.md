# TVL2_GPU
Compile in case Cmake does not work:
clang++-13 main.cu -o a --cuda-gpu-arch=sm_72 -L/usr/local/cuda/lib64 -lcudart -ldl -lrt -pthread  --cuda-path=/usr/local/cuda --no-cuda-version-check -std=c++20 


