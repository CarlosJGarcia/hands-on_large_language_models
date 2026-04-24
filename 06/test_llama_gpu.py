# Tests llama-cpp-python library with CUDA integration

from llama_cpp import llama_cpp

# This will print the supported backends
print(f"Is CUDA supported: {llama_cpp.llama_supports_gpu_offload()}")