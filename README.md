**Instalación del entorno** \
conda create --name hands-on_llm_cuda --clone llm_scratch_cuda
conda activate hands-on_llm_cuda \
conda install -c conda-forge sentencepiece \
conda install -c conda-forge protobuf \
conda install -c conda-forge gensim \
conda install -c conda-forge scikit-learn \
pip install transformers accelerate huggingface_hub tokenizers \
pip install --upgrade bitsandbytes \
pip install datasets \
pip install zstandard lz4 \
pip install sentence-transformers \
pip install openai

**Instalación de llama-cpp-python con soporte CUDA** \
sudo apt update \
sudo apt upgrade -y \
sudo apt install -y build-essential cmake \
export CMAKE_ARGS="-DGGML_CUDA=on" \
export FORCE_CMAKE=1 \
pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python \


**Sobre la instalación de llama-cpp-python con soporte CUDA** \
Para que la librería use la GPU, es necesario compilarla en local, a partir del código fuente, usando el CUDA toolkit (CUDA Compute Platform) \

**Llama, llama.cpp y ollama** \
ollama está desarrollado usando la librería llama.cpp \
llama.cpp se desarrolló para permitir cargar y usar el modelo Llama de Meta en portátiles con CPU o CPU+GPU. Anteriormente Llama solo funcionaba usando PyTorch en servidores \
llama.cpp es la librería core en C/C++ para cargar y usar LLMs. llama-cpp-python es un wrapper de mas alto nivel en python.
Por este motivo, se descarga llama-cpp-python con PIP, se compila el código llama.cpp usando el CUDA toolkit y ya se puede usar la libraria python
Actualmente, tanto llama.cpp como ollama permiten usar cualquier LLM, no solo Llama, sino también Gemma, Mistral, etc. \


**Importante:** \
La instalación de transformers la hago con PIP ya que la versión que instala conda es muy antigua (v4) y no puede acceder a los modelos del libro (v5)

Con la instalación de bitsandbytes ocurre lo mismo, la hago con PIP ya que la versión que instala conda es antigua. Además la instalo con el parámetro --upgrade para que pip use la versión más moderna que es la que funciona bien con los drivers CUDA v13.0 que son los actuales en 2026. 

No hacer $ conda update --all -c conda-forge -y porque instala una versión de PyTorch sin CUDA

Para comprobar las versiones de librerías y que está funcionando CUDA en PyTorch ejecutar `version.py` \
Para monitorizar el uso de la GPU usar `./monitor.sh` en una ventana de Terminal


| GPU Category | Model | Key Advantage for AI |
| :--- | :--- | :--- |
| **Enthusiast / Prosumer** | **RTX 3090 24GB** | Gold standard for local development in 2022-2023.|
| **Enthusiast / Prosumer** | **RTX 4090 24GB** | Gold standard for local development in 2023-2025. Large LLM fine-tuning and prototyping before moving to enterprise hardware. |
| **New Professional Standard** | **RTX 5090 32GB** | Current top-tier consumer and data scientist choice in 2026, offering more VRAM and significant performance gains. |
| **Budget / Student** | **RTX 3060 12GB / 5060 Ti 16GB** | Remain popular in 2026 due to their VRAM-to-price ratio, allowing students and hobbyists to run LLM inference and small-scale training projects locally without a massive investment. These two together are the equivalent of a RTX 4090 with 24 GB.|
