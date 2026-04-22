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

**Instalacion de la librería llama-cpp-python con soporte CUDA** \
Para que la librería use la GPU, es necesario compilar la librearí a partir del código fuente en local, usando con la versión instalada del CUDA toolkit (CUDA Compute Platform) \
ollama está desarrollado usando la librería llama-cpp \
llama-cpp se desarrolló para permitir cargar y usar el modelo Llama de Meta en portátiles con CPU o CPU+GPU. Anteriormente Llama solo funcionaba usando PyTorch en servidores \
Actualmente, tanto llama.cpp como ollama permiten usar cualquier modelo, no solo Llama, sino también Gemma, Mistral \

$ sudo apt update \
$ sudo apt upgrade -y \
$ sudo apt install -y build-essential cmake \
$ export CMAKE_ARGS="-DGGML_CUDA=on" \
$ export FORCE_CMAKE=1 \
$ pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python \


**Importante:** \
La instalación de transformers la hago con PIP ya que la versión que instala conda es muy antigua (v4) y no puede acceder a los modelos del libro (v5)

Con la instalación de bitsandbytes ocurre lo mismo, la hago con PIP ya que la versión que instala conda es antigua. Además la instalo con el parámetro --upgrade para que pip use la versión más moderna que es la que funciona bien con los drivers CUDA v13.0 que son los actuales en 2026. 

No hacer $ conda update --all -c conda-forge -y porque instala una versión de PyTorch sin CUDA

Para comprobar las versiones de librerías y que está funcionando CUDA en PyTorch ejecutar `version.py` \
Para monitorizar el uso de la GPU usar `./monitor.sh` en una ventana de Terminal
