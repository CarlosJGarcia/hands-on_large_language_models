**Instalación del entorno** \
conda create --name hands-on_llm_cuda --clone llm_scratch_cuda \
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

pip install mteb \
conda install rich -c conda-forge \
conda install nltk -c conda-forge \
pip install peft \
pip install trl \
pip install langchain \
pip install langchain-community \

**No hacer esto ya que rompe todo el entorno -->** conda install umap-learn -c conda-forge \
**En todo caso, probar esto  -->** pip install umap-learn   



**Sobre la instalación de llama-cpp-python con soporte CUDA** \
Para que la librería use la GPU, es necesario compilarla en local, a partir del código fuente, usando el CUDA toolkit (CUDA Compute Platform) 

**Llama, llama.cpp y ollama** \
ollama está desarrollado usando la librería llama.cpp
- Es una aplicación de línea de comandos (sin GUI) para Linux, MacOS o Windows
- Para interactuar mediante chat con el modelo en ollama se usa Open WebUI, AnythingLLM o alguna extensión de navegador

LM Studio es una alternativa a ollama, también basado en llama.cpp
- ollama se usa mas en servidores y workstations, al ser por línea de comandos
- LM Studio se usa más como aplicación y herramienta interactiva, al ser un GUI

llama.cpp:
- Es una librería en C/C++ para cargar y usar LLMs.
- Los modelos (LLM) deben estar en formato GGUF (GPT-Generated Unified Format)
- Se desarrolló para permitir cargar y usar el modelo Llama de Meta en portátiles con CPU o CPU+GPU. Anteriormente Llama solo funcionaba usando PyTorch en servidores
- Actualmente, tanto llama.cpp como ollama permiten usar cualquier LLM, no solo Llama, sino también Gemma, Mistral, etc.

llama-cpp-python
- Es una librería wrapper de llama.cpp de mas alto nivel, para python. 
- Por este motivo, para usar llama.cpp desde Python, el proceso de instalación de llama-cpp-python es el siguiente:
   1. Se descarga llama-cpp-python con PIP
   2. A continuación PIP, compila el código llama.cpp

       Opción A: Compilación usando el CUDA toolkit, que incluye el compilador nvcc -> llama-cpp-python para GPU 

       Opción B: Compilación con (gcc Linux, clang MacOS o cl.exe Windows) -> llama-cpp-python para CPU
   3. Ya se puede usar la libraria python

\
**Importante:** \
La instalación de Transformers la hago con PIP ya que la versión que instala conda es muy antigua (Transformers v4) y no puede acceder a los modelos del libro (Transformers v5)

Con la instalación de bitsandbytes ocurre lo mismo, la hago con PIP ya que la versión que instala conda es antigua. Además la instalo con el parámetro --upgrade para que pip use la versión más moderna que es la que funciona bien con los drivers CUDA v13.0 que son los actuales en 2026. 

No hacer `$ conda update --all -c conda-forge -y` porque instala una versión de PyTorch sin CUDA

Para comprobar las versiones de librerías y que está funcionando CUDA en PyTorch ejecutar `version.py` \
Para monitorizar el uso de la GPU usar `./monitor.sh` en una ventana de Terminal
