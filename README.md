**Instalación del entorno** \
conda create --name hands-on_llm_cuda --clone llm_scratch_cuda
conda activate hands-on_llm_cuda \
conda install -c conda-forge sentencepiece \
conda install -c conda-forge protobuf \
conda install -c conda-forge gensim \
pip install transformers accelerate huggingface_hub tokenizers \
pip install --upgrade bitsandbytes \


**Importante:** \
La instalación de transformers la hago con PIP ya que la versión que instala conda es muy antigua (v4) y no puede acceder a los modelos del libro (v5) \
Con la instalación de bitsandbytes ocurre lo mismo, la hago con PIP ya que la versión que instala conda es antigua. Además la instalo con el parámetro --upgrade para que pip use la versión más moderna que es la que funciona bien con los drivers CUDA v13.0 que son los actuales en 2026. \


No hacer $ conda update --all -c conda-forge -y porque instala una versión de PyTorch sin CUDA \

Para comprobar las versiones de librerías y que está funcionando CUDA en PyTorch ejecutar `version.py` \
Para monitorizar el uso de la GPU usar `./monitor.sh` en una ventana de Terminal
