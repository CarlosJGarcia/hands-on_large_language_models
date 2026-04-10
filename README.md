**Instalación del entorno** \
conda create --name hands-on_llm_cuda --clone llm_scratch_cuda
conda activate hands-on_llm_cuda \
pip install transformers accelerate huggingface_hub tokenizers \


**Importante:** \
La instalación de transformers la hago con PIP ya que la versión que instala conda es muy antigua (v4) y no puede acceder a los modelos del libro (v5)

No hacer $ conda update --all -c conda-forge -y porque instala una versión de PyTorch sin CUDA \
Para comprobar que está funcionando CUDA en PyTorch ejecutar version.py
