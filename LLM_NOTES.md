**GPUs** \

| GPU Category | Model | Key Advantage for AI |
| :--- | :--- | :--- |
| **Enthusiast / Prosumer** | **RTX 3090 24GB** | Gold standard for local development in 2022-2023.|
| **Enthusiast / Prosumer** | **RTX 4090 24GB** | Gold standard for local development in 2023-2025. Large LLM fine-tuning and prototyping before moving to enterprise hardware. |
| **New Professional Standard** | **RTX 5090 32GB** | Current top-tier consumer and data scientist choice in 2026, offering more VRAM and significant performance gains. |
| **Budget / Student** | **RTX 3060 12GB / 5060 Ti 16GB** | Remain popular in 2026 due to their VRAM-to-price ratio, allowing students and hobbyists to run LLM inference and small-scale training projects locally without a massive investment. These two together are the equivalent of a RTX 4090 with 24 GB.|

| GPU Category | Architecture |
| :--- | :--- |
| **30-series** | **Ampere** |
| **40-series** | **Ada Lovelace** |
| **50-series** | **Blackwell** | Current top-tier consumer and data scientist choice in 2026, offering more VRAM and significant performance gains. |



**Tipos de modelos** \

An LLM is essentially a massive Transformer neural network.


| | | | |
|---------|------------------|----------------|-------|
| **Frontier model** | 100B to Trillions | Google Gemini | Los modelos 'tope de gama' que se ejecutan en un datacenter y a los que se accede mediante un API a través de Internet (Gemini, Claude, ChatGPT) |
| **Workstation model** | 8B to 70B | Meta's Llama 3 | Run on desktop computers with dedicated harware or high memory machines like Mac Studio or DGX Sparc |
| **Edge model** | <1B to 8B | Microsoft Phi-3 | Los modelos compactos pensados para ejecutarse localmente en un smartphone o en un ordenador, incluso con hardware dedicado (RTX 3060 12 GB, RTX 5060 16 GB, etc.) |

- Meta Llama 3 8B Unquantized + Context  - 24 GB RAM
- Meta Llama 3 70B Unquantized + Context - 160 GB RAM 
- Gemma 4 E4B: Edge model. 4 Billion parameters
- Gemma 4 12B: Workstation model. 12 Billion parameters. But requires quantization to work in 16 GB VRAM or RAM. \


- Para ejecutar los modelos workstation y tener buen rendimiento en inferencia se usa el Mac Studio con 192 GB de unified memory y ancho de banda de memoria 800GB/s
- Para tener buen rendimiento en prompting, tokenización y análisis de datos (RAG) se una Nvidia DGX Sparc con 128 GB de unified memory y ancho de banda 270GB/s
- Como alternativa, se construyen workstations-frankenstein con varias tarjetas RTX en paralelo
- DGX Sparc es más rápido en tokenización. Mac Studio es mas rápido en inferencia ya que tiene más ancho de banda
- Ancho de banda VRAM RTX 5090: 1.79 TB/s (más del doble que el Mac Studio de 2026)
- Ancho de banda VRAM RTX 3060: 360 GB/s (menos de la mitad que el Mac Studio 2026)


| | Representational Models | Embedding Models | Generative Models |
|---------|------------------|----------------|-------|
| Primary Goal | Son modelos no-generativos. Son modelos "decodificadores". Transforman los datos de entrada (imágenes, texto, audio, etc.) en un tensor numérico. A partir de ese tensor podemos clasificar los datos (análisis de sentimientos, búsquedas en documentos, spam / no-spam). | Mapean los datos a puntos en el espacio basados en parecidos. Permiten agrupar los datos (por ejemplo las palabras "rey" y "reina") por múltiples criterios | Generan una salida basándose en patrones aprendidos. Son modelos diseñados para crear contenido nuevo a partir de una instrucción (prompt). Entienden el sentido de un texto (necesario para 'entender' el prompt) y predicen la continuación más probable, para generar respuestas, código, resúmenes o traducciones |
| Output | Características del input |Vector numérico de longitud fija | Token, imagen, vídieo |
| Philosophy | ¿Que es esto? | ¿Donde encaja esto? | ¿Qué viene a continuación? |
| Example | BERT | Word2Vec | GPT-4 |


