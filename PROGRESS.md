# Learning Progress Tracker

| Chapter | Completion Status | Date Completed | Notes |
|---------|------------------|----------------|-------|
| *Section 1 - Understanding Language Models* |
| 1. Introduction | Completed | 03/Apr/2026 | - |
| 2. Tokenization and embeddings | Completed | 12/Apr/2026 | - |
| 3. Looking into LLMs | Completed | 13/Apr/2026 | - |
| *Section 2 - Using Pretrained Language Models* |
| 4. Text classification | Not started | - | - |
---

### Notes
- Frontier model: Los modelos 'tope de gama' que se ejecutan en un datacenter y a los que se accede mediante un API a través de internet (Gemini, Claude, ChatGPT)

- Edge model: Los modelos compactos pensados para ejecutarse localmente en un smartphone o en un ordenador, incluso con hardware dedicado (RTX 3060 12 GB, RTX 5060 16 GB, etc)

- Representation model: Normalmente son modelos no-generativos. Entienden el sentido de un texto y lo pueden clasificar (análisis de sentimientos, búsquedas en el documento, es spam o no-spam)


### Librerías

- Data Science:
    - scipy, pandas

- Deep Learning:
    - keras, tensorflow
    - pytorch

- Large Language Models:
    - Core
        - Transformers: Modelos
        - Accelerate  : Bridge  CPU -> GPU
        - bitsandbytes: Quantization
    - Datasets
        - datasets       : datasets
        - zstandard y lz4: unzipper
    - Tokenización
        - tokenizer
        - tiktoker
        - sentencepiece y protobuf: usados por Google y Llama
    - Varios
        - gensim: Tareas 'clasicas' de NLP (Natural Language Processing) tipo Word2Vec
