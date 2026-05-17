# Learning Progress Tracker

| Chapter | Completion Status | Date Completed | Notes |
|---------|------------------|----------------|-------|
| *Section 1 - Understanding Language Models* |
| 1. Introduction | Completed | 03/Apr/2026 | - |
| 2. Tokenization and embeddings | Completed | 12/Apr/2026 | - |
| 3. Looking into LLMs | Completed | 13/Apr/2026 | - |
| *Section 2 - Using Pretrained Language Models* |
| 4. Text classification | Completed | 19/Apr/2026 | - |
| 5. Text clustering and topic modeling | Started | - | Library issues |
| 6. Prompt engineering | Completed | 24/Apr/2026 | - |
| 7. Advanced text generation techniques and tools | Ongoing | - | - |
| 8. Semantic Search and Retrieval - Augmented Generation| Completed | 17/May/2026 | - |
| 9. Multimodal Large Language Models | Not started | - | - |
| *Section 3 - Training and Fine-Tuning Language Models* |
| 10. Creating Text Embedding Models | Completed | 27/Apr/2026 | - |
| 11. Fine-Tuning Representation Models for Classification | Not started | | - |
| 12. Fine-Tuning Generation Models | Completed | 28/Apr/2026 | - |


### Librerías

- Data Science:
    - numpy, scipy, pandas

- Visualization:
    - matplotlib: estático
    - plotly    : interactivo

- Traditional Machine Learning:
    - sci-kit-learn: random forests, no GPU

- Deep Learning:
    - keras, tensorflow: Google
    - pytorch          : Meta (Facebook)

- Large Language Models:
    - Core
        - transformers: Modelos de Hugging Face
        - accelerate  : Bridge  CPU -> GPU de Hugging Face
        - bitsandbytes: Quantization. La llama transformers cuando al cargar un modelo indicamos el parámetro load_in_4bit=True

    - Datasets:
        - datasets       : datasets de Hugging Face
        - zstandard y lz4: unzipper, usado por datasets

    - Tokenización:
        - tokenizers     : tokenizer de Hugging Face
        - tiktoker       : tokenizer de OpenAI (ChatGPT)
        - sentencepiece  : tokenizer de Google
        - protobuf       : serializador de Google

    - Varios:
        - gensim: Tareas 'clasicas' de NLP (Natural Language Processing) tipo Word2Vec

