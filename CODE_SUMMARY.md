| File | Libraries | Model_ID |
| :--- | :--- | :--- |
| ./version.py | sys, torch, openai, tiktoken, importlib.metadata, tokenizers, transformers, llama_cpp, llama_cpp.llama | |
| ./01/01.py | transformers, AutoModelForCausalLM, AutoTokenizer, pipeline, GenerationConfig | microsoft/Phi-3-mini-4k-instruct |
| ./06/test_llama_gpu.py | llama_cpp | |
| ./06/06-01-gemma.py | os, openai | |
| ./06/06-02.py | json, warnings, llama_cpp.llama | *fp16.gguf |
| ./06/06-01.py | torch, transformers, GenerationConfig, AutoModelForCausalLM, AutoTokenizer, pipeline | |
| ./04/04-03.py | numpy, tqdm, datasets, sklearn.metrics, transformers.pipelines.pt_utils, transformers, AutoModelForSequenceClassification, AutoTokenizer, pipeline | rotten_tomatoes |
| ./04/04-04.py | numpy, datasets, sklearn.metrics, sklearn.linear_model, sklearn.metrics.pairwise, sentence_transformers | rotten_tomatoes |
| ./04/04-02.py | time, datasets, transformers, pipeline, AutoModelForSequenceClassification, AutoTokenizer | rotten_tomatoes |
| ./04/04-05-pipeline.py | torch, tqdm, datasets, sklearn.metrics, transformers.pipelines.pt_utils, transformers, pipeline, AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM | rotten_tomatoes |
| ./04/04-06-ollama-gemma4.py | openai | |
| ./04/04-06.py | openai | |
| ./04/04-05.py | torch, datasets, transformers, AutoTokenizer, AutoModelForSeq2SeqLM | rotten_tomatoes |
| ./04/04-01.py | datasets | rotten_tomatoes |
| ./02/02-03.py | gensim.downloader | |
| ./02/02-02.py | transformers, AutoModel, AutoTokenizer | microsoft/deberta-v3-xsmall |
| ./02/02-04.py | numpy, pandas, urllib.request, gensim.models | |
| ./02/02-01.py | transformers, AutoModelForCausalLM, AutoTokenizer | microsoft/Phi-3-mini-4k-instruct |
| ./12/12-1.py | torch, rich.console, datasets, trl, SFTTrainer, SFTConfig, transformers, pipeline, AutoTokenizer, TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig, peft, AutoPeftModelForCausalLM, LoraConfig, prepare_model_for_kbit_training, get_peft_model | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| ./12/12-2.py | warnings, sys, types, torch, trl, DPOConfig, DPOTrainer, rich.console, datasets, peft, PeftModel, AutoPeftModelForCausalLM, transformers, BitsAndBytesConfig, AutoTokenizer, LoraConfig, prepare_model_for_kbit_training, get_peft_model | |
| ./03/03-01.py | time, torch, transformers, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline | microsoft/Phi-3-mini-4k-instruct |
| ./03/03-01-gemma4.py | torch, transformers, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline, BitsAndBytesConfig | |
| ./03/03-01-mistral.py | torch, transformers, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline, BitsAndBytesConfig | |
| ./10/10-3.py | numpy, pandas, tqdm, rich.console, datasets, sentence_transformers, InputExample, CrossEncoder, losses, SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, CrossEncoderTrainer, CrossEncoderTrainingArguments, EmbeddingSimilarityEvaluator | |
| ./10/10-2.py | rich.console, datasets, sentence_transformers, losses, EmbeddingSimilarityEvaluator, SentenceTransformerTrainingArguments, SentenceTransformerTrainer | |
| ./10/test-rich.py | rich.table, rich.console | |
| ./10/10-1.py | mteb, warnings, datasets, sentence_transformers, losses, EmbeddingSimilarityEvaluator, SentenceTransformerTrainingArguments, SentenceTransformerTrainer | |
| ./10/10-4.py | nltk, tqdm, rich.console, datasets, sentence_transformers, losses, DenoisingAutoEncoderDataset, EmbeddingSimilarityEvaluator, models, SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments | |
| ./05/05-1.py | umap, rich.console, datasets, sentence_transformers | |
