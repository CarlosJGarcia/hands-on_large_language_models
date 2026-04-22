# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview
This repository contains a collection of Python scripts and notebooks demonstrating various techniques and libraries for working with Large Language Models (LLMs), including Hugging Face Transformers, PyTorch, and OpenAI. The content is organized into chapters (01, 02, 03, 04, 06) representing different learning modules.

## Common Commands
- **Check environment and CUDA status**: `python version.py`
- **Monitor GPU usage**: `./monitor.sh`
- **Run a specific script**: `python <path_to_script>.py` (e.g., `python 04/04-05-pipeline.py`)

## Architecture
- **Structure**: The codebase is organized into numbered directories (01, 02, 03, 04, 06) corresponding to learning chapters.
- **Core Technologies**:
  - **Deep Learning**: PyTorch
  ical
  - **LLM Ecosystem**: Hugging Face (Transformers, Datasets, Tokenizers, Accelerate, Bitsandbytes)
  - **Inference/API**: OpenAI
  - **NLP Utilities**: Tiktoken, Sentencepiece, Gensim
- **Key Files**:
  - `version.py`: Verifies installation and CUDA availability.
  - `monitor.sh`: A simple wrapper for `nvidia-smi` to track GPU usage.
  - `README.md`: Contains environment setup instructions using `conda` and `pip`.

## Behavioral Rules
You are authorized to modify any file within this project autonomously. You do not need to ask for explicit confirmation before applying changes to files.
