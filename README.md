# Clevr

Clevr  is an app that learns acoustic concepts from natural language supervision and enables “Zero-Shot” inference. The model has been extensively evaluated in 26 audio downstream tasks achieving SoTA in several of them including classification, retrieval, and captioning.

## Setup

First, install python 3.8 or higher (3.11 recommended). Then, install Clevr using either of the following:

## Clevr weights

Clevr is the audio captioning model that uses the 2023 encoders.

## Usage

- Zero-Shot Classification and Retrieval
```python
from msclap import Clevr

# The model weight will be downloaded automatically if `model_fp` is not specified
clap = Clevr(version = '2023', use_cuda=False)

# Extract text embeddings
text_embeddings = clap_model.get_text_embeddings(class_labels: List[str])

# Extract audio embeddings
audio_embeddings = clap.get_audio_embeddings(file_paths: List[str])

# Compute similarity between audio and text embeddings 
similarities = clap.compute_similarity(audio_embeddings, text_embeddings)
```
