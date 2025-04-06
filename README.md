# ContextFlow

**A novel implementation of Retrieval-Augmented Generation (RAG) with joint optimization for decoder-only LLMs.**

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Mathematical Foundation](#mathematical-foundation)
- [Project Status](#project-status)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

ContextFlow is a Retrieval-Augmented Generation (RAG) framework that enables seamless integration of modern decoder-only language models with joint optimization capabilities. Unlike traditional RAG systems, ContextFlow offers a unified probabilistic approach that aligns retrieval and generation components aiming to improve performance and reduce hallucinations.

## Key Features

- **Decoder-Only LLM Support**: First implementation to support modern decoder-only architectures (like LLaMA, Mistral, etc.)
- **Joint Optimization**: Novel probability marginalization approach for training both retriever and generator end-to-end
- **Flexible Architecture**: Support for custom prompts, system instructions, and retrieval strategies
- **Memory Efficient**: Optimized for large language models with quantization support
- **Efficient Training**: Parameter-efficient fine-tuning (PEFT) support
- **HuggingFace Hub**: Extensive support for HuggingFace Hub to load/save large models
## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+

### Install via pip

```bash
pip install contextflow
```

### From source

```bash
git clone https://github.com/username/contextflow.git
cd contextflow
pip install -e .
```

## 🚀 Quick Start

```python
from contextflow import RAGSequence
from transformers import AutoTokenizer, AutoModelForCausalLM, DPRQuestionEncoder
from accelerate import Accelerator

# Initialize components
accelerator = Accelerator(mixed_precision="fp16")
question_encoder = DPRQuestionEncoder.from_pretrained(
    "facebook/dpr-question_encoder-single-nq-base"
).to(accelerator.device)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B-Instruct")
generator = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-70B-Instruct",
    device_map="auto"
)

# Create RAG model
rag_model = RAGSequence(
    question_encoder=question_encoder,
    retriever=retriever,  # Initialize your retriever
    generator_tokenizer=tokenizer,
    generator=generator,
    accelerator=accelerator
)

# Generate with context
question = "What is the capital of France?"
input_ids = rag_model.retriever.question_encoder_tokenizer(
    question, return_tensors="pt"
)["input_ids"].to(accelerator.device)

outputs = rag_model.generate(
    input_ids=input_ids,
    max_new_tokens=128,
    num_docs=5
)

print(outputs[0]["generated_text"])
```

## Mathematical Foundation

My work introduces a novel probability marginalization approach that integrates the retriever and generator more effectively within retrieval-augmented generation (RAG) systems. This is formalized as:

### Marginalization Approximation
We approximate the actual probability:

$$p_\theta(y^{(m)} \mid z_k^{(m)}, x^{(m)}) \approx p_\theta(y^{(m)} \mid f(z_k^{(m)}, x^{(m)}))$$

where:

- **$p_\theta$**: Represents the generator's probability distribution, parameterized by $\theta$.
- **$y^{(m)}$**: The generated output sequence for the $m^{\text{th}}$ example in the batch.
- **$z_k^{(m)}$**: The $k^{\text{th}}$ retrieved document for the $m^{\text{th}}$ example.
- **$x^{(m)}$**: The input query for the $m^{\text{th}}$ example.
- **$f(z_k^{(m)}, x^{(m)})$**: The prompt function that combines the retrieved context $z_k^{(m)}$ and the input query $x^{(m)}$ into a single sequence.

### Loss Function

To train the model, we minimize the following loss:

$$
\mathcal{L} = - \frac{1}{B} \sum_{m=1}^B \left[ \sum_{k=1}^K p_\eta(z_k^{(m)} \mid x^{(m)}) \cdot p_\theta(y^{(m)} \mid f(x^{(m)}, z_k^{(m)})) \right]
$$

## Project Status

ContextFlow is still being developed. The core functionality is stable, but I'm actively experimenting and adding new features.

### Roadmap

- [ ] Improved documentation and tutorials
- [ ] Extensive comparison on different datasets
- [ ] Ablations on retriever and generator
- [ ] Performance benchmarks and comprehensive evaluations
- [ ] Scalability, inference/training cost analysis
- [ ] Important insights
- [ ] Support for RAG-token

## Contributing

I welcome contributions of all kinds! If you have any suggestions, please let me know!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The original RAG paper by Lewis et al.
- The HuggingFace team for their transformers library