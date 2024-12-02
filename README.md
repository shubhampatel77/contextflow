# ContextFlow

A novel implementation of Retrieval-Augmented Generation (RAG) that enables seamless integration of decoder-only language models with joint optimization capabilities.

## Key Features

- **Decoder-Only LLM Support**: First-of-its-kind implementation enabling RAG with modern decoder-only architectures
- **Joint Optimization**: Novel probability marginalization approach for training both retriever and generator
- **Flexible Architecture**: Support for custom prompts, system instructions, and retrieval strategies
- **Memory Efficient**: Optimized for large language models with quantization support

## Mathematical Foundation

Our work introduces a novel probability marginalization approach that integrates the retriever and generator more effectively within retrieval-augmented generation (RAG) systems. This is formalized as:

### Marginalization Approximation
We approximate the actual probability:

$$p_θ(y^{(m)} | z_k^{(m)}, x^{(m)}) \approx p_θ(y^{(m)} | f(z_k^{(m)}, x^{(m)}))$$

where:

- **$p_θ$**: Represents the generator's probability distribution, parameterized by $\theta$.
- **$y^{(m)}$**: The generated output sequence for the $m^{th}$ example in the batch.
- **$z_k^{(m)}$**: The $k^{th}$ retrieved document for the $m^{th}$ example.
- **$x^{(m)}$**: The input query for the $m^{th}$ example.
- **$f(z_k^{(m)}, x^{(m)})$**: The prompt function that combines the retrieved context $z_k^{(m)}$ and the input query $x^{(m)}$ into a single sequence. This includes user instructions, the system prompt, and the retrieved document.

By consolidating $z_k^{(m)}$ and $x^{(m)}$ into $f(z_k^{(m)}, x^{(m)})$, the generator operates directly on the combined prompt, bypassing the need for separate representations of context and query.

### Loss Function

To train the model, we minimize the following loss:

$$
\mathcal{L} = - \frac{1}{B} \sum_{m=1}^B \left[ \sum_{k=1}^K p_η(z_k^{(m)} | x^{(m)}) \cdot p_θ(y^{(m)} | f(x^{(m)}, z_k^{(m)})) \right]
$$

where:

- **$\mathcal{L}$**: The average negative log-likelihood loss over the batch of size $B$.
- **$p_η(z_k^{(m)} | x^{(m)})$**: The retriever's probability of selecting the $k^{th}$ document $z_k^{(m)}$ given the query $x^{(m)}$, parameterized by $\eta$.
- **$p_θ(y^{(m)} | f(x^{(m)}, z_k^{(m)}))$**: The generator's probability of generating $y^{(m)}$ based on the combined prompt $f(x^{(m)}, z_k^{(m)})$.
- **$K$**: The total number of retrieved documents for each query.

### Explanation of the Terms:

1. **Retriever Contribution**:
   - $p_η(z_k^{(m)} | x^{(m)})$: Captures the retriever's confidence in each retrieved document.
   - The inner summation $\sum_{k=1}^K$ marginalizes over all retrieved documents, weighting their influence on the loss by their retrieval probabilities.

2. **Generator Contribution**:
   - $p_θ(y^{(m)} | f(x^{(m)}, z_k^{(m)}))$: Evaluates the likelihood of the generated sequence given the prompt formed by the query and retrieved document.

3. **Batch-Averaged Loss**:
   - The outer summation $\frac{1}{B} \sum_{m=1}^B$ averages the loss across all examples in the batch.

This loss formulation ensures that both the retriever and generator are trained end-to-end, optimizing their interaction to achieve better alignment between retrieval and generation.


## Installation

```bash
pip install contextflow
```

## Quick Start

```python
from contextflow import RAGSequence
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator

# Initialize components
accelerator = Accelerator(mixed_precision="fp16")
question_encoder = DPRQuestionEncoder.from_pretrained(
    "facebook/dpr-question_encoder-single-nq-base"
    ).to(accelerator.device)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B-Instruct")
generator = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-70B-Instruct")

# Create RAG model
rag_model = RAGSequence(
    question_encoder=question_encoder,
    retriever=retriever,
    generator_tokenizer=tokenizer,
    generator=generator
)

# Generate with context
outputs = rag_model.generate(
    input_ids=input_ids,
    max_new_tokens=128,
    num_docs=5
)
```

## Training Example

```python
from contextflow.training import train_rag

# Train the model
train_rag(
    model=rag_model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    num_epochs=3,
    learning_rate=5e-5
)
```


## Future Development

- [ ] Multi-GPU training support
- [ ] Additional retrieval strategies
- [ ] Additional fine-tuning strategies
- [ ] Improved documentation and tutorials
- [ ] Performance benchmarks
- [ ] Support for RAG-token

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Any contribution is welcome! Please feel free to submit a Pull Request.