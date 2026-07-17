# Models

Define neural architectures, model-agnostic wrappers, and projection interfaces here.

## Swapping Models
- Design model-agnostic wrappers using Hugging Face `AutoModel` to support easily switching between:
  - Qwen
  - SmolLM
  - TinyLlama
  - Gemma
  - Llama / Mistral / Phi
  - Custom architectures
- Keep the models modular and decoupled from the training loops.
