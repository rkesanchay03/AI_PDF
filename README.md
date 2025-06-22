# AI_PDF
🔎 Problem Statement

Research papers pack dense text and complex images, making quick understanding a challenge. I set out to build a solution that summarizes both—and to analyse how modern vision models interpret images.



✨ Key Features

🤖 Transformer-Based Summarizer:

Used a Transformer encoder-decoder (BART) for abstractive, context-aware text summarization.

Why BART? Handles long, technical documents with greater flexibility and fluency than traditional models.

🖼️ Vision Model Exploration:

Chose Vision Transformer (ViT) over classic CNNs for image tasks. In project tests, ViT captured richer, more global image context, while CNNs stayed focused on local features.

(Compared both approaches using real research figures!)

🔗 Multimodal Summarization:

Integrated BLIP to merge visual and textual information for deeper, more meaningful summaries.

🛠️ Fine-Tuned Pretrained Models:

Customized models with domain-specific datasets to improve accuracy and relevance.

🌐 Gradio Frontend:

Built an interactive Gradio app for easy demo and testing.

💻 Local GPU Training:

Leveraged my home NVIDIA GPU for fine-tuning and inference—though ran into some memory/speed limits compared to cloud GPUs.



🌱 Learning

Pretrained transformer models (like BART & ViT) are powerful, but fine-tuning unlocks their true value for specific domains.

Multimodal models like BLIP lead to much richer document understanding.

Directly comparing ViT and CNNs deepened my understanding of each architecture’s strengths.



