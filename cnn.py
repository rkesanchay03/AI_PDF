import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import gradio as gr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- BLIP-1 (ResNet CNN Encoder) ----
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

def blip_caption(image):
    if image.mode != "RGB": image = image.convert("RGB")
    inputs = blip_processor(image, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs, max_new_tokens=30)
    return blip_processor.decode(out[0], skip_special_tokens=True)

# ---- ViT-GPT2 (Vision Transformer Encoder) ----
vit_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
vit_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
vit_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

def vit_caption(image):
    if image.mode != "RGB": image = image.convert("RGB")
    pixel_values = vit_processor(images=image, return_tensors="pt").pixel_values.to(device)
    
    output_ids = vit_model.generate(pixel_values, max_length=16, num_beams=1)
    return vit_tokenizer.decode(output_ids[0], skip_special_tokens=True)



with gr.Blocks(title="CNN vs ViT Encoder Image Caption Comparison") as demo:
    gr.Markdown("# CNN Encoder (ResNet) vs Vision Transformer (ViT) Encoder\nUpload an image to see captions from both approaches.")
    with gr.Row():
        with gr.Column():
            gr.Markdown("### ResNet (BLIP-1 CNN Encoder)")
            cnn_image = gr.Image(type="pil", label="Input Image")
            cnn_output = gr.Textbox(label="Caption (BLIP-1/ResNet)")
            cnn_image.upload(blip_caption, cnn_image, cnn_output)
        with gr.Column():
            gr.Markdown("### Vision Transformer (ViT-GPT2 Encoder)")
            vit_image = gr.Image(type="pil", label="Input Image")
            vit_output = gr.Textbox(label="Caption (ViT-GPT2)")
            vit_image.upload(vit_caption, vit_image, vit_output)
    with gr.Row():
        gr.Examples(
            examples=[
                ["https://images.unsplash.com/photo-1519125323398-675f0ddb6308"], # Example URL
            ],
            inputs=[cnn_image, vit_image],
        )
    gr.Markdown(
        """
        ## Analysis
        - BLIP-1 uses a CNN (ResNet) vision backbone.
        - ViT-GPT2 uses a Vision Transformer model as the encoder for images.
        - You can see the difference in captioning style, detail, and accuracy.
        """
    )

demo.launch()
