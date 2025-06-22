import fitz  
import io
from PIL import Image
from datasets import load_dataset
import gradio as gr
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    Blip2Processor,
    Blip2ForConditionalGeneration,
)

#GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Text Summarizer (BART) ---
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn",force_download=True)
text_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn").to(device)
summarizer = pipeline("summarization", model=text_model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# --- Vision Transformer (ViT-GPT2 Captioning) ---
vit_caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
vit_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
vit_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# --- BLIP-2 Multi-modal Summarization ---
blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
if torch.cuda.is_available():
    blip_model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", device_map="auto", torch_dtype=torch.float16
    )
else:
    blip_model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", device_map="cpu", torch_dtype=torch.float32
    )


def extract_text_and_images(pdf_path):
    texts, images = [], []
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            # Get text
            page_text = page.get_text("text")
            if page_text and page_text.strip():
                texts.append(page_text)
            else:
                texts.append("[No text found on this page]")
            # Get images
            for img in page.get_images(full=True):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                try:
                    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    images.append(image)
                except Exception:
                    continue  # Skip unreadable images
        if not texts:
            texts.append("[No text found in PDF]")
        if not images:
            images.append(Image.new("RGB", (224, 224), color="white"))
    except Exception as e:
        texts = [f"[PDF extraction error: {e}]"]
        images = []
    return texts, images

def summarize_text_chunks(chunks):
    results = []
    for chunk in chunks:
        if not chunk.strip() or len(chunk.strip()) < 50:
            results.append("[No summary: text too short]")
            continue
        try:
            output = summarizer(chunk[:1024], max_length=100, min_length=30, truncation=True)[0]["summary_text"]
            results.append(output)
        except Exception as e:
            results.append(f"[Summarization error: {e}]")
    return results

def caption_images(images):
    captions = []
    for img in images:
        try:
            if img.mode != "RGB":
                img = img.convert("RGB")
            pixel_values = vit_processor(images=img, return_tensors="pt").pixel_values.to(device)
            output_ids = vit_caption_model.generate(pixel_values, max_length=16, num_beams=4)
            caption = vit_tokenizer.decode(output_ids[0], skip_special_tokens=True)
            captions.append(caption)
        except Exception as e:
            captions.append(f"[Image captioning error: {e}]")
    return captions

def multimodal_summarize_page(image, text):
    prompt = f"Summarize this image in the context of the text: {text[:300]}"
    try:
        inputs = blip_processor(images=image, text=prompt, return_tensors="pt").to(blip_model.device)
        output = blip_model.generate(**inputs, max_new_tokens=100)
        summary = blip_processor.decode(output[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        return f"[Multi-modal error: {e}]"

def build_report(text_summaries=None, image_captions=None, multimodal_summaries=None):
    report = ""
    if multimodal_summaries:
        report += "# Multi-Modal Summary (Text + Image)\n"
        for i, s in enumerate(multimodal_summaries):
            report += f"- Page {i+1}: {s}\n"
    if text_summaries:
        report += "\n# Text-Only Summary\n"
        for i, s in enumerate(text_summaries):
            report += f"- Page {i+1}: {s}\n"
    if image_captions:
        report += "\n# Image Captions\n"
        for j, c in enumerate(image_captions):
            report += f"- Image {j+1}: {c}\n"
    return report

def interface(pdf_file):
    pdf_path = pdf_file.name
   
    texts, images = extract_text_and_images(pdf_path)
   
    text_summaries = summarize_text_chunks(texts)
  
    image_captions = caption_images(images)
   
    multimodal_summaries = []
    for img, txt in zip(images, texts):
        summary = multimodal_summarize_page(img, txt)
        multimodal_summaries.append(summary)
    # Build report
    return build_report(
        text_summaries=text_summaries,
        image_captions=image_captions,
        multimodal_summaries=multimodal_summaries
    )

if __name__ == "__main__":
    gr.Interface(
        fn=interface,
        inputs=gr.File(label="Upload a PDF"),
        outputs=gr.Textbox(label="Summarized Output", lines=25),
        title="AI PDF Summarizer (Text & Image)",
        description="Summarizes AI research papers with BART(Text), ViT(Image), and BLIP-2 (Text+Image)."
    ).launch()
