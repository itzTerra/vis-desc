#!/usr/bin/env python3
from sentence_transformers import SentenceTransformer
from light_embed import TextEmbedding
import wn
from django.conf import settings
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification

wn.download("oewn:2024")

embed_minilm = TextEmbedding(
    "sentence-transformers/all-MiniLM-L6-v2", cache_folder=settings.MODEL_CACHE_DIR
)
embed_modernbert = SentenceTransformer(
    "lightonai/modernbert-embed-large",
    truncate_dim=256,
    backend="onnx",
    cache_folder=settings.MODEL_CACHE_DIR,
    model_kwargs={"file_name": "model_quantized.onnx"},
)

# nli_roberta
model = ORTModelForSequenceClassification.from_pretrained(
    "richardr1126/roberta-base-zeroshot-v2.0-c-ONNX",
    file_name="model_quantized.onnx",
    cache_dir=settings.MODEL_CACHE_DIR,
)
tokenizer = AutoTokenizer.from_pretrained(
    "richardr1126/roberta-base-zeroshot-v2.0-c-ONNX",
    cache_dir=settings.MODEL_CACHE_DIR,
)

print("Models downloaded and cached.")
