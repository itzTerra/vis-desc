from .nli_base import ORTNLIZeroshotClassifier


class NLIRoberta(ORTNLIZeroshotClassifier):
    """https://huggingface.co/richardr1126/roberta-base-zeroshot-v2.0-c-ONNX"""

    def get_model_name(self):
        return "richardr1126/roberta-base-zeroshot-v2.0-c-ONNX"

    def get_model_file_name(self):
        return "model_quantized.onnx"


class NLIDebertaLarge(ORTNLIZeroshotClassifier):
    """https://huggingface.co/richardr1126/deberta-v3-large-zeroshot-v2.0-ONNX"""

    def get_model_name(self):
        return "richardr1126/deberta-v3-large-zeroshot-v2.0-ONNX"

    def get_model_file_name(self):
        return "model_quantized.onnx"


class NLIModernBERTLarge(ORTNLIZeroshotClassifier):
    """https://huggingface.co/onnx-community/ModernBERT-large-zeroshot-v2.0-ONNX"""

    def get_model_name(self):
        return "onnx-community/ModernBERT-large-zeroshot-v2.0-ONNX"

    def get_model_file_name(self):
        return "model_int8.onnx"
