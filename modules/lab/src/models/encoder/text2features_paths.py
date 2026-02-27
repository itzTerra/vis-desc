from text2features import FeatureExtractorPipelineResources
from utils import DATA_DIR, IMAG_DATA_DIR


FEATURE_PIPELINE_RESOURCES = FeatureExtractorPipelineResources(
    char_ngrams_path=DATA_DIR / "features" / "char_ngrams_features.csv",
    pos_ngrams_path=DATA_DIR / "features" / "pos_ngrams_features.csv",
    dep_tree_node_ngrams_path=DATA_DIR
    / "features"
    / "dep_tree_node_ngrams_features.csv",
    dep_tree_relation_ngrams_path=DATA_DIR
    / "features"
    / "dep_tree_relation_ngrams_features.csv",
    dep_tree_complete_ngrams_path=DATA_DIR
    / "features"
    / "dep_tree_complete_ngrams_features.csv",
    concreteness_singleword_path=IMAG_DATA_DIR / "words.csv",
    concreteness_multiword_path=IMAG_DATA_DIR / "multiword.csv",
    prepositions_path=IMAG_DATA_DIR / "prepositions.csv",
    places_path=IMAG_DATA_DIR / "places.txt",
)

FEATURE_SERVICE_MODERNBERT_ONNX_PATH = (
    DATA_DIR / "models" / "modernbert_embed" / "model.onnx"
)
