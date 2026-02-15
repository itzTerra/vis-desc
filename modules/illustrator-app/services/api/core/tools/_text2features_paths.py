from core.tools._text2features import FeatureExtractorPipelineResources
from core.utils import DATA_DIR


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
    concreteness_singleword_path=DATA_DIR / "datasets" / "words.csv",
    concreteness_multiword_path=DATA_DIR / "datasets" / "multiword.csv",
    prepositions_path=DATA_DIR / "datasets" / "prepositions.csv",
    places_path=DATA_DIR / "datasets" / "places.txt",
)

FEATURE_SERVICE_MODERNBERT_ONNX_PATH = (
    DATA_DIR / "models" / "modernbert_embed" / "model.onnx"
)
