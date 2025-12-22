#!/usr/bin/env python3

import os
import sys
import filecmp

# List of file pairs to check for identity
PAIRS = [
    (
        "modules/lab/src/book_preprocessing.py",
        "modules/illustrator-app/services/api/core/tools/book_preprocessing.py",
    ),
    (
        "modules/lab/src/book_segmenting.py",
        "modules/illustrator-app/services/api/core/tools/book_segmenting.py",
    ),
    (
        "modules/lab/src/text2features.py",
        "modules/illustrator-app/services/api/core/tools/text2features.py",
    ),
    (
        "modules/lab/data/features/char_ngrams_features.csv",
        "modules/illustrator-app/services/api/data/features/char_ngrams_features.csv",
    ),
    (
        "modules/lab/data/features/dep_tree_complete_ngrams_features.csv",
        "modules/illustrator-app/services/api/data/features/dep_tree_complete_ngrams_features.csv",
    ),
    (
        "modules/lab/data/features/dep_tree_node_ngrams_features.csv",
        "modules/illustrator-app/services/api/data/features/dep_tree_node_ngrams_features.csv",
    ),
    (
        "modules/lab/data/features/dep_tree_relation_ngrams_features.csv",
        "modules/illustrator-app/services/api/data/features/dep_tree_relation_ngrams_features.csv",
    ),
    (
        "modules/lab/data/features/pos_ngrams_features.csv",
        "modules/illustrator-app/services/api/data/features/pos_ngrams_features.csv",
    ),
    (
        "modules/lab/data/datasets/concreteness/multiword.csv",
        "modules/illustrator-app/services/api/data/datasets/multiword.csv",
    ),
    (
        "modules/lab/data/datasets/concreteness/places.txt",
        "modules/illustrator-app/services/api/data/datasets/places.txt",
    ),
    (
        "modules/lab/data/datasets/concreteness/prepositions.csv",
        "modules/illustrator-app/services/api/data/datasets/prepositions.csv",
    ),
    (
        "modules/lab/data/datasets/concreteness/words.csv",
        "modules/illustrator-app/services/api/data/datasets/words.csv",
    ),
    (
        "modules/lab/src/models/nli/nli_base.py",
        "modules/illustrator-app/services/api/core/tools/evaluators/nli_base.py",
    ),
    (
        "modules/lab/src/models/nli/nli_roberta.py",
        "modules/illustrator-app/services/api/core/tools/evaluators/nli_roberta.py",
    ),
]

for file1, file2 in PAIRS:
    if not os.path.exists(file1):
        print(f"Warning: File {file1} does not exist, skipping pair")
        continue
    if not os.path.exists(file2):
        print(f"Warning: File {file2} does not exist, skipping pair")
        continue

    if not filecmp.cmp(file1, file2, shallow=False):
        print(f"Error: Files {file1} and {file2} are not identical")
        sys.exit(1)

print("All specified files are in sync")
sys.exit(0)
