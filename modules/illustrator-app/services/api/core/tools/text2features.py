from pathlib import Path
import numpy as np

# import onnxruntime as rt
# from transformers import AutoTokenizer
from typing import Literal
from light_embed import TextEmbedding
from dataclasses import dataclass
from typing import Any
from booknlp.booknlp import BookNLP, EnglishBookNLPConfig, Token
import pandas as pd
import ast
from numpy.typing import NDArray
import syllables
import wn
import regex
# from sentence_transformers import SentenceTransformer


@dataclass
class TokenData:
    text: str
    itext: str  # Lowercase text
    pos: str  # Universal Dependencies POS tag
    fine_pos: str  # English fine-grained POS tag
    lemma: str
    sentence_id: int
    within_sentence_id: int
    event: bool
    is_stop: bool
    like_num: bool
    morph_tense: str | None

    @staticmethod
    def from_token(token: Token) -> "TokenData":
        return TokenData(
            text=token.text,
            itext=token.itext,
            pos=token.pos,
            fine_pos=token.fine_pos,
            lemma=token.lemma,
            sentence_id=token.sentence_id,
            within_sentence_id=token.within_sentence_id,
            event=token.event,
            is_stop=token.is_stop,
            like_num=token.like_num,
            morph_tense=token.morph.get("Tense"),
        )


@dataclass
class EntityData:
    start_token: int
    end_token: int
    cat: str  # Entity category (PER, FAC, GPE, LOC, VEH, ORG)
    text: str
    coref: int
    prop: str

    @staticmethod
    def from_entity_dict(entity_dict: dict[str, Any]) -> "EntityData":
        return EntityData(
            start_token=entity_dict["start_token"],
            end_token=entity_dict["end_token"],
            cat=entity_dict["cat"],
            text=entity_dict["text"],
            coref=entity_dict["coref"],
            prop=entity_dict["prop"],
        )


@dataclass
class NounChunkData:
    text: str
    start: int
    end: int
    length: int

    @staticmethod
    def from_noun_chunk(noun_chunk: Any) -> "NounChunkData":
        return NounChunkData(
            text=noun_chunk.text,
            start=noun_chunk.start,
            end=noun_chunk.end,
            length=len(noun_chunk),
        )


@dataclass
class SentenceToken:
    text: str
    pos: str
    dep: str
    children: list["SentenceToken"]


@dataclass
class SentenceData:
    root: SentenceToken

    @staticmethod
    def dfs_build_tree(spacy_token: Token, sentence_token: SentenceToken):
        for child in spacy_token.children:
            child_token = SentenceToken(child.text, child.pos_, child.dep_, [])
            sentence_token.children.append(child_token)
            SentenceData.dfs_build_tree(child, child_token)

    @staticmethod
    def from_sentence(sent: Any) -> "SentenceData":
        root = SentenceToken(sent.root.text, sent.root.pos_, sent.root.dep_, [])
        sent_data = SentenceData(root=root)
        SentenceData.dfs_build_tree(sent.root, root)
        return sent_data

    def get_tokens(self) -> list[SentenceToken]:
        tokens = []

        def dfs(token: SentenceToken):
            tokens.append(token)
            for child in token.children:
                dfs(child)

        dfs(self.root)
        return tokens


@dataclass
class ExtCtx:
    """Extraction context with serializable data for feature extraction."""

    text: str  # Original text for debugging
    tokens: list[TokenData]
    words: list[TokenData]  # Non-punctuation/symbol tokens
    noun_chunks: list[NounChunkData]
    entities: list[EntityData]
    supersense: list[tuple[int, int, str, str]]  # (start_token, end_token, label, text)
    sents: list[SentenceData]

    @staticmethod
    def from_booknlp_ctx(booknlp_ctx: Any, text: str) -> "ExtCtx":
        tokens_data = [TokenData.from_token(token) for token in booknlp_ctx.tokens]
        return ExtCtx(
            text=text,
            tokens=tokens_data,
            words=[
                token_data
                for token_data in tokens_data
                if token_data.pos not in ["PUNCT", "SYM"]
            ],
            noun_chunks=[
                NounChunkData.from_noun_chunk(chunk)
                for chunk in booknlp_ctx.noun_chunks
            ],
            entities=[
                EntityData.from_entity_dict(entity) for entity in booknlp_ctx.entities
            ],
            supersense=list(booknlp_ctx.supersense),
            sents=[SentenceData.from_sentence(sent) for sent in booknlp_ctx.sents],
        )


@dataclass
class FeatureExtractorPipelineResources:
    """Holds paths for loaded resources"""

    char_ngrams_path: Path
    pos_ngrams_path: Path
    dep_tree_node_ngrams_path: Path
    dep_tree_relation_ngrams_path: Path
    dep_tree_complete_ngrams_path: Path
    concreteness_singleword_path: Path
    concreteness_multiword_path: Path
    prepositions_path: Path
    places_path: Path


class FeatureExtractorPipeline:
    FEATURE_COUNT = 3671
    PREPROCESS_TRANSLATION_TABLE = str.maketrans(
        {
            "\r": None,
            "’": "'",
            "“": '"',
            "”": '"',
            "？": "?",
            "！": "!",
        }
    )
    CONTENT_UD_POS_TAGS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}
    PUNCTUATION_SYMBOL_MAP = {
        ".": 0,
        ",": 1,
        ":": 2,
        ";": 3,
        "!": 4,
        "?": 5,
        '"': 6,
        "'": 7,
        "(": 8,
        ")": 9,
        "-": 10,
        "–": 11,
        "—": 12,
        "…": 13,
        "/": 14,
        "\\": 15,
    }
    ENG_POS_TAG_MAP = {
        tag: i
        for i, tag in enumerate(
            [
                "$",
                "''",
                ",",
                "-LRB-",
                "-RRB-",
                ".",
                ":",
                "ADD",
                "AFX",
                "CC",
                "CD",
                "DT",
                "EX",
                "FW",
                "HYPH",
                "IN",
                "JJ",
                "JJR",
                "JJS",
                "LS",
                "MD",
                "NFP",
                "NN",
                "NNP",
                "NNPS",
                "NNS",
                "PDT",
                "POS",
                "PRP",
                "PRP$",
                "RB",
                "RBR",
                "RBS",
                "RP",
                "SYM",
                "TO",
                "UH",
                "VB",
                "VBD",
                "VBG",
                "VBN",
                "VBP",
                "VBZ",
                "WDT",
                "WP",
                "WP$",
                "WRB",
                "XX",
                "_SP",
                "``",
            ]
        )
    }
    # http://people.ischool.berkeley.edu/~dbamman/pubs/pdf/naacl2019_literary_entities.pdf
    ENTITY_CATEGORY_MAP = {
        "PER": 0,
        "FAC": 1,
        "GPE": 2,
        "LOC": 3,
        "VEH": 4,
        "ORG": 5,
    }
    # https://aclanthology.org/W06-1670.pdf
    SUPERSENSE_LABELS_NOUNS = [
        "act",
        "animal",
        "artifact",
        "attribute",
        "body",
        "cognition",
        "communication",
        "event",
        "feeling",
        "food",
        "group",
        "location",
        "motive",
        "object",
        "quantity",
        "phenomenon",
        "plant",
        "possession",
        "process",
        "person",
        "relation",
        "shape",
        "state",
        "substance",
        "time",
        "Tops",
    ]
    SUPERSENSE_LABELS_VERBS = [
        "body",
        "change",
        "cognition",
        "communication",
        "competition",
        "consumption",
        "contact",
        "creation",
        "emotion",
        "motion",
        "perception",
        "possession",
        "social",
        "stative",
        "weather",
    ]
    SUPERSENSE_LABELS = [
        *[f"noun.{label}" for label in SUPERSENSE_LABELS_NOUNS],
        *[f"verb.{label}" for label in SUPERSENSE_LABELS_VERBS],
    ]
    SUPERSENSE_LABEL_MAP = {label: i for i, label in enumerate(SUPERSENSE_LABELS)}
    UD_TO_WN_POS = {
        "NOUN": "n",
        "VERB": "v",
        "ADJ": "a",
        "ADV": "r",
    }

    @staticmethod
    def preprocess(text: str) -> str:
        return text.translate(FeatureExtractorPipeline.PREPROCESS_TRANSLATION_TABLE)

    def __init__(self, resources: FeatureExtractorPipelineResources):
        config = EnglishBookNLPConfig(
            model="small",
            pipeline="entity,event,supersense",
            verbose=False,
        )
        self.booknlp = BookNLP("en", config)

        self._char_features_map = {
            ngram: i
            for i, ngram in enumerate(pd.read_csv(resources.char_ngrams_path)["ngram"])
        }
        self._pos_features_map = {
            ast.literal_eval(ngram): i
            for i, ngram in enumerate(pd.read_csv(resources.pos_ngrams_path)["ngram"])
        }

        node_features = [
            ast.literal_eval(ngram)
            for ngram in pd.read_csv(resources.dep_tree_node_ngrams_path)["ngram"]
        ]
        relation_features = [
            ast.literal_eval(ngram)
            for ngram in pd.read_csv(resources.dep_tree_relation_ngrams_path)["ngram"]
        ]
        complete_features = [
            ast.literal_eval(ngram)
            for ngram in pd.read_csv(resources.dep_tree_complete_ngrams_path)["ngram"]
        ]

        self._dep_tree_ngram_map = {}
        offset = 0
        for ngram in node_features:
            self._dep_tree_ngram_map[ngram] = offset
            offset += 1
        self._node_features_end = offset

        for ngram in relation_features:
            self._dep_tree_ngram_map[ngram] = offset
            offset += 1
        self._relation_features_end = offset

        for ngram in complete_features:
            self._dep_tree_ngram_map[ngram] = offset
            offset += 1
        self._dep_tree_features_total = offset

        wn.config.allow_multithreading = True
        self._wordnet = wn.Wordnet("oewn:2024")

        concreteness_data_single = pd.read_csv(resources.concreteness_singleword_path)
        concreteness_data_multi = pd.read_csv(resources.concreteness_multiword_path)

        self._multiword_trie = MultiWordExpressionTrie()
        for _, row in concreteness_data_single.iterrows():
            self._multiword_trie.add_expression(row["word"], row["conc"], 1)
        for _, row in concreteness_data_multi.iterrows():
            self._multiword_trie.add_expression(
                row["expression"], row["conc"], row["n_words"]
            )

        prep_imag_data = pd.read_csv(resources.prepositions_path, na_values=["NA"])
        self._compiled_regex_patterns = {}
        self._exact_match_patterns = {}
        self._multi_word_patterns = {}
        for _, row in prep_imag_data.iterrows():
            prep = row["prep"]
            imag = row["imag"]

            if row["is_regex"] == 1:
                # Compile regex patterns
                self._compiled_regex_patterns[prep] = (
                    regex.compile(prep, regex.IGNORECASE),
                    imag,
                )
            elif row["n_words"] > 1:
                # Multi-word exact matches
                self._multi_word_patterns[prep.casefold()] = imag
            else:
                # Single word exact matches
                self._exact_match_patterns[prep.casefold()] = (
                    imag,
                    row["pos_adp"],
                    row["pos_nonadp"],
                )

        places_data = (resources.places_path).read_text().splitlines()
        self._places_trie = MultiWordExpressionTrie()
        for place in places_data:
            self._places_trie.add_expression(place, 0, len(place.split()))

    def get_ctx(self, text: str) -> ExtCtx:
        booknlp_ctx = self.booknlp.process(text=text)
        return ExtCtx.from_booknlp_ctx(booknlp_ctx, text)

    def extract(
        self, text: str, preprocess=True, ctx: ExtCtx | None = None
    ) -> NDArray[np.float32]:
        if preprocess:
            text = self.preprocess(text)
        if ctx is None:
            ctx = self.get_ctx(text)

        features = []
        features.append(self.extract_quote_ratio(text))
        features.extend(self.extract_char_ngrams(text))
        features.extend(self.extract_word_length_by_char(ctx))
        features.extend(self.extract_ngram_word_length_by_char(ctx))
        features.extend(self.extract_sentence_length_by_word(ctx))
        features.append(self.extract_numeric_word_ratio(ctx))
        features.append(self.extract_ttr(ctx))
        features.append(self.extract_lexical_density(ctx))
        features.extend(self.extract_syllable_ratios(ctx))
        features.append(self.extract_stopwords(ctx))
        features.extend(self.extract_articles(ctx))
        features.extend(self.extract_punctuation(ctx))
        features.append(self.extract_contractions(ctx))
        features.extend(self.extract_casing(ctx))
        features.extend(self.extract_casing_bigrams(ctx))
        features.extend(self.extract_pos_frequency(ctx))
        features.extend(self.extract_pos_ngrams(ctx))
        features.extend(self.extract_dependency_tree_structure(ctx))
        features.extend(self.extract_dependency_tree_relations(ctx))
        features.extend(self.extract_noun_phrase_lengths(ctx))
        features.extend(self.extract_entity_categories(ctx))
        features.append(self.extract_events(ctx))
        features.extend(self.extract_supersense(ctx))
        features.extend(self.extract_tense(ctx))
        features.extend(self.extract_polysemy(ctx))
        features.extend(self.extract_word_concreteness(ctx))
        features.extend(self.extract_preposition_imageability(ctx))
        features.append(self.extract_places(ctx))
        # assert len(features) == FEATURE_COUNT, f"Expected {FEATURE_COUNT} features, got {len(features)}"

        return np.array(features, dtype=np.float32)

    def extract_quote_ratio(self, text: str) -> float:
        """
        Count characters between properly closed quotes only.
        Ignores unclosed/malformed quotes.

        Time complexity: O(n), where n is text length
        """
        if not text:
            return 0.0

        total_chars = len(text)
        quoted_chars = 0
        in_quote = False
        quote_char = None
        current_span_chars = 0

        for char in text:
            if not in_quote and char in ['"', "'"]:
                in_quote = True
                quote_char = char
                current_span_chars = 0
            elif in_quote and char == quote_char:
                # Found closing quote - add this span to counter
                quoted_chars += current_span_chars
                in_quote = False
                quote_char = None
                current_span_chars = 0
            elif in_quote:
                current_span_chars += 1

        return (quoted_chars / total_chars) if total_chars > 0 else 0.0

    def extract_char_ngrams(self, text: str) -> NDArray[np.float32]:
        """Required ctx: 'text'

        Time complexity: O(mn), where m is number of characters and n is n-gram size
        """
        MAX_N = 5
        text = text.casefold()
        occurences = np.zeros(len(self._char_features_map), dtype=np.float32)
        total_ngrams_count = 0

        for n in range(2, MAX_N + 1):
            for i in range(len(text) - n + 1):
                ngram = text[i : i + n]
                if ngram in self._char_features_map:
                    occurences[self._char_features_map[ngram]] += 1
                total_ngrams_count += 1

        if total_ngrams_count == 0:
            return occurences
        return occurences / total_ngrams_count

    def extract_word_length_by_char(self, ctx: ExtCtx) -> NDArray[np.float32]:
        """Required ctx: 'words'

        Time complexity: O(n), where n is number of words

        Returns n-dimensional vector, where each dimension i (1 ≤ i < n) represents the relative frequency of words within the document with the length of i characters. The last dimension n is reserved for words longer than or equal to n.
        """
        DIM = 15
        length_counts = np.zeros(DIM, dtype=np.float32)
        if not ctx.words:
            return length_counts

        for token in ctx.words:
            length = len(token.text)
            if length >= DIM:
                length_counts[DIM - 1] += 1
            else:
                length_counts[length - 1] += 1
        return length_counts / len(ctx.words)

    def extract_ngram_word_length_by_char(self, ctx: ExtCtx) -> NDArray[np.float32]:
        """Required ctx: 'words'

        Time complexity: O(mn), where m is number of words and n is n-gram size

        Returns n-dimensional vector, where each dimension i (1 ≤ i < n) represents the relative frequency of character length sums of word triplets (3-grams) in the range [3i, 3i+2]. The last dimension n is reserved for 3-grams longer than or equal to 3n. Captures length patterns across word sequences.
        """
        DIM = 15
        N = 3
        length_counts = np.zeros(DIM, dtype=np.float32)
        if not ctx.words:
            return length_counts

        total_ngrams = max(len(ctx.words) - N + 1, 1)
        for i in range(total_ngrams):
            ngram = ctx.words[i : i + N]
            length_sum = sum(len(token.text) for token in ngram)
            if length_sum >= N * DIM:
                length_counts[DIM - 1] += 1
            else:
                index = (length_sum - N) // N
                length_counts[index] += 1
        return length_counts / total_ngrams

    def extract_sentence_length_by_word(self, ctx: ExtCtx) -> NDArray[np.float32]:
        """Required ctx: 'words'

        Time complexity: O(n), where n is number of words

        Returns (n+1)-dimensional vector, where each dimension i (1 ≤ i < n) represents the relative frequency of sentences within the document with the length of i words. The last dimension n is reserved for sentences longer than or equal to n. Last (n+1-th) dimension is doc. avg sentence length by word.
        """
        BIN_SIZE = 3
        # https://www.researchgate.net/publication/387225234_Distribution_of_sentence_length_of_English_complex_sentences
        UPPER_SENTENCE_LENGTH = 78
        DIM = UPPER_SENTENCE_LENGTH // BIN_SIZE
        length_counts = np.zeros(DIM, dtype=np.float32)
        cur_sent_idx = 0
        cur_sent_length = 0
        total_length = 0

        for token in ctx.words:
            sent_idx = token.sentence_id
            if sent_idx == cur_sent_idx:
                cur_sent_length += 1
            else:
                total_length += cur_sent_length
                bin_idx = cur_sent_length // BIN_SIZE
                if bin_idx >= DIM:
                    length_counts[DIM - 1] += 1
                else:
                    length_counts[bin_idx] += 1
                cur_sent_idx = sent_idx
                cur_sent_length = 1
        total_length += cur_sent_length
        bin_idx = cur_sent_length // BIN_SIZE
        if bin_idx >= DIM:
            length_counts[DIM - 1] += 1
        else:
            length_counts[bin_idx] += 1

        rel_freq = length_counts / (cur_sent_idx + 1)
        avg_sent_length = min(
            total_length / (cur_sent_idx + 1) / UPPER_SENTENCE_LENGTH, 1.0
        )
        return np.concatenate([rel_freq, np.array([avg_sent_length], dtype=np.float32)])

    def extract_numeric_word_ratio(self, ctx: ExtCtx) -> float:
        """Required ctx: 'words'

        Time complexity: O(n), where n is number of words
        """
        if not ctx.words:
            return 0.0
        numeric_word_count = sum(token.like_num for token in ctx.words)
        return numeric_word_count / len(ctx.words)

    def extract_ttr(self, ctx: ExtCtx) -> float:
        """Required ctx: 'words'

        Time complexity: O(n), where n is number of words

        Returns TTR (unique words to total words), case-insensitive, non-lemmatized, including stop words, not including punctuation
        """
        if not ctx.words:
            return 0.0
        unique_words = set(token.itext for token in ctx.words)
        return len(unique_words) / len(ctx.words)

    def extract_lexical_density(self, ctx: ExtCtx) -> float:
        """Required ctx: 'words'

        Time complexity: O(n), where n is number of words

        Returns lexical density (content words to total words), non-lemmatized, including stop words, not including punctuation
        """
        if not ctx.words:
            return 0.0
        content_word_count = sum(
            token.pos in FeatureExtractorPipeline.CONTENT_UD_POS_TAGS
            for token in ctx.words
        )
        return content_word_count / len(ctx.words)

    def extract_syllable_ratios(self, ctx: ExtCtx) -> NDArray[np.float32]:
        """Required ctx: 'words'

        Time complexity: O(n), where n is number of words
        """
        if not ctx.words:
            return np.array([0.0, 0.0], dtype=np.float32)

        # https://www.researchgate.net/figure/Distribution-of-the-number-of-syllables-per-word-in-Websters-dictionary_fig3_220597266
        UPPER_SYLLABLE_COUNT = 8
        syllable_total = 0
        three_or_more = 0

        for token in ctx.words:
            syllable_count = syllables.estimate(token.text)
            syllable_total += syllable_count
            if syllable_count >= 3:
                three_or_more += 1
        return np.array(
            [
                min(syllable_total / len(ctx.words) / UPPER_SYLLABLE_COUNT, 1.0),
                three_or_more / len(ctx.words),
            ],
            dtype=np.float32,
        )

    def extract_stopwords(self, ctx: ExtCtx) -> float:
        """Required ctx: 'words'

        Time complexity: O(n), where n is number of words
        """
        return (
            sum(token.is_stop for token in ctx.words) / len(ctx.words)
            if ctx.words
            else 0.0
        )

    def extract_articles(self, ctx: ExtCtx) -> NDArray[np.float32]:
        """Required ctx: 'words'\n
        returns: (definite_article_ratio, indefinite_article_ratio)

        Time complexity: O(n), where n is number of words
        """
        if not ctx.words:
            return np.array([0.0, 0.0], dtype=np.float32)

        indef_count = 0
        def_count = 0
        for token in ctx.words:
            if token.itext in ["a", "an"]:
                indef_count += 1
            elif token.itext == "the":
                def_count += 1
        return np.array(
            [def_count / len(ctx.words), indef_count / len(ctx.words)], dtype=np.float32
        )

    def extract_punctuation(self, ctx: ExtCtx) -> NDArray[np.float32]:
        """Required ctx: 'tokens'

        Time complexity: O(n), where n is number of tokens
        """
        occurrences = np.zeros(
            len(FeatureExtractorPipeline.PUNCTUATION_SYMBOL_MAP), dtype=np.float32
        )
        if not ctx.tokens:
            return occurrences

        for token in ctx.tokens:
            if token.text in FeatureExtractorPipeline.PUNCTUATION_SYMBOL_MAP:
                occurrences[
                    FeatureExtractorPipeline.PUNCTUATION_SYMBOL_MAP[token.text]
                ] += 1
        return occurrences / len(ctx.tokens)

    def extract_contractions(self, ctx: ExtCtx) -> float:
        """Required ctx: 'words'

        Time complexity: O(n), where n is number of words
        """
        return (
            sum("'" in token.text for token in ctx.words) / len(ctx.words)
            if ctx.words
            else 0.0
        )

    def extract_casing(self, ctx: ExtCtx) -> NDArray[np.float32]:
        """Required ctx: 'words'

        Time complexity: O(n), where n is number of words

        Returns 3 (first word, inbetween, last word) * 3 (lowercase, uppercase, capitalized) = 9-dimensional vector, where each dimension represents the relative frequency of words in different positions (first, inbetween, last) and different casing (lowercase, uppercase, capitalized).
        """
        words = ctx.words
        counts = np.zeros(
            9, dtype=np.float32
        )  # [first_lower, first_upper, first_title, inbetween_lower, inbetween_upper, inbetween_title, last_lower, last_upper, last_title]
        if not words:
            return counts

        for i, word in enumerate(words):
            # Determine case type (0=lower, 1=upper, 2=title)
            if word.text.islower():
                case_idx = 0
            elif word.text.isupper():
                case_idx = 1
            elif word.text.istitle():
                case_idx = 2
            else:
                continue

            # Determine position (0=first, 1=inbetween, 2=last)
            if word.within_sentence_id == 0:
                position_idx = 0
            elif i == len(words) - 1 or (
                i < len(words) - 1 and words[i + 1].within_sentence_id == 0
            ):
                position_idx = 2
            else:
                position_idx = 1

            counts[position_idx * 3 + case_idx] += 1

        return counts / len(words)

    def extract_casing_bigrams(self, ctx: ExtCtx) -> NDArray[np.float32]:
        """Required ctx: 'words'

        Time complexity: O(n), where n is number of words

        Returns 3 (lowercase, uppercase, capitalized) * 3 (lowercase, uppercase, capitalized) = 9-dimensional vector, where each dimension represents the relative frequency of bigrams in different casing combinations.
        """
        words = ctx.words
        total_bigrams = len(words) - 1
        bigram_counts = np.zeros(
            9, dtype=np.float32
        )  # lower_lower, lower_upper, lower_title, upper_lower, upper_upper, upper_title, title_lower, title_upper, title_title
        if total_bigrams <= 0:
            return bigram_counts

        for i in range(total_bigrams):
            first_text = words[i].text
            second_text = words[i + 1].text

            # Determine case type for first word (0=lower, 1=upper, 2=title, 3=other)
            if first_text.islower():
                first_case = 0
            elif first_text.isupper():
                first_case = 1
            elif first_text.istitle():
                first_case = 2
            else:
                continue

            # Determine case type for second word
            if second_text.islower():
                second_case = 0
            elif second_text.isupper():
                second_case = 1
            elif second_text.istitle():
                second_case = 2
            else:
                continue

            bigram_counts[first_case * 3 + second_case] += 1

        return bigram_counts / total_bigrams

    def extract_pos_frequency(self, ctx: ExtCtx) -> NDArray[np.float32]:
        """Required ctx: 'tokens'

        Time complexity: O(n), where n is number of tokens
        """
        tag_counts = np.zeros(
            len(FeatureExtractorPipeline.ENG_POS_TAG_MAP), dtype=np.float32
        )
        if not ctx.tokens:
            return tag_counts

        for token in ctx.tokens:
            if token.fine_pos in FeatureExtractorPipeline.ENG_POS_TAG_MAP:
                tag_counts[
                    FeatureExtractorPipeline.ENG_POS_TAG_MAP[token.fine_pos]
                ] += 1

        return tag_counts / len(ctx.tokens)

    def extract_pos_ngrams(self, ctx: ExtCtx) -> NDArray[np.float32]:
        """Required ctx: 'tokens'

        Time complexity: O(mn), where m is number of tokens and n is n-gram size

        Uses coarse UD tags:
        ADJ: adjective
        ADP: adposition
        ADV: adverb
        AUX: auxiliary
        CCONJ: coordinating conjunction
        DET: determiner
        INTJ: interjection
        NOUN: noun
        NUM: numeral
        PART: particle
        PRON: pronoun
        PROPN: proper noun
        PUNCT: punctuation
        SCONJ: subordinating conjunction
        SYM: symbol
        VERB: verb
        X: other
        """
        MAX_N = 4
        tokens = ctx.tokens
        occurences = np.zeros(len(self._pos_features_map), dtype=np.float32)
        total_ngrams_count = 0

        for n in range(2, MAX_N + 1):
            for i in range(len(tokens) - n + 1):
                ngram = tuple(token.pos for token in tokens[i : i + n])
                if ngram in self._pos_features_map:
                    occurences[self._pos_features_map[ngram]] += 1
                total_ngrams_count += 1

        if total_ngrams_count == 0:
            return occurences
        return occurences / total_ngrams_count

    def extract_dependency_tree_structure(self, ctx: ExtCtx) -> NDArray[np.float32]:
        """Required ctx: 'sents'

        Complexity: O(n), where n is number of tokens in all sentences
        """
        MAX_TREE_DEPTH = 17
        MAX_BRANCHING_FACTOR = 18
        MAX_TREE_WIDTH = 56

        depths = np.zeros(MAX_TREE_DEPTH + 1, dtype=np.float32)
        branching_factors = np.zeros(MAX_BRANCHING_FACTOR, dtype=np.float32)
        widths = np.zeros(MAX_TREE_WIDTH, dtype=np.float32)

        def traverse_tree(token: SentenceToken) -> tuple[int, int]:
            """
            Single-pass traversal that computes depth and leaf count.
            Returns (max_depth, leaf_count).
            """
            if not token.children:
                # Leaf node
                return (0, 1)

            # Non-leaf node - record branching factor
            f = min(len(token.children), MAX_BRANCHING_FACTOR)
            branching_factors[f - 1] += 1

            max_child_depth = 0
            total_leaves = 0

            for child in token.children:
                child_depth, child_leaves = traverse_tree(child)
                max_child_depth = max(max_child_depth, child_depth)
                total_leaves += child_leaves

            return (1 + max_child_depth, total_leaves)

        for sent in ctx.sents:
            root = sent.root

            depth, width = traverse_tree(root)

            depth = min(depth, MAX_TREE_DEPTH)
            depths[depth] += 1

            width = min(width, MAX_TREE_WIDTH)
            widths[width - 1] += 1

        if ctx.sents:
            depths /= len(ctx.sents)
            widths /= len(ctx.sents)
        if (bf_sum := sum(branching_factors)) > 0:
            branching_factors /= bf_sum

        return np.concatenate([depths, branching_factors, widths])

    def extract_dependency_tree_relations(self, ctx: ExtCtx) -> NDArray[np.float32]:
        """Required ctx: 'sents'

        Returns n-grams with >= 2% ref. document frequency from 3 groups:
        1. Node n-grams (2-4-grams) - ascending path of node labels (POS tags)
        2. Relation n-grams (1-4-grams) - ascending path of edge labels (dependency relations)
        3. Complete n-grams (2-4-grams) - path with both node and edge labels
        """
        occurences = np.zeros(self._dep_tree_features_total, dtype=np.float32)

        def get_ascending_paths(
            token: SentenceToken,
            path_nodes: list[str],
            path_rels: list[str],
            visited: set[int],
        ):
            """Get all ascending paths starting from this token"""
            token_id = id(token)
            if token_id in visited:  # Avoid cycles
                return

            visited.add(token_id)
            path_nodes.append(token.pos)

            # Process node n-grams (2-4)
            path_len = len(path_nodes)
            for n in range(2, min(5, path_len + 1)):
                ngram = tuple(path_nodes[-n:])
                if ngram in self._dep_tree_ngram_map:
                    occurences[self._dep_tree_ngram_map[ngram]] += 1

            # Process relation n-grams (1-4)
            rel_len = len(path_rels)
            for n in range(1, min(5, rel_len + 1)):
                ngram = tuple(path_rels[-n:])
                if ngram in self._dep_tree_ngram_map:
                    occurences[self._dep_tree_ngram_map[ngram]] += 1

            # Process complete n-grams (2-4) - alternating node-rel-node
            for n in range(2, min(5, path_len + 1)):
                if rel_len >= n - 1:
                    complete_path = []
                    for i in range(n):
                        complete_path.append(path_nodes[path_len - n + i])
                        if i < n - 1:
                            complete_path.append(path_rels[rel_len - n + 1 + i])
                    complete_ngram = tuple(complete_path)
                    if complete_ngram in self._dep_tree_ngram_map:
                        occurences[self._dep_tree_ngram_map[complete_ngram]] += 1

            for child in token.children:
                path_rels.append(child.dep)
                get_ascending_paths(child, path_nodes, path_rels, visited)
                path_rels.pop()

            path_nodes.pop()
            visited.remove(token_id)

        for sent in ctx.sents:
            get_ascending_paths(sent.root, [], [], set())

        node_sum = np.sum(occurences[: self._node_features_end])
        if node_sum > 0:
            occurences[: self._node_features_end] /= node_sum

        relation_sum = np.sum(
            occurences[self._node_features_end : self._relation_features_end]
        )
        if relation_sum > 0:
            occurences[self._node_features_end : self._relation_features_end] /= (
                relation_sum
            )

        complete_sum = np.sum(occurences[self._relation_features_end :])
        if complete_sum > 0:
            occurences[self._relation_features_end :] /= complete_sum

        return occurences

    def extract_noun_phrase_lengths(self, ctx: ExtCtx) -> NDArray[np.float32]:
        """Required ctx: 'words', 'noun_chunks'

        Time complexity: O(n), where n is number of noun chunks

        Returns n-dimensional vector, where each dimension i (1 ≤ i < n) represents the relative frequency of noun phrases within the document with the length of i words. The last dimension n is reserved for noun phrases longer than or equal to n.
        """
        DIM = 14
        occurences = np.zeros(DIM, dtype=np.float32)
        if not ctx.noun_chunks or not ctx.words:
            return occurences

        for chunk in ctx.noun_chunks:
            length = chunk.length
            if length < DIM:
                occurences[length - 1] += length
            else:
                occurences[DIM - 1] += length
        return occurences / len(ctx.words)

    def extract_entity_categories(self, ctx: ExtCtx) -> NDArray[np.float32]:
        """Required ctx: 'words', 'entities'

        Time complexity: O(n), where n is number of entities
        """
        category_counts = np.zeros(
            len(FeatureExtractorPipeline.ENTITY_CATEGORY_MAP), dtype=np.float32
        )
        if not ctx.entities or not ctx.words:
            return category_counts

        seen_corefs = set()

        for entity in ctx.entities:
            if entity.coref in seen_corefs:
                continue
            seen_corefs.add(entity.coref)
            if entity.cat in FeatureExtractorPipeline.ENTITY_CATEGORY_MAP:
                category_counts[
                    FeatureExtractorPipeline.ENTITY_CATEGORY_MAP[entity.cat]
                ] += 1
        return category_counts / len(ctx.words)

    def extract_events(self, ctx: ExtCtx) -> float:
        """Required ctx: 'words'

        Time complexity: O(n), where n is number of words
        """
        return (
            sum(token.event for token in ctx.words) / len(ctx.words)
            if ctx.words
            else 0.0
        )

    def extract_supersense(self, ctx: ExtCtx) -> NDArray[np.float32]:
        """Required ctx: 'words', 'supersense'

        Time complexity: O(n), where n is number of supersense annotations
        """
        label_counts = np.zeros(
            len(FeatureExtractorPipeline.SUPERSENSE_LABEL_MAP), dtype=np.float32
        )
        if not ctx.supersense or not ctx.words:
            return label_counts

        for _, _, label, _ in ctx.supersense:
            if label in FeatureExtractorPipeline.SUPERSENSE_LABEL_MAP:
                label_counts[FeatureExtractorPipeline.SUPERSENSE_LABEL_MAP[label]] += 1
        return label_counts / len(ctx.words)

    def extract_tense(self, ctx: ExtCtx) -> NDArray[np.float32]:
        """Required ctx: 'words'

        Time complexity: O(n), where n is number of words
        """
        # [past, present, future, none]
        tense_counts = np.zeros(4, dtype=np.float32)
        if not ctx.words:
            return tense_counts

        for token in ctx.words:
            tense = token.morph_tense
            if tense == "Past":
                tense_counts[0] += 1
            elif tense == "Pres":
                tense_counts[1] += 1
            elif tense == "Fut":
                tense_counts[2] += 1
            else:
                tense_counts[3] += 1
        return tense_counts / len(ctx.words)

    def extract_polysemy(self, ctx: ExtCtx) -> NDArray[np.float32]:
        """Required ctx: 'words'

        Time complexity: O(n), where n is number of words
        """
        BINS = 15
        polysemy_counts = np.zeros(BINS, dtype=np.float32)
        if not ctx.words:
            return polysemy_counts

        counted_words = 0
        for token in ctx.words:
            if token.pos not in FeatureExtractorPipeline.UD_TO_WN_POS:
                continue
            senses = self._wordnet.synsets(
                token.lemma, pos=FeatureExtractorPipeline.UD_TO_WN_POS[token.pos]
            )
            if len(senses) == 0:
                continue
            bin_idx = min(len(senses), BINS) - 1
            polysemy_counts[bin_idx] += 1
            counted_words += 1

        if counted_words == 0:
            return polysemy_counts
        return polysemy_counts / counted_words

    def extract_word_concreteness(self, ctx: ExtCtx) -> NDArray[np.float32]:
        """Required ctx: 'tokens', 'words'

        Time complexity: O(n*m), where n is number of words, m is max expression length

        Multi-word expressions (just expressions from now on) have priority.
        Expressions are counted as 1 word and total count of words is reduced with every found expression.
        This scores ~94 % of words in reference dataset.

        Returns (BINS+1)-dimensional vector where first BINS dimensions are histogram frequencies
        and the last dimension is the average concreteness score.
        """
        BINS = 20
        occurences = np.zeros(BINS, dtype=np.float32)
        if not ctx.tokens:
            return np.zeros(BINS + 1, dtype=np.float32)

        # Find multi-word expression matches first (longest preferred)
        multiword_matches = self._multiword_trie.find_longest_matches(ctx.tokens)

        if not multiword_matches:
            return np.zeros(BINS + 1, dtype=np.float32)

        total_score = 0.0
        for _, _, score in multiword_matches:
            # Count as single word regardless of actual length
            index = min(int(score * BINS), BINS - 1)
            occurences[index] += 1
            total_score += score

        # Calculate effective word count (original count - multi-word matches + match count)
        # end_idx is inclusive - match_count addition in accounted in the reduction (difference is always 1 word short)
        multiword_reduction = sum(
            end_idx - start_idx for start_idx, end_idx, _ in multiword_matches
        )
        effective_word_count = len(ctx.words) - multiword_reduction
        rel_freq = occurences / effective_word_count
        avg_score = total_score / len(multiword_matches)

        return np.concatenate([rel_freq, np.array([avg_score], dtype=np.float32)])

    def extract_preposition_imageability(self, ctx: ExtCtx) -> NDArray[np.float32]:
        """Required ctx: 'tokens', 'words'

        Time complexity: O(mk + n), where m is number of words, k is average multi-word phrase length, and n is number of matched prepositions

        Returns: (BINS+1)-dimensional vector where first BINS dimensions are histogram frequencies
        and the last dimension is the average imageability value.
        """
        BINS = 10

        tokens = ctx.tokens
        if not tokens or not ctx.words:
            return np.zeros(BINS + 1, dtype=np.float32)

        imageability_values = []
        matched_positions = set()

        # Multiword
        for prep_phrase, imag_val in self._multi_word_patterns.items():
            phrase_words = prep_phrase.split()
            phrase_len = len(phrase_words)

            for i in range(len(tokens) - phrase_len + 1):
                if i in matched_positions:
                    continue
                if all(
                    tokens[i + j].itext == phrase_words[j] for j in range(phrase_len)
                ):
                    if not any(
                        pos in matched_positions for pos in range(i, i + phrase_len)
                    ):
                        imageability_values.append(imag_val)
                        matched_positions.update(range(i, i + phrase_len))

        # Singleword regex
        for compiled_pattern, imag_val in self._compiled_regex_patterns.values():
            for i, tok in enumerate(tokens):
                if i in matched_positions:
                    continue
                if compiled_pattern.search(tok.itext):
                    imageability_values.append(imag_val)
                    matched_positions.add(i)

        # Singleword
        for i, token in enumerate(tokens):
            if i in matched_positions:
                continue

            matched = (
                token.itext
                if token.itext in self._exact_match_patterns
                else (
                    token.lemma if token.lemma in self._exact_match_patterns else None
                )
            )

            if matched is not None:
                val, pos_adp, pos_nonadp = self._exact_match_patterns[matched]
                final_val = None

                if token.pos == "ADP" and not np.isnan(pos_adp):
                    final_val = pos_adp
                elif token.pos != "ADP" and not np.isnan(pos_nonadp):
                    final_val = pos_nonadp
                elif np.isnan(pos_adp) and np.isnan(pos_nonadp):
                    final_val = val

                if final_val is not None:
                    imageability_values.append(final_val)
                    matched_positions.add(i)

        if not imageability_values:
            return np.zeros(BINS + 1, dtype=np.float32)

        hist_counts = np.zeros(BINS, dtype=np.float32)
        total = 0.0
        for imag_val in imageability_values:
            bin_idx = min(int(imag_val * BINS), BINS - 1)
            hist_counts[bin_idx] += 1
            total += imag_val
        hist_freqs = hist_counts / len(ctx.words)
        avg_imageability = total / len(imageability_values)

        return np.concatenate(
            [hist_freqs, np.array([avg_imageability], dtype=np.float32)]
        )

    def extract_places(self, ctx: ExtCtx) -> float:
        """Required ctx: 'tokens', 'words'

        Time complexity: O(n*m), where n is number of words, m is max place name length
        """
        if not ctx.tokens or not ctx.words:
            return 0.0
        place_matches = self._places_trie.find_longest_matches(ctx.tokens)
        return len(place_matches) / len(ctx.words)


class MultiWordExpressionTrie:
    """
    Trie-based data structure for efficient multi-word expression matching.
    Supports both original text and lemma lookups with longest match preference.
    """

    def __init__(self):
        self.root = {}

    def add_expression(self, expression: str, score: float, n_words: int):
        """Add an expression to the trie with both original and lemmatized versions"""
        words = expression.casefold().split()
        self._add_to_trie(self.root, words, score, n_words)

    def _add_to_trie(self, node: dict, words: list[str], score: float, n_words: int):
        """Helper to add words to trie"""
        for word in words:
            if word not in node:
                node[word] = {}
            node = node[word]
        # Mark end of expression with score and length
        node["__END__"] = (score, n_words)

    def find_longest_matches(self, tokens) -> list[tuple[int, int, float]]:
        """
        Find all non-overlapping longest matches in the token sequence.
        Returns list of (start_idx, end_idx, score) tuples.

        Time complexity: O(n*m) where n is number of tokens, m is max expression length
        """
        matches = []
        used_positions = set()

        i = 0
        while i < len(tokens):
            if i in used_positions:
                i += 1
                continue

            # Try to find longest match starting at position i
            best_match = None

            # Try both original text and lemma for first token
            for first_token_text in [
                tokens[i].itext.casefold(),
                tokens[i].lemma.casefold(),
            ]:
                if first_token_text not in self.root:
                    continue

                match = self._find_match_from_position(tokens, i, first_token_text)
                if match and (best_match is None or match[1] > best_match[1]):
                    best_match = match

            if best_match:
                start_idx, end_idx, score = best_match
                matches.append((start_idx, end_idx, score))
                # Mark all positions in this match as used
                for pos in range(start_idx, end_idx + 1):
                    used_positions.add(pos)
                i = end_idx + 1
            else:
                i += 1

        return matches

    def _find_match_from_position(self, tokens, start_idx: int, first_token_text: str):
        """Find longest match starting from given position and first token text"""
        node = self.root[first_token_text]
        best_match = None

        # Check if single word match
        if "__END__" in node:
            score, n_words = node["__END__"]
            best_match = (start_idx, start_idx, score)

        # Try to extend the match
        for i in range(start_idx + 1, len(tokens)):
            current_token = tokens[i]
            found_continuation = False

            # Try both original text and lemma for continuation
            for token_text in [
                current_token.itext.casefold(),
                current_token.lemma.casefold(),
            ]:
                if token_text in node:
                    node = node[token_text]
                    found_continuation = True

                    # Check if this is end of an expression
                    if "__END__" in node:
                        score, n_words = node["__END__"]
                        best_match = (start_idx, i, score)
                    break

            if not found_continuation:
                break

        return best_match


class FeatureService:
    def __init__(
        self,
        feature_pipeline_resources: FeatureExtractorPipelineResources,
        cache_dir: str | None = None,
    ):
        self.embed_minilm = TextEmbedding(
            "sentence-transformers/all-MiniLM-L6-v2",
            cache_folder=cache_dir,
        )
        # Not needed yet
        # self.embed_modernbert = SentenceTransformer(
        #     "lightonai/modernbert-embed-large",
        #     truncate_dim=256,
        #     backend="onnx",
        #     cache_folder=cache_dir,
        #     model_kwargs={"file_name": "model_quantized.onnx"},
        # )

        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     "lightonai/modernbert-embed-large"
        # )
        # # Load ModernBERT model from onnx file
        # self.sess = rt.InferenceSession(modernbert_onnx_path)
        # self.input_name = self.sess.get_inputs()[0].name
        # self.output_name = self.sess.get_outputs()[0].name

        self.feature_extractor = FeatureExtractorPipeline(
            resources=feature_pipeline_resources
        )

    def get_modernbert_embeddings(self, texts: list[str]) -> np.ndarray:
        """
        Computes embeddings for a list of texts using the ONNX lightonai/modernbert-embed-large model.
        """
        # Transform into nomic-embed format (https://huggingface.co/nomic-ai/nomic-embed-text-v1.5#task-instruction-prefixes)
        texts = [f"classification: {text}" for text in texts]
        return self.embed_modernbert.encode(texts)

        # # Tokenize the input texts
        # inputs = self.tokenizer(
        #     texts, padding=True, truncation=True, return_tensors="np"
        # )

        # # Prepare inputs for ONNX runtime
        # # The model expects a dictionary of its inputs, here we use 'input_ids'
        # # Your model might have other inputs like 'attention_mask'.
        # # You can check sess.get_inputs() to be sure.
        # onnx_inputs = {name.name: inputs[name.name] for name in self.sess.get_inputs()}

        # # Run inference
        # last_hidden_state = self.sess.run([self.output_name], onnx_inputs)[0]

        # # Perform mean pooling to get the sentence embedding
        # attention_mask = inputs["attention_mask"]
        # mask_expanded = np.expand_dims(attention_mask, -1)
        # sum_embeddings = np.sum(last_hidden_state * mask_expanded, 1)
        # sum_mask = np.sum(mask_expanded, 1)
        # # Clamp sum_mask to avoid division by zero
        # sum_mask = np.maximum(sum_mask, 1e-9)
        # mean_pooled_embeddings = sum_embeddings / sum_mask

        # return mean_pooled_embeddings

    def get_features(
        self,
        texts: list[str],
        model: Literal[
            "sentence-transformers/all-MiniLM-L6-v2", "lightonai/modernbert-embed-large"
        ],
    ) -> list[list[float]]:
        if model == "lightonai/modernbert-embed-large":
            embeddings = self.get_modernbert_embeddings(texts)
        else:
            embeddings = self.embed_minilm.encode(texts)

        features = np.array([self.feature_extractor.extract(text) for text in texts])
        embeddings = np.hstack((embeddings, features))

        return embeddings
