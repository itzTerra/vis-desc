# from typing import Iterable
# from transformers import pipeline
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from multiprocessing import Manager, cpu_count, set_start_method

# from core.tools.evaluators.base import BaseEvaluator


# class DebertaMNLIEvaluator(BaseEvaluator):
#     candidate_labels = ["descriptive", "non_descriptive"]
#     hypothesis_template = "This text is {} in terms of visual details of characters, setting, or environment."

#     def __init__(self, candidate_labels=None, hypothesis_template=None, create_model=True):
#         if candidate_labels is not None:
#             self.CANDIDATE_LABELS = candidate_labels
#         if hypothesis_template is not None:
#             self.HYPOTHESIS_TEMPLATE = hypothesis_template
#         self.classifier = None
#         if create_model:
#             self.create_model()

#     # Resource handler usable with the 'with' statement
#     class Opts:
#         def __init__(
#             self,
#             parent_classifier: "DebertaMNLIEvaluator",
#             candidate_labels=None,
#             hypothesis_template=None,
#         ):
#             self.parent_classifier = parent_classifier
#             self.candidate_labels = candidate_labels
#             self.hypothesis_template = hypothesis_template

#         def __enter__(self):
#             if self.candidate_labels is not None:
#                 self.old_candidate_labels = self.parent_classifier.candidate_labels
#                 self.parent_classifier.candidate_labels = self.candidate_labels
#             if self.hypothesis_template is not None:
#                 self.old_hypothesis_template = (
#                     self.parent_classifier.hypothesis_template
#                 )
#                 self.parent_classifier.hypothesis_template = self.hypothesis_template
#             return self.parent_classifier

#         def __exit__(self, exc_type, exc_value, traceback):
#             if self.candidate_labels is not None:
#                 self.parent_classifier.candidate_labels = self.old_candidate_labels
#             if self.hypothesis_template is not None:
#                 self.parent_classifier.hypothesis_template = (
#                     self.old_hypothesis_template
#                 )

#     def set_options(self, candidate_labels=None, hypothesis_template=None):
#         return self.Opts(self, candidate_labels, hypothesis_template)

#     def create_model(self):
#         if self.classifier is None:
#             self.classifier = pipeline(
#                 "zero-shot-classification", model="microsoft/deberta-base-mnli"
#             )

#     def evaluate_segment(self, segment: str) -> float:
#         self.create_model()

#         assert self.classifier is not None, "Classifier model is not created"

#         result = self.classifier(
#             segment,
#             self.candidate_labels,
#             hypothesis_template=self.hypothesis_template,
#         )
#         positive_score, negative_score = result["scores"] # type: ignore
#         return positive_score - negative_score # type: ignore

#     def evaluate_segment_batch(self, segments: list[str], classifier=None) -> list[float]:
#         if not classifier:
#             self.create_model()
#             classifier = self.classifier
#         assert classifier is not None, "Classifier model is not created"
#         results = classifier(
#             segments,
#             self.candidate_labels,
#             hypothesis_template=self.hypothesis_template,
#         )
#         return [result["scores"][0] - result["scores"][1] for result in results] # type: ignore

#     def evaluate_segments(
#         self, segments: list[str], use_multiprocessing=False
#     ) -> Iterable[tuple[str, float]]:
#         self.create_model()
#         SCORE_THRESHOLD = 0.95
#         TO_SKIP = 8
#         batch_size = max(len(segments) // (cpu_count() * 3) + 1, 8)

#         skipped_segment_idx = []
#         total_skipped = 0

#         # Wave 1
#         index = 0
#         while index < len(segments):
#             segment = segments[index]
#             score = self.evaluate_segment(segment)
#             yield (segment, score)
#             if score > SCORE_THRESHOLD:
#                 to_skip = min(TO_SKIP, len(segments) - index - 1)
#                 skipped_segment_idx.extend(range(index + 1, index + to_skip + 1))
#                 index += to_skip
#                 total_skipped += to_skip
#             index += 1
#         print(f"Total skipped in wave 1: {total_skipped}")
#         # print(f"Skipped segments: {skipped_segment_idx}")

#         # Wave 2
#         batches = []
#         for i in range(0, len(skipped_segment_idx), batch_size):
#             batch_segments = []
#             for j in range(i, min(i + batch_size, len(skipped_segment_idx))):
#                 batch_segments.append(segments[skipped_segment_idx[j]])
#             batches.append(batch_segments)
#         print(
#             f"Created {len(batches)} batches for wave 2 with batch size = {batch_size}."
#         )

#         if not use_multiprocessing:
#             for batch in batches:
#                 scores = self.evaluate_segment_batch(batch)
#                 for segment, score in zip(batch, scores):
#                     yield (segment, score)
#         else:
#             set_start_method("spawn", force=True)
#             with Manager() as manager:
#                 shared_classifier = manager.dict()
#                 shared_classifier["model"] = self.classifier

#                 with ProcessPoolExecutor(
#                     max_workers=min(len(batches), cpu_count())
#                 ) as executor:
#                     future_to_segment = {
#                         executor.submit(
#                             self.evaluate_segment_batch,
#                             batch,
#                             shared_classifier["model"],
#                         ): batch
#                         for batch in batches
#                     }
#                     for future in as_completed(future_to_segment):
#                         batch = future_to_segment[future]
#                         try:
#                             scores = future.result()
#                             for segment, score in zip(batch, scores):
#                                 yield (segment, score)
#                         except Exception as exc:
#                             print(f"Segment generated an exception: {exc}")

