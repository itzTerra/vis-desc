class NLIZeroshotClassifier:
    def __init__(
        self, candidate_labels=None, hypothesis_template=None, create_model=True
    ):
        if candidate_labels is not None:
            self.candidate_labels = candidate_labels
        if hypothesis_template is not None:
            self.hypothesis_template = hypothesis_template
        self.classifier = None
        if create_model:
            self.create_model()

    # Resource handler usable with the 'with' statement
    class Opts:
        def __init__(
            self,
            parent_classifier: "NLIZeroshotClassifier",
            candidate_labels=None,
            hypothesis_template=None,
        ):
            self.parent_classifier = parent_classifier
            self.candidate_labels = candidate_labels
            self.hypothesis_template = hypothesis_template

        def __enter__(self):
            if self.candidate_labels is not None:
                self.old_candidate_labels = self.parent_classifier.candidate_labels
                self.parent_classifier.candidate_labels = self.candidate_labels
            if self.hypothesis_template is not None:
                self.old_hypothesis_template = (
                    self.parent_classifier.hypothesis_template
                )
                self.parent_classifier.hypothesis_template = self.hypothesis_template
            return self.parent_classifier

        def __exit__(self, exc_type, exc_value, traceback):
            if self.candidate_labels is not None:
                self.parent_classifier.candidate_labels = self.old_candidate_labels
            if self.hypothesis_template is not None:
                self.parent_classifier.hypothesis_template = (
                    self.old_hypothesis_template
                )

    def set_options(self, candidate_labels=None, hypothesis_template=None):
        return self.Opts(self, candidate_labels, hypothesis_template)

    def create_model(self):
        raise NotImplementedError

    def evaluate_segments(self, segments: list[str]) -> list[list[float]]:
        self.create_model()

        results = self.classifier(
            segments,
            self.candidate_labels,
            hypothesis_template=self.hypothesis_template,
        )
        # Return scores in the order of candidate labels
        return [
            [
                result["scores"][result["labels"].index(label)]
                for label in self.candidate_labels
            ]
            for result in results
        ]
