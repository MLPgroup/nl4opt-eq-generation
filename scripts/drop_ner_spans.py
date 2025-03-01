import json
import random
import re
import numpy as np
from argparse import ArgumentParser


class DropNERSpans:
    def __init__(self, dataset_file: str, out_file: str, prob: float = 0.15, max_span_width: int = 4):
        self.dataset_file = dataset_file
        self.out_file = out_file
        self.prob = prob
        self.max_span_width = max_span_width

        all_labels = set()
        with open(self.dataset_file, "r") as f:
            self.examples = [json.loads(line) for line in f]
            for example in self.examples:
                labels = [span["label"] for key, doc in example.items() for span in doc["spans"]]
                all_labels.update(labels)

        self.all_labels = list(all_labels)


    def is_overlapping(self, span1, span2):
        return span2[0] <= span1[1] <= span2[1] or span1[0] <= span2[1] <= span1[1]


    def find_non_overlapping_spans(self, existing_spans, possible_spans):
        non_overlapping_spans = []
        for possible_span in possible_spans:
            overlap = any([self.is_overlapping(possible_span, existing_span) for existing_span in existing_spans])
            if not overlap:
                non_overlapping_spans.append(possible_span)

        return non_overlapping_spans


    def find_possible_spans(self, document, spans):
        existing_spans = [(span["start"], span["end"]) for span in spans]
        space_indices = np.array([m.start() for m in re.finditer(r"\s", document) if document[m.start() - 1] != "." and document[m.end()] != "."])
        start_indices = space_indices[:-1] + 1
        end_indices = space_indices[1:]

        possible_spans = []
        for width in range(self.max_span_width):
            possible_spans.extend(list(zip(start_indices[: -width], end_indices[width :])))

        return self.find_non_overlapping_spans(existing_spans, possible_spans)


    def noisy_spans(self, document, spans):
        noisy_spans = []
        possible_spans = self.find_possible_spans(document, spans)

        used_spans = []
        for span in spans:
            rand = random.random()
            if rand > self.prob:
                noisy_spans.append(span)
                continue

        return noisy_spans


    def process(self):
        for example in self.examples:
            assert len(example) == 1

            key = next(iter(example.keys()))
            noisy_spans = self.noisy_spans(example[key]["document"], example[key]["spans"])
            example[key]["spans"] = noisy_spans

        with open(self.out_file, "w") as f:
            for example in self.examples:
                f.write(json.dumps(example))
                f.write("\n")


def main():
    parser = ArgumentParser("Script to drop NER spans.")
    parser.add_argument("--dataset_file", type = str, required = True)
    parser.add_argument("--out_file", type = str, required = True)
    parser.add_argument("--prob", type = float, default = 0.15)
    parser.add_argument("--seed", type = int, default = 42)

    args = parser.parse_args()

    random.seed(args.seed)

    ner_predictor = DropNERSpans(
        dataset_file = args.dataset_file,
        out_file = args.out_file,
        prob = args.prob,
    )
    ner_predictor.process()


if __name__ == "__main__":
    main()
