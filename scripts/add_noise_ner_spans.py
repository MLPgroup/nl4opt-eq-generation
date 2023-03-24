import json
import random
from argparse import ArgumentParser


class AddNERSpanNoise:
    def __init__(self, dataset_file: str, out_file: str, prob: float = 0.15):
        self.dataset_file = dataset_file
        self.out_file = out_file
        self.prob = prob

        all_labels = set()
        with open(self.dataset_file, "r") as f:
            self.examples = [json.loads(line) for line in f]
            for example in self.examples:
                labels = [span["label"] for key, doc in example.items() for span in doc["spans"]]
                all_labels.update(labels)

        self.all_labels = list(all_labels)


    def noisy_spans(self, document, spans):
        noisy_spans = []
        for span in spans:
            rand = random.random()
            if rand > self.prob:
                noisy_spans.append(span)
                continue

            # Drop the span in half the cases.
            if rand <= self.prob / 2:
                continue

            # Replace label in half the cases.
            if rand > self.prob / 2:
                span["label"] = random.choice(self.all_labels)
                noisy_spans.append(span)

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
    parser = ArgumentParser("Script to predict NER spans.")
    parser.add_argument("--dataset_file", type = str, required = True)
    parser.add_argument("--out_file", type = str, required = True)
    parser.add_argument("--prob", type = float, default = 0.15)

    args = parser.parse_args()

    ner_predictor = AddNERSpanNoise(
        dataset_file = args.dataset_file,
        out_file = args.out_file,
        prob = args.prob,
    )
    ner_predictor.process()


if __name__ == "__main__":
    main()
