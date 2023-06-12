import os
import argparse
import json
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.utils import load_model, get_tagset
from utils.reader import CoNLLReader


class NERPredictor:
    """
    This class predicts the named entities for the problem descriptions in an input dataset,
    specified by `datafile`, using a trained NER model, specified by `ner_model_dir`. It
    replaces the "spans" property present in the input dataset with the predicted NER spans
    and saves the output to a path, specified by `outfile`. It does not modify the input file.
    Only NL4Opt baseline model is supported. To run this script, a Python environment for the
    NL4Opt NER baseline model is required. Refer to https://github.com/nl4opt/nl4opt-subtask1-baseline
    to setup the Python environment and train a model.

    Parameters
    ----------
    ner_model_dir: str
        Directory with a trained NER model.

    datafile: str
        NL4Opt generation dataset file, for example, train.jsonl, dev.jsonl, or test.jsonl.

    outfile: str
        A complete path for saving the output.
    """
    def __init__(self, ner_model_dir: str, datafile: str, outfile: str) -> None:
        assert os.path.exists(ner_model_dir), f"{ner_model_dir} does not exist!"
        assert os.path.exists(datafile), f"{datafile} does not exist!"
        assert not os.path.exists(outfile), f"{outfile} already exists!"

        self.ner_model_dir = ner_model_dir
        self.datafile = datafile
        self.outfile = outfile
        self.batch_size = 32

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.target_vocab = get_tagset("conll")
        self.model, _ = load_model(self.ner_model_dir, tag_to_id = self.target_vocab)
        self.model.to(self.device)
        self.model.eval()


    def read_example(self, example):
        keys = example.keys()
        assert len(keys) == 1
        key = next(iter(keys))
        example = example[key]

        max_len = max(token["id"] for token in example["tokens"]) + 1
        tokens = [None] * max_len
        for token in example["tokens"]:
            tokens[token["id"]] = token["text"]

        spans = ["O"] * max_len
        for span in example["spans"]:
            start = span["token_start"]
            end = span["token_end"] + 1
            for idx, span_idx in enumerate(range(start, end)):
                label = f"B-{span['label']}" if idx == 0 else f"I-{span['label']}"
                spans[span_idx] = label

        return [
            tokens,
            ["_"] * max_len,
            ["_"] * max_len,
            spans,
        ]


    def get_ner_data(self, data: str):
        with open(data) as f:
            for line in f:
                example = json.loads(line)
                yield self.read_example(example), None


    def get_dataset(self, filepath: str):
        reader = CoNLLReader(
            max_instances = 10000,
            max_length = 512,
            target_vocab = self.target_vocab,
            encoder_model = self.model.encoder_model,
        )
        reader.read_data(filepath, self.get_ner_data)
        return reader


    def compute_spans(self, example, predicted_tags):
        assert len(example.keys()) == 1
        key = next(iter(example.keys()))

        # Validate
        # The following is required because there are tokens with text ' ' which causes tokenization
        # to not return anything and hence to predicted tags for that token.
        tokens_wo_ws = [token for token in example[key]["tokens"] if len(token["text"].replace(" ", "")) > 0]
        assert len(tokens_wo_ws) == len(predicted_tags), f"tokens w/o whitespace: {len(tokens_wo_ws)}, tags: {len(predicted_tags)}"

        # Process
        span_idx = 0
        prev_label, prev_start, prev_end, prev_start_idx, prev_end_idx  = None, -1, -1, -1, -1

        spans = []
        document = ""
        # token_idx and span_idx are required because of whitespace tokens present in example["tokens"]
        # as the model skip these tokens.
        for token_idx, token in enumerate(example[key]["tokens"]):
            char_start = len(document)
            document += token["text"]
            char_end = len(document)
            if token["ws"]:
                document += " "

            if len(token["text"].replace(" ", "")) == 0:
                continue

            curr_tag = predicted_tags[span_idx]
            if curr_tag == "O" or curr_tag.startswith("B-"):
                if prev_label is not None:
                    # handle previous span
                    span = {
                        "label": prev_label,
                        "token_start": prev_start,
                        "token_end": prev_end,
                        "start": prev_start_idx,
                        "end": prev_end_idx,
                        "text": document[prev_start_idx : prev_end_idx],
                        "type": "span",
                    }
                    spans.append(span)
                    prev_label, prev_start, prev_end, prev_start_idx, prev_end_idx  = None, -1, -1, -1, -1

            if curr_tag.startswith("B-"):
                prev_label = curr_tag[2:]
                prev_start = token_idx
                prev_end = token_idx
                prev_start_idx = char_start
                prev_end_idx = char_end

            if curr_tag.startswith("I-"):
                prev_end = token_idx
                prev_end_idx = char_end

            span_idx += 1

        if prev_label is not None:
            span = {
                "label": prev_label,
                "token_start": prev_start,
                "token_end": prev_end,
                "start": prev_start_idx,
                "end": prev_end_idx,
                "text": document[prev_start_idx : prev_end_idx],
                "type": "span",
            }
            spans.append(span)

        return spans


    def predict_spans(self):
        dataset = self.get_dataset(self.datafile)
        dataloader = DataLoader(dataset, batch_size = self.batch_size, shuffle = False, collate_fn = self.model.collate_batch)

        predicted_tags = []
        for batch in tqdm(dataloader):
            preds = self.model.predict_tags(batch, self.device)
            for pred in preds:
                predicted_tags.append([s for s in pred])

        return predicted_tags


    def process(self):
        examples = []
        with open(self.datafile, "r") as f:
            for line in f:
                examples.append(json.loads(line))

        predicted_tags = self.predict_spans()
        assert len(examples) == len(predicted_tags)
        print(f"Number of examples: {len(examples)}\nNumber of predicted spans: {len(predicted_tags)}")

        for example, tags in zip(examples, predicted_tags):
            predicted_spans = self.compute_spans(example, tags)
            example[next(iter(example.keys()))]["spans"] = predicted_spans

        with open(self.outfile, "w") as f:
            for example in examples:
                f.write(json.dumps(example))
                f.write("\n")


def main():
    parser = argparse.ArgumentParser("Script to predict the NER spans.")
    parser.add_argument("--ner_model_dir", type = str, required = True)
    parser.add_argument("--datafile", type = str, required = True)
    parser.add_argument("--outfile", type = str, required = True)
    args = parser.parse_args()

    ner_predictor = NERPredictor(ner_model_dir = args.ner_model_dir, datafile = args.datafile, outfile = args.outfile)
    ner_predictor.process()


if __name__ == "__main__":
    main()
