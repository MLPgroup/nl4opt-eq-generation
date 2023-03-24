"""
This script can be used to predict NER spans for the generation dataset
rather than using the ground truth NER spans provided with the dataset.
This script is a part of experiments that test the model performance
when the NER span information is not available in the dataset.
This requires a model trained to predict NER. To train the baseline model,
refer to https://github.com/nl4opt/nl4opt-subtask1-baseline.

To run the script, add the path to `nl4opt-subtask1-baseline` directory to
`PYTHONPATH`. Note that this script only works for NER models trained with
the baseline code. Make sure the dependencies mentioned in `nl4opt-subtask1-baseline`
are installed before running this script.
"""
import json
import torch
from utils.utils import get_reader, load_model, get_tagset
from transformers import AutoTokenizer
from argparse import ArgumentParser


class NERPredictor:
    def __init__(self, dataset_file: str, out_file: str, ner_model_dir: str):
        self.dataset_file = dataset_file
        self.ner_model_dir = ner_model_dir
        self.out_file = out_file
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Load the dataset
        with open(self.dataset_file, "r") as f:
            self.examples = [json.loads(line) for line in f]

        # Load the NER model
        self.target_vocab = get_tagset("conll")
        self.model, _ = load_model(self.ner_model_dir, tag_to_id = self.target_vocab)
        self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model.encoder_model)


    def predict_spans(self, example):
        document = example["document"].lower()
        tokens = self.tokenizer(document, add_special_tokens = False, return_tensors = "pt")["input_ids"]
        tokens = tokens.to(self.device)
        mask = torch.ones_like(tokens, dtype = torch.bool, device = "cuda")
        token_mask = torch.ones_like(tokens, dtype = torch.bool, device = self.device)
        dummy_tags = torch.ones_like(tokens, device = "cuda").fill_(self.target_vocab["O"])
        dummy_metadata = {}

        batch = tokens, dummy_tags, mask, token_mask, dummy_metadata

        output = self.model.perform_forward_step(batch, mode = "predict")
        predicted_tags = output["token_tags"][0]

        lentillnow = 0
        prev_tag = None
        start_idx = None
        end_idx = None

        spans = {}
        for idx, (token, tag) in enumerate(zip(tokens[0], predicted_tags)):
            token = self.tokenizer.convert_ids_to_tokens([token])[0]
            if idx == 0:
                start = 0
                end = len(token) - 1
            else:
                start = lentillnow
                end += len(token)

            lentillnow = end

            curr_tag = tag.split("-")[-1]
            begin = tag.split("-")[0] == "B"

            if curr_tag != prev_tag and prev_tag is not None and prev_tag != "O":
                assert start_idx and end_idx
                spans[(start_idx, end_idx)] = prev_tag

            if curr_tag != "O":
                if begin:
                    if token.startswith("▁$"):
                        start_idx = start + 2
                    elif token.startswith("▁") or token.startswith("$"):
                        start_idx = start + 1
                    else:
                        start_idx = start
                    end_idx = end
                else:
                    end_idx = end

            prev_tag = curr_tag

        if prev_tag is not None and prev_tag != "O":
            assert start_idx and end_idx
            spans[(start_idx, end_idx)] = prev_tag

        return self.tokenizer.decode(tokens[0]), spans


    def process(self):
        for example in self.examples:
            assert len(example) == 1

            key = next(iter(example.keys()))
            pred_document, pred_spans = self.predict_spans(example[key])
            pred_span_list = []
            for span, label in pred_spans.items():
                pred_span_list.append({
                    "start": span[0],
                    "end": span[1],
                    "label": label,
                })

            example[key]["pred_document"] = pred_document
            example[key]["pred_spans"] = pred_span_list

        with open(self.out_file, "w") as f:
            for example in self.examples:
                f.write(json.dumps(example))
                f.write("\n")


def main():
    parser = ArgumentParser("Script to predict NER spans.")
    parser.add_argument("--dataset_file", type = str, required = True)
    parser.add_argument("--out_file", type = str, required = True)
    parser.add_argument("--ner_model_dir", type = str, required = True)

    args = parser.parse_args()

    ner_predictor = NERPredictor(
        dataset_file = args.dataset_file,
        out_file = args.out_file,
        ner_model_dir = args.ner_model_dir,
    )
    ner_predictor.process()


if __name__ == "__main__":
    main()
