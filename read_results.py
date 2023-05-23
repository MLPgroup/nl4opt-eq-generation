import argparse
import torch
import glob
import json
import statistics as s


def read_results(dirname, beam_size, is_copy, is_ner, is_per_declr):
    dirlist = glob.glob(f"output/{dirname}/*")
    score_dict = {}
    for dirpath in dirlist:
        result_file = f"{dirpath}/test_results_bs{beam_size}.json"
        ckpt = f"{dirpath}/best-checkpoint.mdl"
        config = torch.load(ckpt, map_location = torch.device("cpu"))["config"]
        seed = config["seed"]

        assert config["use_copy"] == is_copy
        assert config["enrich_ner"] == is_ner
        assert config["is_per_declr"] == is_per_declr

        with open(result_file) as f:
            score = f.read()
            score_dict[seed] = round(float(score), 3)

    return score_dict


def main():
    parser = argparse.ArgumentParser("Script to read results")
    parser.add_argument("--config", type = str, required = True)
    parser.add_argument("--beam_size", type = int, required = True)
    parser.add_argument("--n_expected_runs", type = int, required = True)
    parser.add_argument("--no_copy", action = "store_true", default = False)
    parser.add_argument("--no_ner", action = "store_true", default = False)
    parser.add_argument("--no_per_declr", action = "store_true", default = False)
    args = parser.parse_args()

    is_copy = not args.no_copy
    is_ner = not args.is_ner
    is_per_declr = not args.no_per_declr

    results = read_results(args.config, args.beam_size, is_copy = is_copy, is_ner = is_ner, is_per_declr = is_per_declr)
    assert len(results) == args.n_expected_runs

    if len(results) > 1:
        max_acc = max(results.values())
        mean_acc = round(s.mean(results.values()), 3)
        std_acc = round(s.stdev(results.values()), 3)
        print(f"all: {json.dumps(results)}\nmax: {max_acc}\nmean: {mean_acc} \u00B1 {std_acc}")
    else:
        print(f"all: {json.dumps(results)}")


if __name__ == "__main__":
    main()

