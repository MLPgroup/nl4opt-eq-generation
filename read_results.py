import argparse
import torch
import glob
import json
import statistics as s


def read_results(dirname, beam_size):
    dirlist = glob.glob(f"output/{dirname}/*")
    score_dict = {}
    for dirpath in dirlist:
        result_file = f"{dirpath}/test_results_bs{beam_size}.json"
        ckpt = f"{dirpath}/best-checkpoint.mdl"
        seed = torch.load(ckpt, map_location = torch.device("cpu"))["config"]["seed"]
        with open(result_file) as f:
            score = f.read()
            score_dict[seed] = round(float(score), 3)
    
    return score_dict


def main():
    parser = argparse.ArgumentParser("Script to read results")
    parser.add_argument("--config", type = str, required = True)
    parser.add_argument("--beam_size", type = int, required = True)
    parser.add_argument("--n_expected_runs", type = int, required = True)
    args = parser.parse_args()

    results = read_results(args.config, args.beam_size)
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

