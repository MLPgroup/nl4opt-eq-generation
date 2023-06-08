# NL4Opt Subtask 2
This repository contains the source code of "Named Entity-Based Enrichment with BART" for the second subtask of NL4Opt. Refer to [the NL4Opt page](https://nl4opt.github.io/) for details about the subtask.

This repository is based on the baseline code for the second subtask which can be found at [the NL4Opt's official implementation](https://github.com/nl4opt/nl4opt-subtask2-baseline). Check their [README.md](https://github.com/nl4opt/nl4opt-subtask2-baseline/blob/main/README.md) for details.
# Environment Setup
Use `environment.yml` to setup the environment:

```
conda env create -f environment.yml -n <ENV_NAME>
conda activate <ENV_NAME>
```

Verify that it was installed:
```
conda env list
```

# Running the Pipeline
## Download the Dataset
First, copy the dataset files `train.jsonl` and `dev.jsonl` to the `data` subdirectory. The dataset files can be found under the `generation_task` subdirectory in [the dataset repository](https://github.com/nl4opt/nl4opt-competition).

## Training Configurations
The config files for training are present in the `configs` subdirectory.
- `baseline.json`: Config for the baseline model.
- `default.json`: Config for our approach that we used for the final submission.

## Running the Pipeline
The training and testing pipeline can be run using `train_and_evaluate.sh`. This script expects Miniconda installed under `~/miniconda3/` and `test.jsonl` present under the `data` subdirectory.
```
bash train_and_evaluate.sh
```

## Training and Evaluating the Model
To train the model, run the following:
```
python train.py --config configs/default.json --seed 42
```
The important parameters in the training are:
- `use_copy` uses a copy mechanism that computes $P_\text{copy}$ over the input tokens.
- `per_declaration` controls each training data sample to correspond to a single declaration of a given LP problem instead of the entire formulation (i.e. all declarations in the problem).
- `use_prompt` uses a declaration prompt to focus the generation. For example, the `<OBJ_DIR>` is used as a prompt for generating the objective declaration.
- `enrich_ner` controls if the named entity information should be added to the input before feeding it to the model.

To evaluate the model, run the following:
```
python test.py --gpu <gpu id> --checkpoint <checkpoint.mdl> --test-file <test.jsonl> --batch-size <test_batch_size> --beam-size <beam_size>
```

# Contact
For any queries, feel free to reach out to gangwar2 [at] illinois [dot] edu.