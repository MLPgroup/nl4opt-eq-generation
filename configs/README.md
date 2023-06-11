## Config Files

| Config Name                             | Model               | NER | Copy | Per Declaration | Noisy Datasets                    |
| --------------------------------------- | ------------------- | --- | ---- | --------------- | --------------------------------- |
| default.json                            | facebook/bart-large | Y   | Y    | N               | -                                 |
| default_nocopy.json                     | facebook/bart-large | Y   | N    | N               | -                                 |
| default_noner.json                      | facebook/bart-large | N   | Y    | N               | -                                 |
| default_noner_nocopy.json               | facebook/bart-large | N   | N    | N               | -                                 |
| default_base.json                       | facebook/bart-base  | Y   | Y    | N               | -                                 |
| default_base_nocopy.json                | facebook/bart-base  | Y   | N    | N               | -                                 |
| default_base_noner.json                 | facebook/bart-base  | N   | Y    | N               | -                                 |
| default_base_noner_nocopy.json          | facebook/bart-base  | N   | N    | N               | -                                 |
| default_noisy_p0.2.json                 | facebook/bart-large | Y   | Y    | N               | p=0.2                             |
| default_base_noisy_p0.2.json            | facebook/bart-base  | Y   | Y    | N               | p=0.2                             |
| default_noisy_p0.5.json                 | facebook/bart-large | Y   | Y    | N               | p=0.5                             |
| default_base_noisy_p0.5.json            | facebook/bart-base  | Y   | Y    | N               | p=0.5                             |
| default_predicted_ner.json              | facebook/bart-large | Y   | Y    | N               | Predicted NERs                    |
| default_predicted_ner_dropped.json      | facebook/bart-large | Y   | Y    | N               | Predicted NERs with Dropped Spans |
| default_base_predicted_ner.json         | facebook/bart-base  | Y   | Y    | N               | Predicted NERs                    |
| default_base_predicted_ner_dropped.json | facebook/bart-base  | Y   | Y    | N               | Predicted NERs with Dropped Spans |