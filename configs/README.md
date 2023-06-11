## Config Files

| Config Name                             | Model               | NER       | Copy       | Per Declaration       | Noisy Datasets                    |
| --------------------------------------- | ------------------- | --------- | ---------- | --------------------- | --------------------------------- |
| default.json                            | facebook/bart-large | &check;   | &check;    | &cross;               | -                                 |
| default_nocopy.json                     | facebook/bart-large | &check;   | &cross;    | &cross;               | -                                 |
| default_noner.json                      | facebook/bart-large | &cross;   | &check;    | &cross;               | -                                 |
| default_noner_nocopy.json               | facebook/bart-large | &cross;   | &cross;    | &cross;               | -                                 |
| default_base.json                       | facebook/bart-base  | &check;   | &check;    | &cross;               | -                                 |
| default_base_nocopy.json                | facebook/bart-base  | &check;   | &cross;    | &cross;               | -                                 |
| default_base_noner.json                 | facebook/bart-base  | &cross;   | &check;    | &cross;               | -                                 |
| default_base_noner_nocopy.json          | facebook/bart-base  | &cross;   | &cross;    | &cross;               | -                                 |
| default_noisy_p0.2.json                 | facebook/bart-large | &check;   | &check;    | &cross;               | $p=0.2$                           |
| default_base_noisy_p0.2.json            | facebook/bart-base  | &check;   | &check;    | &cross;               | $p=0.2$                           |
| default_noisy_p0.5.json                 | facebook/bart-large | &check;   | &check;    | &cross;               | $p=0.5$                           |
| default_base_noisy_p0.5.json            | facebook/bart-base  | &check;   | &check;    | &cross;               | $p=0.5$                           |
| default_predicted_ner.json              | facebook/bart-large | &check;   | &check;    | &cross;               | Predicted NERs                    |
| default_predicted_ner_dropped.json      | facebook/bart-large | &check;   | &check;    | &cross;               | Predicted NERs with Dropped Spans |
| default_base_predicted_ner.json         | facebook/bart-base  | &check;   | &check;    | &cross;               | Predicted NERs                    |
| default_base_predicted_ner_dropped.json | facebook/bart-base  | &check;   | &check;    | &cross;               | Predicted NERs with Dropped Spans |