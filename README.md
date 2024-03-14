# Annotated-Semantic-Predications-from-SemMedDB
I went throught the pre-processing, fine-tuning process in `Interview task.ipynb`. Then develop the `task.py` for further multiple experiments.

## Instructions
  use `run.sh` to run multiple task at the same time. If more then one GPU available, you could modify the GPU index to achieve parallel computing (e.g. in MSI).
  
  Use `task.py` to run a single experiment

  Every new results are added in the end of `result.txt` file

## Single experiment examples
  - Fine-tune `Bert-base-uncased` model for `10` epochs with Training-Validation-Test split seed `3`
      ```ruby
      python task.py -model bert-base-uncased -epoch 10 -random 3
      ```

  - Fine-tune `roberta-base` model for `20` epochs with Training-Validation-Test split seed `50`
      ```ruby
      python task.py -model roberta-base -epoch 20 -random 50
      ```

Please note that currently only four models can be trained and evaluated: `BERT-base-uncased`, `DistilBERT-base-uncased`, `RoBERTa-
base`, and `BERT-large-uncased`.
  
