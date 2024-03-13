# Annotated-Semantic-Predications-from-SemMedDB

## Instructions
  use `run.sh` to run multiple task at the same time. If more then one GPU available, could modify it to achieve parallel computing.
  
  Use `task.py` ro run a single experiment

  Results are saved in the end of `result.txt` file

## Single experiment examples
  - Fine-tune Bert-base-uncased model for 10 epoch with Training-Validation-Test split seed 3
      ```ruby
      python task.py -model bert-base-uncased -epoch 10 -random 3
      ```

  - Fine-tune roberta-base model for 20 epoch with Training-Validation-Test split seed 50
      ```ruby
      python task.py -model roberta-base -epoch 20 -random 50
      ```
  
