---
language:
- en
license: apache-2.0
tags:
- generated_from_trainer
datasets:
- glue
metrics:
- accuracy
model-index:
- name: qnli-big-bird
  results:
  - task:
      name: Text Classification
      type: text-classification
    dataset:
      name: GLUE QNLI
      type: glue
      args: qnli
    metrics:
    - name: Accuracy
      type: accuracy
      value: 0.9040820062236866
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# qnli-big-bird

This model is a fine-tuned version of [google/bigbird-roberta-base](https://huggingface.co/google/bigbird-roberta-base) on the GLUE QNLI dataset.
It achieves the following results on the evaluation set:
- Loss: 0.2706
- Accuracy: 0.9041

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 64
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3.0

### Training results



### Framework versions

- Transformers 4.12.0.dev0
- Pytorch 1.9.0+cu102
- Datasets 1.12.1
- Tokenizers 0.10.3
