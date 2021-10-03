# coding-template

## Summary

The summary can contain but is not limited to:

- Code structure.

- Commands to reproduce your experiments.

- Write-up of your findings and conclusions.

- Ipython notebooks can be organized in `notebooks`.

We explored the Longformer, Linformer, Bigbird, and Performer architectures on the IMDB, SST2, QNLI, and QQP datasets.

## Reference

We used the following code references from huggingface and other sources to train and use pretrained models:

https://huggingface.co/transformers/v2.5.0/examples.html

https://github.com/huggingface/transformers/blob/v4.6.0-release/examples/pytorch/text-classification/run_glue.py

https://huggingface.co/google/bigbird-roberta-base

https://huggingface.co/allenai/longformer-base-4096

https://pythonrepo.com/repo/lucidrains-performer-pytorch-python-pytorch-utilities

https://pypi.org/project/linformer-pytorch/

## IMDB

## SST2

### Model and Dataset

We explored the SST2 dataset (Stanfored Sentiment Treebank). This is a binary sentiment analysis classification task that contains phrases with fine-grained sentiment labels in parse trees from movie reviews. It is similar to the previous SST1 dataset, but has neutral reviews removed. Some of the leading scores on this datset include XLNet-Large (96.8%), MT-DNN-ensemble (96.5%), and Snorkel MeTaL (96.2%). 

I explored using the following models on the dataset:

* Bigbird
* Longformer
* Performer
* Linformer

I was able to fine-tune Bigbird and Longformer for the task, but do to resource and time constraints, I was unable to fine-tune Performer or Linformer.

### Code

The code exists in the sst2.ipynb notebook file. I tried two approaches to fine-tune the models. For the models that had cards in huggingface (Bigbird and Longformer), I was able to pull the ptretrained model and tokenizer from online. For the other two models, I used the imported libraries to define and instantiate the models. I attempted to created a Trainer object and train and evaluate on the data. However, this led to issues with memory in Google colab. To overcome this, I used the run_glue.py script provided by huggingface to train their models. Since SST2 is a part of the GLUE benchmark, I was able to train and evaluate on it using this script. This script takes in the model name, task name, whether to train and evaluate, and values such as the learning rate, batch size, and number of epochs as parameters. I had set the batch size to 32, learning rate to 2e-5, and number of epochs to 3. Unfortunately, due to limited compute resources through colab, I was unable to experiment with different variations of the batch size, learning rate, or number of epochs. 

### Performance

### Bigbird

*** train metrics ***

Runtime - 48:27 \
Loss - 0.1315

*** eval metrics ***

Runtime - 4.59 seconds \
Accuracy - 95.41%

### Longformer

*** train metrics ***

Runtime - 42:36 \
Loss - 0.1286

*** eval metrics ***

Runtime - 4.25 seconds \
Accuracy - 96.18%

### Performer

*** eval metrics ***

Runtime - 4.45 seconds \
Accuracy - 52%

### Linformer

*** eval metrics ***

Runtime - 4.56 seconds \
Accuracy - 51%

### Analysis

As mentioned above, I was only able to iterate over one set of parameters for training and was unable to fine-tune Performer and Linformer. As this was a binary classification task (either positive or negative sentiment), a model that predicts one class for every input would have an accuracy of around 50% for an evenly distributed dataset. When examining the output from the Performer and Linformer models following softmax, the probability over both classes were hovering around 50%. I often saw an output of [.49, .51]. To me, this meant that the models were unsure over every evaluation input and that the pretrained model performed no better than than a default model that predicted the same class on any input. However, the evaluation runtime was comparable to the other models that were fine-tuned to the task.

When looking at the other two models (Bigbird and Longformer), I was able to successfully train them using the provided script from huggingface. The training times were comparable for fine-tuning and were not too expensive when training on Google Colab Pro using a GPU. They were able to train in under an hour. I first tried using the models in a similar fashion to Performer and Linformer, without fine-tuning, and also recieved accuracies around 50%. This shows me that the values I received for Performer and Linformer were not abnormal. However, after fine-tuning, I received very good results of 95% and above for both Bigbird and Longformer. Longformer performed slightly better at 96.18%, but I am interested in seeing how these differences would change based on experiments on the hyperparameters, namely learning rate. I am unable to provide a suggestion over these two models for this task as both models were designed to perform better on documents of greater length. The examples that I examined in the SST2 dataset were relatively short in length, sometimes being less than a complete sentence. 

## QNLI

## QQP
