from utils import prepare_data_and_metric, load_data, accuracy
from performer_model import PerformerLM
from linformer_lm import LinformerLM
from transformers import (
    LongformerTokenizer,
    LongformerForSequenceClassification,
    BigBirdTokenizer,
    BigBirdForSequenceClassification,
    TrainingArguments,
    Trainer
)
import numpy as np
import time
import torch
PATH = "C:\\Users\\trowb\\OneDrive\\Documents\\IMDB Dataset.csv\\IMDB Dataset.csv"

def compute_metrics(eval_pred, metric):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

def train(epochs=1, batch_size=128, depth = 1, lr=0.01, model_type = 'performer', test_num = 1):
    # Set hyperparameters from the parser
    if model_type == "performer":
        tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        model = PerformerLM(num_tokens=50265 + 7, max_seq_len=4096, dim=64, depth=int(depth), heads=4)
    elif model_type == "linformer":
        tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        model = LinformerLM(num_tokens=50265 + 7, dim=64, seq_len=4096, depth=int(depth), heads=4)
    elif model_type == "longformer":
        tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096')
    elif model_type == "bigbird":
        tokenizer = BigBirdTokenizer.from_pretrained('google/bigbird-roberta-base')
        model = BigBirdForSequenceClassification.from_pretrained('google/bigbird-roberta-base')
    # Set up the cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Set up loss function and optimizer
    if model_type == "longformer" or model_type == "bigbird":
        encoded_dataset, metric = prepare_data_and_metric(tokenizer)
        train_args = TrainingArguments(
            f"{model_type}-{test_num}",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model='accuracy',
        )
        trainer = Trainer(
            model,
            train_args,
            train_dataset=encoded_dataset["train"],
            eval_dataset=encoded_dataset['test'],
            tokenizer=tokenizer,
            compute_metrics=lambda x: compute_metrics(x, metric)
        )
        s = time.time()
        #trainer.train()
        print(trainer.evaluate())
        print("TIME:", time.time() - s)
    else:
        loss_func = torch.nn.CrossEntropyLoss()
        optim = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay = 0.0005)

        # Set up training data and validation data
        data_train, data_val = load_data(PATH, tokenizer, 4096, batch_size=batch_size)

        # Set up loggers
        global_step = 0

        # Wrap in a progress bar.
        accuracy_vals = list()
        loss_vals = list()
        val_acc_vals = list()
        s = time.time()
        for epoch in range(epochs):
            # Set the model to training mode.
            model.train()

            train_accuracy_val = list()
            train_loss_val = list()
            for x, attention_mask, y in data_train:
                x = x.to(device)
                y = y.to(device)

                y_pred = model(x)
                train_accuracy_val.append(accuracy(y_pred, y))

                # Compute loss and update model weights.
                loss = loss_func(y_pred, y)
                train_loss_val.append(loss)

                loss.backward()
                optim.step()
                optim.zero_grad()

                # Add loss to TensorBoard.
                global_step += 1

            train_accuracy_total = torch.FloatTensor(train_accuracy_val).mean().item()
            train_loss_total = torch.FloatTensor(train_loss_val).mean().item()
            print(train_accuracy_total)
            accuracy_vals.append(train_accuracy_total)
            loss_vals.append(train_loss_total)


            # Set the model to eval mode and compute accuracy.
            # No need to change this, but feel free to implement additional logging.
            model.eval()

            accuracys_val = list()
            for x, _, y in data_val:
                x = x.to(device)
                y = y.to(device)
                y_pred = model(x)
                accuracys_val.append(accuracy(y_pred, y))

            accuracy_total = torch.FloatTensor(accuracys_val).mean().item()
            val_acc_vals.append(accuracy_total)
            print(accuracy_total)
        print(accuracy_vals)
        print(loss_vals)
        print(val_acc_vals)
        print("TIME:", time.time() - s)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-e', '--epoch', type=int, default=10)
    parser.add_argument('-b', '--batchsize', type=int, default=64)
    parser.add_argument('-l', '--lr', type=float, default=1e-3)
    parser.add_argument('-m', '--model', type=str, default='performer')
    parser.add_argument('-n', '--num', type=int, default=1)

    args = parser.parse_args()
    train(args)