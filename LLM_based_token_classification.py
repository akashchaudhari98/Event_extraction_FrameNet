import pandas as pd
import numpy as np
import evaluate
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification
from transformers import TrainingArguments
from transformers import Trainer
from transformers import DataCollatorForTokenClassification


class LLMTC:
    def __init__(self,
                data_path,
                data_label,
                label_path,
                model_name = "bert-base-cased"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        self.data_path = data_path
        self.data_label = label_path

    def labels_data(self):
        label = pd.read_csv(self.data_label)
        label = label[["0","1"]]
        self.label2id = {label['0']:label['1'] for i,label in label.iterrows()}
        self.id2label = {label['1']:label['0'] for i,label in label.iterrows()}

    def load_data_maven(self):
        data = pd.read_csv(self.data_path)
        data = data[["BIO_tags","token"]]
        data['BIO_tags'] = data['BIO_tags'].apply(eval)
        data['token'] = data['token'].apply(eval)
        data = data.dropna()
        self.dataset = Dataset.from_pandas(data)

    def align_labels_with_tokens(self,labels, word_ids):
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id != current_word:
                # Start of a new word!
                current_word = word_id
                label = -100 if word_id is None else labels[word_id]
                new_labels.append(label)
            elif word_id is None:
                # Special token
                new_labels.append(-100)
            else:
                # Same word as previous token
                label = labels[word_id]
                # If the label is B-XXX we change it to I-XXX
                if label % 2 == 1:
                    label += 1
                new_labels.append(label)

        return new_labels

    def tokenize_and_align_labels(self,examples):
        tokenized_inputs = self.tokenizer(
            examples["token"], truncation=True, is_split_into_words=True
        )
        all_labels = examples["BIO_tags"]
        new_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(self.align_labels_with_tokens(labels, word_ids))
        tokenized_inputs["labels"] = new_labels
        return tokenized_inputs
    
    def preprocess_data(self):
        self.tokenized_datasets = self.dataset.map(
            self.tokenize_and_align_labels,
            batched=True,
            remove_columns= self.dataset.column_names,
        )
    
    def compute_metrics(self,eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        #print("label_names:",label_names)
        # Remove ignored index (special tokens) and convert to labels
        true_labels = [[self.id2label[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [self.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": all_metrics["overall_precision"],
            "recall": all_metrics["overall_recall"],
            "f1": all_metrics["overall_f1"],
            "accuracy": all_metrics["overall_accuracy"],
        }
    
    def model(self):
        self.model = AutoModelForTokenClassification.from_pretrained(
                self.model_name,
                id2label=self.id2label,
                label2id=self.label2id,
                from_tf=True
            )
    
    def split_data(self):
        self.tokenized_datasets = self.tokenized_datasets.train_test_split(test_size=0.1)  
    
    def train_model(self):
        args = TrainingArguments(
            "bert-finetuned-ner",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            num_train_epochs=3,
            weight_decay=0.01,
            per_device_train_batch_size = 4
        )
        trainer = Trainer(
        model=self.model,
        args=args,
        train_dataset=self.tokenized_datasets["train"],
        eval_dataset=self.tokenized_datasets["test"],
        data_collator=self.data_collator,
        compute_metrics=self.compute_metrics,
        tokenizer=self.tokenizer,
        )
        trainer.train()
        trainer.save_model("model_BERT_maven")

         