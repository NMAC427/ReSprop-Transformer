from transformers import AutoTokenizer, BertForSequenceClassification
from transformers import TrainingArguments, Trainer, TrainerCallback
from datasets import load_dataset
import numpy as np
import evaluate
from resprop_linear import resprofify_bert


# Load Model
model_name = "prajjwal1/bert-tiny"

tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label={
        0: "Negative",
        1: "Positive"
    }
)

# Load Data
def tokenize_function(examples):
    return tokenizer(examples["text"], max_length=128, padding="max_length", truncation=True)

dataset = load_dataset("fancyzhx/yelp_polarity")
tokenized_datasets = dataset.map(tokenize_function, batched=True)

train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(512))
eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(512))


# Accuracy
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


class AccuracyLoggerCallback(TrainerCallback):
    def __init__(self):
        self.results = {}

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            epoch = int(state.epoch)
            self.results[epoch] = metrics.get("eval_accuracy")


evaluation_results = {}
for reuse_percentage in [0.0, 0.5, 0.7, 0.9, 0.95, 0.99]:
    model = resprofify_bert(base_model, reuse_percentage=reuse_percentage)
    accuracy_logger = AccuracyLoggerCallback()

    training_args = TrainingArguments(
        output_dir="trainer_out",
        eval_strategy="epoch",
        num_train_epochs=32,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[accuracy_logger],
    )

    trainer.train()

    evaluation_results[reuse_percentage] = accuracy_logger.results

print()
print("----------------")
print()
print(evaluation_results)