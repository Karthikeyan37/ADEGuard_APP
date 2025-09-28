# train_biobert_ner_smallcheck.py
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    pipeline
)

# 1. Load the train/val data
dataset = load_dataset(
    "json",
    data_files={
        "train": "train_ner.json",
        "validation": "val_ner.json"
    }
)

# 2. Define full label list explicitly
label_list = ["O", "B-ADE", "I-ADE", "B-DRUG", "I-DRUG", "B-SEVERITY", "I-SEVERITY"]
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}

print("Labels:", label_list)

# 3. Load model & tokenizer
model_ckpt = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

model = AutoModelForTokenClassification.from_pretrained(
    model_ckpt,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

# 4. Tokenize and align labels
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=256   # use smaller max length for faster test
    )

    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # special tokens
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[labels[word_idx]])
            else:
                # For subwords, convert B- to I-
                if labels[word_idx].startswith("B-"):
                    label_ids.append(label2id["I-" + labels[word_idx][2:]])
                else:
                    label_ids.append(label2id[labels[word_idx]])
            previous_word_idx = word_idx
        all_labels.append(label_ids)

    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)
data_collator = DataCollatorForTokenClassification(tokenizer)

# 5. Take a small subset for quick overfit test


small_train = tokenized_dataset["train"]
small_val = tokenized_dataset["validation"]


# 6. Training args with a realistic learning rate
# training_args = TrainingArguments(
#     output_dir="./biobert_ade_ner_small",
#     evaluation_strategy="steps",
#     eval_steps=50,
#     save_strategy="epoch",
#     logging_strategy="steps",
#     logging_steps=10,
#     learning_rate=2e-5,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     num_train_epochs=3,
#     weight_decay=0.01,
#     load_best_model_at_end=True,
#     metric_for_best_model="eval_loss"
# )

training_args = TrainingArguments(
    output_dir="./biobert_ade_ner_large",
    do_eval=True,        # old style instead of evaluation_strategy
    logging_steps=10,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)


# 7. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train,
    eval_dataset=small_val,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# 8. Train
trainer.train()
trainer.evaluate()


# 9. Evaluate
metrics = trainer.evaluate()
print("Validation metrics:", metrics)

# 10. Save the fine-tuned model
trainer.save_model("./biobert_ade_ner_large")

# 11. Reload fine-tuned model for inference
from transformers import AutoModelForTokenClassification
fine_tuned_model = AutoModelForTokenClassification.from_pretrained(
    "./biobert_ade_ner_small"
)

ner_pipeline = pipeline(
    "token-classification",
    model=fine_tuned_model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"
)

# 12. Test on an example
example = "SORENESS IN THE AREA.  ITCHING AND RASH"
outputs = ner_pipeline(example)
print(outputs)
