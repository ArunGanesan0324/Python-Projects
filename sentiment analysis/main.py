import os
from datasets import load_dataset, concatenate_datasets, Value, load_from_disk
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
import numpy as np
import evaluate

# --- Configuration: Reduce the dataset size for faster processing ---
# We'll use a smaller, manageable subset for quick iteration.
MAX_TRAIN_SAMPLES = 5000  # Reduced from 10000 for even faster setup
MAX_TEST_SAMPLES = 1000   # Reduced from 2000 for even faster setup

# --- Define paths for our SMALL, processed datasets ---
# The script will save the processed data here to avoid re-doing work.
subset_suffix = f"_train{MAX_TRAIN_SAMPLES}_test{MAX_TEST_SAMPLES}"
processed_train_path = f"./processed_data{subset_suffix}/train"
processed_test_path = f"./processed_data{subset_suffix}/test"


# --- Main Logic: Load from disk if available, otherwise process and save ---
if os.path.exists(processed_train_path) and os.path.exists(processed_test_path):
    print(f"Found pre-processed subset. Loading from '{processed_train_path}'...")
    train_data = load_from_disk(processed_train_path)
    test_data = load_from_disk(processed_test_path)
    print("Subset loaded successfully.")

else:
    print("Pre-processed subset not found. Starting setup for a smaller dataset...")
    # --- 1. Load Datasets ---
    print("Loading datasets from Hugging Face Hub...")
    imdb = load_dataset("imdb")
    yelp = load_dataset("yelp_polarity")
    amazon = load_dataset("amazon_polarity")

    # --- 2. Create Subsets ---
    print(f"Creating smaller subsets with {MAX_TRAIN_SAMPLES} train and {MAX_TEST_SAMPLES} test samples each.")
    imdb_train_subset = imdb["train"].select(range(MAX_TRAIN_SAMPLES))
    imdb_test_subset = imdb["test"].select(range(MAX_TEST_SAMPLES))
    
    yelp_train_subset = yelp["train"].select(range(MAX_TRAIN_SAMPLES))
    yelp_test_subset = yelp["test"].select(range(MAX_TEST_SAMPLES))

    amazon_train_subset = amazon["train"].select(range(MAX_TRAIN_SAMPLES))
    amazon_test_subset = amazon["test"].select(range(MAX_TEST_SAMPLES))

    # --- 3. Standardize and Combine Subsets ---
    print("Standardizing and combining the subsets...")
    imdb_train_subset = imdb_train_subset.cast_column("label", Value("int64"))
    imdb_test_subset = imdb_test_subset.cast_column("label", Value("int64"))
    
    yelp_train_subset = yelp_train_subset.cast_column("label", Value("int64"))
    yelp_test_subset = yelp_test_subset.cast_column("label", Value("int64"))

    # Corrected this line from the previous version
    amazon_train_subset = amazon_train_subset.cast_column("label", Value("int64"))
    amazon_test_subset = amazon_test_subset.cast_column("label", Value("int64"))

    train_data = concatenate_datasets([imdb_train_subset, yelp_train_subset, amazon_train_subset]).shuffle(seed=42)
    test_data = concatenate_datasets([imdb_test_subset, yelp_test_subset, amazon_test_subset]).shuffle(seed=42)

    # --- 4. Clean, Tokenize, and Format ---
    print("Cleaning and tokenizing the combined subset...")
    train_data = train_data.filter(lambda x: x['text'] is not None and len(x['text']) > 0)
    test_data = test_data.filter(lambda x: x['text'] is not None and len(x['text']) > 0)

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True)
    
    train_data = train_data.map(tokenize, batched=True)
    test_data = test_data.map(tokenize, batched=True)
    
    train_data = train_data.rename_column("label", "labels")
    test_data = test_data.rename_column("label", "labels")
    train_data.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_data.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    # --- 5. Save the Processed Subset ---
    print(f"Saving processed subset to '{processed_train_path}'...")
    train_data.save_to_disk(processed_train_path)
    test_data.save_to_disk(processed_test_path)
    print("Subset saved successfully for future runs.")


# --- Training Phase ---
print("\n--- Starting Training Phase ---")
print(f"Training on {len(train_data)} examples, evaluating on {len(test_data)} examples.")

model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1_score = f1.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    return {"accuracy": acc, "f1": f1_score}

# Use 'eval_strategy' for older versions and 'report_to="none"' to fix the last error
training_args = TrainingArguments(
    output_dir="./roberta-sentiment-multi",
    eval_strategy="epoch",            # Use this for older versions of transformers
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,   # You can increase this if you have a GPU with more memory
    per_device_eval_batch_size=32,
    num_train_epochs=1,               # Reduced from 3 to 1 for a much faster training cycle
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none",                 # THE FIX for the tensorflow.io error
                           # Enable mixed precision for a significant speed boost on compatible GPUs
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    compute_metrics=compute_metrics,
)

print("Starting or resuming training...")

# --- THE FIX for the checkpoint error ---
# Check if a checkpoint exists in the output directory
last_checkpoint = get_last_checkpoint(training_args.output_dir)
if last_checkpoint:
    print(f"Checkpoint found at {last_checkpoint}. Resuming training.")

# If a checkpoint is found, resume training from it. Otherwise, start fresh.
# The 'resume_from_checkpoint' argument can be a path (str) or a boolean.
trainer.train(resume_from_checkpoint=last_checkpoint)
print("Training complete.")

trainer.save_model("./roberta-sentiment-multi-final")
print("Final model saved to ./roberta-sentiment-multi-final")

