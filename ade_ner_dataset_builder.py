# =============================
# ADE NER Pipeline: Gold + Weak (SpaCy offsets, quick test)
# =============================
import pandas as pd
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import spacy
from collections import Counter

# ---------- Paths ----------
BASE_DIR = Path(__file__).parent
outputs_dir = BASE_DIR / "outputs"
outputs_dir.mkdir(exist_ok=True)

gold_csv_file = BASE_DIR / "gold_parsed.csv"
weak_csv_file = BASE_DIR / "weak_labeled_dataset.csv"

# ---------- Load SpaCy tokenizer ----------
nlp = spacy.blank("en")  # tokenizer only

# ---------- Load gold + weak data ----------
gold_df = pd.read_csv(gold_csv_file, low_memory=False)
weak_df = pd.read_csv(weak_csv_file, low_memory=False)

# unify text column name for weak dataset
text_col = "SYMPTOM_TEXT" if "SYMPTOM_TEXT" in weak_df.columns else "text"
weak_df.rename(columns={text_col: "text"}, inplace=True)

# ---------- Ensure weak dataset uses 'weak_labels' column ----------
if "weak_labels" not in weak_df.columns:
    weak_df["annotations"] = [[] for _ in range(len(weak_df))]
else:
    # Convert string JSON to Python list
    def parse_weak_labels(x):
        if pd.isna(x):
            return []
        try:
            return json.loads(x.replace("'", '"'))
        except:
            return []
    weak_df["annotations"] = weak_df["weak_labels"].apply(parse_weak_labels)

# ---------- Optionally limit data for quick test ----------
LIMIT = None  # set None to process full dataset
if LIMIT:
    gold_df = gold_df.head(LIMIT)
    weak_df = weak_df.head(LIMIT)

# ---------- Combine gold + weak ----------
combined_df = pd.concat([gold_df, weak_df], ignore_index=True)

# ---------- Helper: Convert char spans to token-level BIO ----------
def char_spans_to_bio(text, spans):
    if pd.isna(text) or not isinstance(text, str):
        return [], []
    if spans is None or (isinstance(spans, float) and pd.isna(spans)):
        spans = []

    doc = nlp(text)
    tokens = [token.text for token in doc]
    tags = ["O"] * len(tokens)

    for ent in spans:
        if not isinstance(ent, dict):
            continue
        start = ent.get("start")
        end = ent.get("end")
        label = ent.get("label")
        if start is None or end is None or label is None:
            continue

        # assign B-/I- tags to tokens
        for i, token in enumerate(doc):
            token_start = token.idx
            token_end = token.idx + len(token)
            if token_end <= start or token_start >= end:
                continue
            if token_start == start:
                tags[i] = "B-" + label
            else:
                tags[i] = "I-" + label

    # Debug: print first entity found
    if any(tag.startswith("B-") for tag in tags):
        print("Sample entity labeling:")
        for t, tag in zip(tokens, tags):
            if tag != "O":
                print(f"{t}: {tag}")
        print("----")

    return tokens, tags

# ---------- Apply conversion ----------
combined_df[["tokens", "ner_tags"]] = combined_df.apply(
    lambda row: pd.Series(char_spans_to_bio(row["text"], row["annotations"])),
    axis=1
)

# drop rows with empty tokens
combined_df = combined_df[combined_df["tokens"].str.len() > 0]

# ---------- Check tag distribution ----------
tag_counter = Counter(tag for tags in combined_df["ner_tags"] for tag in tags)
print("Tag distribution:", tag_counter)

# ---------- Split into train/validation ----------
train_df, val_df = train_test_split(combined_df, test_size=0.2, random_state=42)

# ---------- Build records ----------
train_records = [{"tokens": t, "ner_tags": tags} for t, tags in zip(train_df["tokens"], train_df["ner_tags"])]
val_records = [{"tokens": t, "ner_tags": tags} for t, tags in zip(val_df["tokens"], val_df["ner_tags"])]

# ---------- Save JSON ----------
with open(outputs_dir / "train_ner.json", "w", encoding="utf-8") as f:
    json.dump(train_records, f, ensure_ascii=False, indent=2)

with open(outputs_dir / "val_ner.json", "w", encoding="utf-8") as f:
    json.dump(val_records, f, ensure_ascii=False, indent=2)

print(f"✅ Prepared {len(train_records)} train and {len(val_records)} val examples.")
