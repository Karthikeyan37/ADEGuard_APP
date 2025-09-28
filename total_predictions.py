from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np

# -------------------------------
# Load fine-tuned NER model
# -------------------------------
model_ckpt = "./biobert_ade_ner_large"  # folder with fine-tuned weights
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForTokenClassification.from_pretrained(model_ckpt)

ner_pipeline = pipeline(
    "token-classification",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"
)

# -------------------------------
# Semantic ADE detection setup
# -------------------------------
ADE_PHRASES = ["headache", "fever", "chills", "muscle pain", "rash", "soreness", "itching"]
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
ade_embeddings = sbert_model.encode(ADE_PHRASES)

def semantic_ade_detection(text, ner_results, threshold=0.7):
    """
    Detect ADEs semantically that might have been missed by NER.
    """
    tokens = text.split()
    detected = set([r['word'].lower() for r in ner_results if r['entity_group'] == "ADE"])
    missed = []

    for token in tokens:
        token_lower = token.lower().strip(".,")  # remove punctuation
        if token_lower in detected:
            continue
        token_emb = sbert_model.encode(token_lower)
        sims = np.dot(ade_embeddings, token_emb) / (np.linalg.norm(ade_embeddings, axis=1) * np.linalg.norm(token_emb))
        if np.max(sims) > threshold:
            start_idx = text.lower().find(token_lower)
            missed.append({
                "entity_group": "ADE",
                "word": token.strip(".,"),
                "start": start_idx,
                "end": start_idx + len(token.strip(".,") ),
                "score": float(np.max(sims))
            })
    return missed

# -------------------------------
# Merge overlapping predictions
# -------------------------------
def merge_predictions(preds):
    """
    Merge overlapping/duplicate ADE spans.
    Keeps the highest score.
    """
    preds_sorted = sorted(preds, key=lambda x: x['start'])
    merged = []
    seen_spans = set()
    
    for p in preds_sorted:
        span = (p['start'], p['end'])
        if span not in seen_spans:
            merged.append(p)
            seen_spans.add(span)
        else:
            # If duplicate span exists, keep the one with higher score
            for idx, existing in enumerate(merged):
                if (existing['start'], existing['end']) == span and p['score'] > existing['score']:
                    merged[idx] = p
                    break
    return merged

# -------------------------------
# Example text
# -------------------------------
example_text = "SORENESS IN THE AREA. ITCHING AND RASH, headache, fever 101, chills, and aching muscles."

# Run NER
ner_preds = ner_pipeline(example_text)

# Semantic ADE detection
semantic_preds = semantic_ade_detection(example_text, ner_preds)

# Combine and merge
all_preds = merge_predictions(ner_preds + semantic_preds)

# Output final predictions
for p in all_preds:
    print(p)
