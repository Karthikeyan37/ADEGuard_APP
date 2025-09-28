# from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# class ADEGuardNER:
#     def __init__(self, model_dir):
#         self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
#         self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
#         self.ner_pipeline = pipeline(
#             "token-classification",
#             model=self.model,
#             tokenizer=self.tokenizer,
#             aggregation_strategy="simple"
#         )

#     def predict(self, text):
#         results = self.ner_pipeline(text)
#         return [
#             {
#                 "entity": r["entity_group"],
#                 "text": r["word"],
#                 "start": r["start"],
#                 "end": r["end"],
#                 "score": r["score"]
#             } 
#             for r in results
#         ]

# # Example
# ner_model = ADEGuardNER("./biobert_ade_ner_large")
# example_text = "severe headache ,fever after flu shot."
# predictions = ner_model.predict(example_text)
# print(predictions)





from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import re

class ADEGuardNER:
    def __init__(self, model_dir):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.ner_pipeline = pipeline(
            "token-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple"
        )
        
        # Severity-related mappings
        self.MODIFIERS = {"mild": 1, "moderate": 2, "severe": 3, "critical": 4, "extreme": 4, "very severe": 4}
        self.INTENSIFIERS = {"very": 1, "extremely": 2, "intense": 1}
        
    def predict(self, text):
        results = self.ner_pipeline(text)
        ner_results = [
            {
                "entity": r["entity_group"],
                "text": r["word"],
                "start": r["start"],
                "end": r["end"],
                "score": r["score"]
            } 
            for r in results
        ]
        return self.assign_severity(text, ner_results)
    
    def fever_severity(self, temp):
        if temp < 100.4:
            return 1
        elif temp < 102:
            return 2
        else:
            return 3

    def assign_severity(self, text, ner_results, window=5):
        tokens = text.split()
        results = []

        for r in ner_results:
            ade_word = r["text"]
            token_idx = next((i for i, t in enumerate(tokens) if ade_word.lower() in t.lower()), None)
            severity_score = 1  # default low

            if token_idx is not None:
                context_tokens = tokens[max(0, token_idx-window): min(len(tokens), token_idx+window+1)]
                context_str = " ".join(context_tokens).lower()

                # Check modifier keywords
                for mod in self.MODIFIERS:
                    if mod in context_str:
                        severity_score = max(severity_score, self.MODIFIERS[mod])

                # Check intensifiers
                for intf in self.INTENSIFIERS:
                    if intf in context_str:
                        severity_score = min(4, severity_score + self.INTENSIFIERS[intf])

                # Check numeric values for fever
                if "fever" in ade_word.lower():
                    match = re.search(r'(\d{2,3})', context_str)
                    if match:
                        temp = float(match.group(1))
                        severity_score = max(severity_score, self.fever_severity(temp))

            # Map numeric score to label
            if severity_score <= 1:
                severity_label = "low"
            elif severity_score == 2:
                severity_label = "medium"
            else:
                severity_label = "high"

            results.append({**r, "severity": severity_label})

        return results

# Example usage
ner_model = ADEGuardNER("./biobert_ade_ner_large")
example_text = "Patient had very severe headache, fever 105, and chills."
predictions = ner_model.predict(example_text)
for p in predictions:
    print(p)
