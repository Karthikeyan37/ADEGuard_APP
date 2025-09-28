#app.py
import os, re, torch
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import hdbscan  # pip install hdbscan
from collections import Counter
import shap
# -----------------------------
# 1. Page Setup
# -----------------------------
st.set_page_config(page_title="ADEGuard", layout="wide")
logo_col, title_col = st.columns([1,6])
with logo_col:
    st.image("CODEbasics_logo.svg", width=80)
with title_col:
    st.title("ADEGuard – AI-Powered ADE & DRUG Detection")

MODEL_PATH = "./biobert_ade_ner_large"

# -----------------------------
# 2. Model Loading (cached)
# -----------------------------
@st.cache_resource
def load_ner_model(path):
    tok = AutoTokenizer.from_pretrained(path)
    mod = AutoModelForTokenClassification.from_pretrained(path)
    return pipeline("token-classification", model=mod, tokenizer=tok, aggregation_strategy="simple")

@st.cache_resource
def load_sbert_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

ner_pipeline = load_ner_model(MODEL_PATH)
sbert_model = load_sbert_model()

# -----------------------------
# 3. Load VAERS phrases
# -----------------------------
EMBEDDINGS_DIR = "./embeddings_cache"
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

try:
    df = pd.read_csv("weak_labeled_dataset.csv")
    sym_cols = [c for c in ["SYMPTOM1","SYMPTOM2","SYMPTOM3","SYMPTOM4","SYMPTOM5"] if c in df.columns]
    drug_cols = [c for c in ["VAX_NAME","VAX_TYPE","DRUG_NAME","PRODUCT"] if c in df.columns]
    ade_phrases = set(df[sym_cols].stack().dropna().str.lower().unique()) if sym_cols else set()
    drug_phrases = set(df[drug_cols].stack().dropna().str.lower().unique()) if drug_cols else set()
    age_map = dict(zip(df['SYMPTOM_TEXT'].str.lower(), df['AGE_YRS'])) if 'SYMPTOM_TEXT' in df and 'AGE_YRS' in df else {}
except Exception as e:
    st.warning(f"VAERS CSV load error: {e}")
    ade_phrases = {"headache", "fever"}
    drug_phrases = {"ibuprofen"}
    age_map = {}

@st.cache_resource
def get_phrase_embeddings(phrases):
    if not phrases:
        return None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return sbert_model.encode(list(phrases), device=device, convert_to_numpy=True, normalize_embeddings=True)

ade_embeddings = get_phrase_embeddings(ade_phrases)
drug_embeddings = get_phrase_embeddings(drug_phrases)

# -----------------------------
# 4. Modifiers
# -----------------------------
modifier_keywords = {
    "high": ["severe","very severe","extreme","intense","critical","life-threatening","hospitalized"],
    "medium": ["moderate","medium","significant","persistent"],
    "low": ["mild","slight","minor","light"]
}
all_modifiers = sum(modifier_keywords.values(), [])
modifier_emb = sbert_model.encode(all_modifiers, normalize_embeddings=True, convert_to_numpy=True)
modifier_labels = [lvl for lvl,phr in modifier_keywords.items() for _ in phr]

def semantic_severity(text_span):
    emb = sbert_model.encode([text_span], normalize_embeddings=True, convert_to_numpy=True)
    sims = np.dot(emb, modifier_emb.T)
    idx = sims.argmax()
    return modifier_labels[idx]

def severity_scoring(text, full_text=None, window=5):
    if full_text is None: full_text=text
    tokens = re.findall(r'\w+', full_text.lower())
    span_tokens = re.findall(r'\w+', text.lower())
    idx = None
    for i in range(len(tokens)-len(span_tokens)+1):
        if tokens[i:i+len(span_tokens)]==span_tokens: idx=i; break
    check = tokens[max(0,idx-window): idx+window] if idx is not None else tokens
    for t in check:
        if t in modifier_keywords["high"]: return "high"
    for t in check:
        if t in modifier_keywords["medium"]: return "medium"
    for t in check:
        if t in modifier_keywords["low"]: return "low"
    return "medium"

# -----------------------------
# 5. Semantic fallback
# -----------------------------
def semantic_detection(text, ner_results, embeddings, phrases, entity_type, threshold=0.7):
    detected_spans = [(r['start'], r['end']) for r in ner_results if r['entity']==entity_type]
    detected_texts = {r['text'].lower() for r in ner_results if r['entity']==entity_type}
    missed=[]
    if embeddings is None or not phrases: return missed
    tokens=text.split()
    token_emb=sbert_model.encode(tokens)
    for token,emb in zip(tokens,token_emb):
        if token.lower() in detected_texts: continue
        sims=np.dot(embeddings,emb)/(np.linalg.norm(embeddings,axis=1)*np.linalg.norm(emb))
        if np.max(sims)>threshold:
            for m in re.finditer(re.escape(token),text,flags=re.I):
                s,e=m.start(),m.end()
                overlap=any(s<=ss<s or s<ee<=e for ss,ee in detected_spans)
                if not overlap:
                    missed.append({"entity":entity_type,"text":text[s:e],"start":s,"end":e,"score":float(np.max(sims))})
                    detected_spans.append((s,e))
                    break
    return missed

# -----------------------------
# 6. Clustering
# -----------------------------
def cluster_entities(preds, age_group=None, cluster_method="hdbscan"):
    if len(preds)==0: return preds
    texts=[p['text'] for p in preds]
    modifiers=[2 if semantic_severity(t)=="high" else 1 if semantic_severity(t)=="medium" else 0 for t in texts]
    emb=sbert_model.encode(texts)
    X=np.hstack([emb, np.array(modifiers).reshape(-1,1)])
    if age_group:
        age_val={"Children":5,"Adult":35,"Senior":65}.get(age_group,35)
        X=np.hstack([X, np.array([age_val]*len(preds)).reshape(-1,1)/100.0])
    if len(X)>1:
        if cluster_method=="hdbscan":
            clusterer=hdbscan.HDBSCAN(min_cluster_size=2)
            labels=clusterer.fit_predict(X)
        else:
            clusterer=AgglomerativeClustering(n_clusters=None,distance_threshold=0.7)
            labels=clusterer.fit_predict(X)
    else:
        labels=[-1]*len(preds)
    for i,p in enumerate(preds):
        p['cluster']=int(labels[i])
        p['modifier']=severity_scoring(p['text'],full_text=example_text)
    return preds

# -----------------------------
# 7. Sidebar
# -----------------------------
st.sidebar.subheader("Detection Controls")
age_group=st.sidebar.selectbox("Select Age Group",["Children","Adult","Senior"])
modifier_level=st.sidebar.selectbox("Highlight Modifier Level",["Low","Medium","High"])
cluster_method=st.sidebar.selectbox("Clustering Method",["hdbscan","agglomerative"])

# -----------------------------
# 8. Input
# -----------------------------
example_text=st.text_area("Enter symptom text:","Severe headache, fever after flu shot.",height=150)

if st.button("Predict ADEs & DRUGs"):
    ner_results=ner_pipeline(example_text)
    preds=[{"entity":r['entity_group'],"text":r['word'],"start":r['start'],"end":r['end'],"score":r['score']} for r in ner_results]
    preds+=semantic_detection(example_text,preds,ade_embeddings,ade_phrases,"ADE")
    preds+=semantic_detection(example_text,preds,drug_embeddings,drug_phrases,"DRUG")
    preds=cluster_entities(preds,age_group=age_group,cluster_method=cluster_method)

    # tab1,tab2,tab3=st.tabs(["Predicted Entities","Clusters","Explainability"])

    # with tab1:
    #     st.subheader("Detected ADEs & DRUGs")
    #     highlighted=example_text
    #     for p in sorted(preds,key=lambda x:x['start'],reverse=True):
    #         color="#b3e6ff" if p['entity']=="ADE" else "#ffd480"
    #         if p['modifier'].lower()==modifier_level.lower(): color="#ff8080"
    #         highlighted=highlighted[:p['start']]+f"<mark style='background-color:{color}'>{p['text']}</mark>"+highlighted[p['end']:]
    #     st.markdown(highlighted,unsafe_allow_html=True)
    #     st.dataframe(pd.DataFrame(preds))

    

    # with tab2:
    #     st.subheader("Entity Clusters (age + modifier-aware)")
    #     if preds:
    #         cluster_df = pd.DataFrame(preds)

    #         # compute 2D embeddings for plotting (fast)
    #         texts = cluster_df['text'].tolist()
    #         embeddings_2d = sbert_model.encode(texts)

    #         # 🔹 make cluster names automatically from top words in each cluster
    #         from collections import Counter
    #         def make_cluster_labels(df, top_n_words=3):
    #             labels = {}
    #             for cluster_id in df['cluster'].unique():
    #                 texts = df[df['cluster'] == cluster_id]['text']
    #                 words = " ".join(texts).lower().split()
    #                 common = [w for w, _ in Counter(words).most_common(top_n_words)]
    #                 labels[cluster_id] = f"Cluster {cluster_id} – {', '.join(common)}"
    #             return labels

    #         cluster_labels = make_cluster_labels(cluster_df)
    #         cluster_df['cluster_name'] = cluster_df['cluster'].map(cluster_labels)

    #         # 🔹 marker size & symbol reflect severity/modifier
    #         size_map = {"low": 6, "medium": 10, "high": 14}
    #         cluster_df['marker_size'] = cluster_df['modifier'].map(size_map)
    #         # optional: symbol encodes severity
    #         cluster_df['symbol'] = cluster_df['modifier']

    #         # 🔹 consistent colour palette for clusters
    #         palette = px.colors.qualitative.Safe
    #         unique_names = sorted(cluster_df['cluster_name'].unique())
    #         color_discrete_map = {name: palette[i % len(palette)] for i, name in enumerate(unique_names)}

    #         fig = px.scatter(
    #             x=embeddings_2d[:, 0],
    #             y=embeddings_2d[:, 1],
    #             color=cluster_df['cluster_name'],  # dynamic legend names
    #             size=cluster_df['marker_size'],    # severity as size
    #             symbol=cluster_df['symbol'],       # severity as symbol
    #             hover_data={
    #                 "Entity": cluster_df['entity'],
    #                 "Text": cluster_df['text'],
    #                 "Severity": cluster_df['modifier']
    #             },
    #             color_discrete_map=color_discrete_map,
    #             title="Entity Clusters (age + modifier-aware)"
    #         )
    #         fig.update_layout(legend_title_text="Detected Clusters")
    #         st.plotly_chart(fig, use_container_width=True)
    #     else:
    #         st.write("No entities detected.")
    
    # with tab3:
    #     st.subheader("Explainability (placeholder)")
    #     tokens=example_text.split()
    #     shap_values=np.random.rand(len(tokens))
    #     st.bar_chart(shap_values)
    # st.success("Prediction complete!")

# -----------------------------
# 9. Tabs
# -----------------------------
    tab1, tab2, tab3 = st.tabs(["Predicted Entities", "Clusters", "Explainability"])

    with tab1:
        st.subheader("Detected ADEs & DRUGs")
        highlighted = example_text
        for p in sorted(preds, key=lambda x: x['start'], reverse=True):
            color = "#b3e6ff" if p['entity'] == "ADE" else "#ffd480"
            if p['modifier'].lower() == modifier_level.lower(): color = "#ff8080"
            highlighted = highlighted[:p['start']] + f"<mark style='background-color:{color}'>{p['text']}</mark>" + highlighted[p['end']:]
        st.markdown(highlighted, unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(preds))

    # -----------------------------
    # 10. Cluster Visualization
    # -----------------------------
    with tab2:
        st.subheader("Entity Clusters (modifier-aware & explainable)")
        if preds:
            cluster_df = pd.DataFrame(preds)

            # 🔹 Cluster labels with descriptive top words & entity counts
            def make_cluster_labels(df, top_n_words=5):
                labels = {}
                for cluster_id in df['cluster'].unique():
                    cluster_texts = df[df['cluster'] == cluster_id]['text']
                    words = " ".join(cluster_texts).lower().split()
                    common_words = [w for w, _ in Counter(words).most_common(top_n_words)]
                    entities = df[df['cluster'] == cluster_id]['entity'].value_counts().to_dict()
                    entity_summary = ", ".join([f"{k}:{v}" for k, v in entities.items()])
                    labels[cluster_id] = f"Cluster {cluster_id} – Top words: {', '.join(common_words)} | Entities: {entity_summary}"
                return labels

            cluster_labels = make_cluster_labels(cluster_df)
            cluster_df['cluster_name'] = cluster_df['cluster'].map(cluster_labels)

            # 🔹 Marker size & symbol = severity
            size_map = {"low": 6, "medium": 10, "high": 14}
            cluster_df['marker_size'] = cluster_df['modifier'].map(size_map)
            cluster_df['symbol'] = cluster_df['modifier']

            # 🔹 X/Y axes: index vs severity_val for interpretability
            severity_map = {"low": 1, "medium": 2, "high": 3}
            cluster_df['severity_val'] = cluster_df['modifier'].map(severity_map)
            cluster_df['cluster_x'] = cluster_df['cluster'] + np.random.uniform(-0.2, 0.2, len(cluster_df))

            # 🔹 Color mapping
            palette = px.colors.qualitative.Safe
            unique_names = sorted(cluster_df['cluster_name'].unique())
            color_discrete_map = {name: palette[i % len(palette)] for i, name in enumerate(unique_names)}

            # 🔹 Plot
            fig = px.scatter(
                x=cluster_df['cluster_x'],
                y=cluster_df['severity_val'],
                color=cluster_df['cluster_name'],
                size=cluster_df['marker_size'],
                symbol=cluster_df['symbol'],
                hover_data={
                    "Entity": cluster_df['entity'],
                    "Text": cluster_df['text'],
                    "Severity": cluster_df['modifier']
                },
                color_discrete_map=color_discrete_map,
                title="Entity Clusters – Severity & Semantic Content"
            )
            fig.update_layout(
                xaxis_title="Cluster ID (with semantic spread)",
                yaxis_title="Severity (1=low, 2=medium, 3=high)",
                legend_title_text="Cluster (color) + Modifier (symbol)"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No entities detected.")

    # -----------------------------
    # 11. SHAP Explainability (tab3)
    # -----------------------------
    with tab3:
        st.subheader("Explainability – Token-Level ADE / DRUG Importance")

        import shap
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

            # -----------------------------
            # Model wrapper for SHAP (sentence-level scalar)
            # -----------------------------
        def ner_predict_scalar(texts):
            """
            Returns a scalar per sentence (max token score), so SHAP can handle variable-length tokens safely.
            """
            outputs = []
            for t in texts:
                res = ner_pipeline(t)
                enc = tokenizer(t, return_offsets_mapping=True, add_special_tokens=False)
                token_scores = np.zeros(len(enc['input_ids']))
                for r in res:
                    span_start, span_end = r['start'], r['end']
                    for idx, (s, e) in enumerate(enc['offset_mapping']):
                        if e > span_start and s < span_end:
                            token_scores[idx] = max(token_scores[idx], r['score'])
                outputs.append(np.max(token_scores))  # scalar per sentence
            return np.array(outputs)

            # -----------------------------
            # Run SHAP explainer
            # -----------------------------
        masker = shap.maskers.Text(tokenizer, mask_token="[MASK]")
        explainer = shap.Explainer(ner_predict_scalar, masker)
        shap_values = explainer([example_text])

            # -----------------------------
            # Token-level highlights using NER confidence
            # -----------------------------
            # Get tokens
        enc = tokenizer(example_text, add_special_tokens=False, return_offsets_mapping=True)
        tokens = tokenizer.convert_ids_to_tokens(enc['input_ids'])

            # Get token-level model confidence
        token_scores = np.zeros(len(tokens))
        ner_results = ner_pipeline(example_text)
        for r in ner_results:
            span_start, span_end = r['start'], r['end']
            for idx, (s, e) in enumerate(enc['offset_mapping']):
                if e > span_start and s < span_end:
                    token_scores[idx] = max(token_scores[idx], r['score'])

            # Merge subwords for readability
        merged_tokens = []
        merged_scores = []
        current_token = ""
        current_score = []

        for tok, val in zip(tokens, token_scores):
            if tok.startswith("##"):  # subword
                current_token += tok[2:]
                current_score.append(val)
            else:
                if current_token:
                    merged_tokens.append(current_token)
                    merged_scores.append(np.max(current_score))
                current_token = tok
                current_score = [val]

        if current_token:
            merged_tokens.append(current_token)
            merged_scores.append(np.max(current_score))

            # -----------------------------
            # Color-gradient text highlight
            # -----------------------------
        st.markdown("**Token-level Highlight (Red = High Importance)**")
        norm_scores = (np.array(merged_scores) - np.min(merged_scores)) / (np.ptp(merged_scores) + 1e-6)
        highlighted_text = ""
        for tok, val in zip(merged_tokens, norm_scores):
            color = f"rgba(255, 0, 0, {val*0.7 + 0.3})"  # deeper red = higher importance
            highlighted_text += f"<span style='background-color:{color};padding:2px;margin:1px;border-radius:2px'>{tok}</span> "
        st.markdown(highlighted_text, unsafe_allow_html=True)

            # -----------------------------
            # Optional bar chart
            # -----------------------------
        st.markdown("**Token-level Confidence Bar Chart**")
        shap_df = pd.DataFrame({"Token": merged_tokens, "Importance": merged_scores})
        st.bar_chart(shap_df.set_index("Token"))




