import os, re, torch
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import hdbscan
from collections import Counter
import shap

# -----------------------------
# 1. Page Setup
# -----------------------------
#st.set_page_config(page_title="ADEGuard", layout="wide")
st.set_page_config(page_title="ADEGuard", layout="wide", page_icon="💊")

st.markdown("""
<style>
/* Dark app background */
[data-testid="stAppViewContainer"] {
    background: #121212;  /* dark black/gray background */
    color: #e0e0e0;       /* default text color */
}

/* Headings font and color */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Segoe UI', sans-serif;
    color: #ffffff;  /* bright headings for contrast */
}

/* Hover effect for highlighted spans */
.hoverable-text span {
    transition: all 0.3s ease;
    display: inline-block;
    padding: 2px 4px;
    border-radius: 4px;
}
.hoverable-text span:hover {
    transform: scale(1.1);
    background-color: rgba(255, 235, 59, 0.3); /* subtle yellow highlight */
    color: #000; /* dark text on hover for contrast */
}
</style>
""", unsafe_allow_html=True)



logo_col, title_col = st.columns([1,6])
with logo_col:
    st.image("CODEbasics_logo.svg", width=80)
with title_col:
    st.markdown("<h1 style='color:#00796b'>ADEGuard – AI-Powered ADE & DRUG Detection</h1>", unsafe_allow_html=True)

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

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
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
    drug_cols = [c for c in ["VAX_NAME","VAX_TYPE"] if c in df.columns]
    ade_phrases = set(df[sym_cols].stack().dropna().str.lower().unique()) if sym_cols else set()
    drug_phrases = set(df[drug_cols].stack().dropna().str.lower().unique()) if drug_cols else set()
except Exception as e:
    st.warning(f"VAERS CSV load error: {e}")
    ade_phrases = {"headache", "fever"}
    drug_phrases = {"ibuprofen","vaccine","shot"}

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
def semantic_detection(text, ner_results, embeddings, phrases, entity_type, threshold=0.6):
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
# 6. Age extraction & classification
# -----------------------------
def extract_age(text: str):
    match = re.search(r'(?:age\s*|aged\s*|)(\d{1,3})', text, re.IGNORECASE)
    if match: return int(match.group(1))
    return None

def classify_age_group(age):
    if age is None: return 'Unknown'
    if age < 18: return 'Child'
    elif age < 60: return 'Adult'
    return 'Senior'

# -----------------------------
# 7. Clustering (with age auto)
# -----------------------------
def cluster_entities(preds, example_text, cluster_method="hdbscan"):
    if len(preds)==0: return preds
    age_val = extract_age(example_text)
    age_group = classify_age_group(age_val)
    for p in preds:
        p['age'] = age_val
        p['age_group'] = age_group

    texts=[p['text'] for p in preds]
    modifiers=[2 if semantic_severity(t)=="high" else 1 if semantic_severity(t)=="medium" else 0 for t in texts]
    emb=sbert_model.encode(texts)
    age_feat = np.array([age_val if age_val else 35]*len(preds)).reshape(-1,1)/100.0
    X=np.hstack([emb, np.array(modifiers).reshape(-1,1), age_feat])

    if len(X)>1:
        if cluster_method=="hdbscan":
            clusterer=hdbscan.HDBSCAN(min_cluster_size=2)
            labels=clusterer.fit_predict(X)
        else:
            clusterer=AgglomerativeClustering(n_clusters=None,distance_threshold=0.6)
            labels=clusterer.fit_predict(X)
    else:
        labels=[-1]*len(preds)
    for i,p in enumerate(preds):
        p['cluster']=int(labels[i])
        p['modifier']=severity_scoring(p['text'],full_text=example_text)
    return preds

# -----------------------------
# 8. Sidebar
# -----------------------------
st.sidebar.subheader("Detection Controls")

# Modifier Level
st.sidebar.markdown('<span style="color:yellow; font-weight:bold;">Highlight Modifier Level</span>', unsafe_allow_html=True)
modifier_level = st.sidebar.selectbox("", ["Low", "Medium", "High"])
st.sidebar.markdown(
    """
    **Context:**  
    - *Low:* Mild symptoms  
    - *Medium:* Moderate impact
    - *High:* Urgent attention needed  
    """,
    unsafe_allow_html=True
)

# Clustering Method
st.sidebar.markdown('<span style="color:yellow; font-weight:bold;">Clustering Method</span>', unsafe_allow_html=True)
cluster_method = st.sidebar.selectbox("", ["hdbscan", "agglomerative"])
st.sidebar.markdown(
    """
    **Context:**  
    - *HDBSCAN:* Density-Based 
    - *Agglomerative:* Hierarchy-Based
    """,
    unsafe_allow_html=True
)


# -----------------------------
# 9. Input
# -----------------------------

# Your text area
# Custom styled label
st.markdown(
    "<div style='font-size:30px; font-weight:bold; color:#1a237e; margin-bottom:5px;'>Enter symptoms here:</div>",
    unsafe_allow_html=True
)

# Text area without label
example_text = st.text_area(
    "",  # leave default label empty
    "Age 45 has severe headache after Shingrix. Age 15 has cold after anthrax. Age 43 has severe rash after taking pneumovax",
    height=150
)

if st.button("Predict ADEs & DRUGs"):
    ner_results=ner_pipeline(example_text)
    preds=[{"entity":r['entity_group'],"text":r['word'],"start":r['start'],"end":r['end'],"score":r['score']} for r in ner_results]
    preds+=semantic_detection(example_text,preds,ade_embeddings,ade_phrases,"ADE")
    preds+=semantic_detection(example_text,preds,drug_embeddings,drug_phrases,"DRUG")
    preds=cluster_entities(preds,example_text,cluster_method=cluster_method)

    tab1, tab2, tab3, tab4 = st.tabs(["Predicted Entities", "Clusters", "Explainability", "ADE–Drug Relevance"])

    # -----------------------------
    # Tab 1: Predicted Entities
    # -----------------------------
    with tab1:
        st.subheader("Detected ADEs & DRUGs")
        highlighted = example_text

        # Sort by start, longest span first to handle overlaps
        preds_sorted = sorted(preds, key=lambda x: (x['start'], -(x['end']-x['start'])))
        merged_preds = []

        for p in preds_sorted:
            overlap = False
            for mp in merged_preds:
                if not (p['end'] <= mp['start'] or p['start'] >= mp['end']):
                    # If overlapping, apply priority
                    if p['entity'] == "DRUG" or (mp['entity']=="ADE" and p['entity']=="ADE" and semantic_severity(p['text']) > semantic_severity(mp['text'])):
                        mp.update(p)  # replace with higher-priority entity
                    overlap = True
                    break
            if not overlap:
                merged_preds.append(p)

        # for p in sorted(merged_preds, key=lambda x: x['start'], reverse=True):
        #     if p['entity'] == "ADE":
        #         sev = p.get('modifier','').lower()
        #         if sev == "high":
        #             color = "#ff0000"
        #         elif sev == "medium":
        #             color = "#ffb6c1"
        #         else:
        #             color = "#b3e6ff"
        #     elif p['entity'] == "DRUG":
        #         color = "#ffd480"
        #     else:
        #         color = "#cccccc"
        #     highlighted = highlighted[:p['start']] + f"<mark style='background-color:{color}'>{p['text']}</mark>" + highlighted[p['end']:]
        selected_level = modifier_level.lower()  # from sidebar: low/medium/high
        for p in sorted(merged_preds, key=lambda x: x['start'], reverse=True):
            if p['entity'] == "ADE":
                sev = p.get('modifier','').lower()
                if sev != selected_level:
                    continue  # skip ADEs not in selected level
                if sev == "high":
                    color = "#ff0000"
                elif sev == "medium":
                    color = "#ffb6c1"
                else:
                    color = "#b3e6ff"
            elif p['entity'] == "DRUG":
                color = "#ffd480"
            else:
                color = "#cccccc"
            highlighted = highlighted[:p['start']] + f"<mark style='background-color:{color}'>{p['text']}</mark>" + highlighted[p['end']:]
        # # Drop cluster column if it exists
        # columns_to_remove = ['cluster']
        # for p in merged_preds:
        #     for col in columns_to_remove:
        #         p.pop(col, None)

        st.markdown(highlighted, unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(merged_preds))

        export_df = pd.DataFrame(merged_preds)

        if 'modifier' not in export_df.columns:
            export_df['modifier'] = export_df['text'].apply(lambda x: severity_scoring(x, full_text=example_text))

        severity_order = {'high': 0, 'medium': 1, 'low': 2}
        export_df['severity_sort'] = export_df['modifier'].str.lower().map(severity_order)
        export_df = export_df.sort_values(['severity_sort', 'text']).drop(columns='severity_sort')

        st.markdown("### Export Detected Entities (Sorted by Severity)")
        st.download_button(
            label="📥 Download Entities CSV",
            data=export_df.to_csv(index=False).encode('utf-8'),
            file_name="detected_entities_sorted.csv",
            mime="text/csv"
        )

    # -----------------------------
        # Tab 2: Cluster Visualization
        # -----------------------------
        with tab2:
            st.subheader("ADE Clusters (severity+age-aware)")
            if preds:
                # Filter only ADEs
                ade_preds = [p for p in preds if p['entity']=='ADE']

                if ade_preds:
                    # Extract age per ADE based on sentence context
                    example_sentences = re.split(r'(?<=\.)\s+', example_text)
                    for p in ade_preds:
                        # Find sentence containing ADE
                        for sent in example_sentences:
                            if p['text'] in sent:
                                p['age'] = extract_age(sent)
                                p['age_group'] = classify_age_group(p['age'])
                                break
                        else:
                            p['age'] = None
                            p['age_group'] = 'Unknown'

                    cluster_df = pd.DataFrame(ade_preds)

                    # Cluster labels for description
                    def make_cluster_labels(df, top_n_words=5):
                        labels = {}
                        for cluster_id in df['cluster'].unique():
                            cluster_texts = df[df['cluster'] == cluster_id]['text']
                            words = " ".join(cluster_texts).lower().split()
                            common_words = [w for w, _ in Counter(words).most_common(top_n_words)]
                            labels[cluster_id] = f"Cluster {cluster_id} – {', '.join(common_words)}"
                        return labels

                    # Prepare features for clustering: embedding + severity + age
                    texts = cluster_df['text'].tolist()
                    modifiers = [2 if semantic_severity(t)=="high" else 1 if semantic_severity(t)=="medium" else 0 for t in texts]
                    emb = sbert_model.encode(texts)
                    ages = np.array([a if a else 35 for a in cluster_df['age']]).reshape(-1,1) / 100.0
                    X = np.hstack([emb, np.array(modifiers).reshape(-1,1), ages])

                    # HDBSCAN clustering
                    import hdbscan
                    if len(X) > 1:
                        clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
                        labels = clusterer.fit_predict(X)
                    else:
                        labels = [-1]*len(ade_preds)
                    cluster_df['cluster'] = labels
                    cluster_labels = make_cluster_labels(cluster_df)
                    cluster_df['cluster_name'] = cluster_df['cluster'].map(cluster_labels)

                    # Plot
                    size_map = {"low": 6, "medium": 10, "high": 14}
                    cluster_df['marker_size'] = cluster_df['modifier'].map(size_map)
                    severity_map = {"low": 1, "medium": 2, "high": 3}
                    cluster_df['severity_val'] = cluster_df['modifier'].map(severity_map)
                    cluster_df['cluster_x'] = cluster_df['cluster'] + np.random.uniform(-0.2,0.2,len(cluster_df))

                    # Color by severity
                    def cluster_color(row):
                        sev = row['modifier'].lower()
                        if sev=='high': return "#ff0000"
                        elif sev=='medium': return "#ffb6c1"
                        else: return "#b3e6ff"
                    cluster_df['color'] = cluster_df.apply(cluster_color, axis=1)

                    fig = px.scatter(
                        cluster_df,
                        x='cluster_x',
                        y='severity_val',
                        size=cluster_df['age'],  # age as size
                        color='color',
                        hover_data={
                            "Text": cluster_df['text'],
                            "Severity": cluster_df['modifier'],
                            "Age": cluster_df['age'],
                            "Age Group": cluster_df['age_group'],
                            "Cluster": cluster_df['cluster_name']
                        },
                        title="ADE Clusters – Severity & Age",
                        color_discrete_map="identity"
                    )
                    fig.update_layout(
                        xaxis_title="Cluster ID (with semantic spread)",
                        yaxis_title="Severity (1=low, 2=medium, 3=high)",
                        legend_title_text="ADE Severity (color)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("No ADEs detected for clustering.")
            else:
                st.write("No entities detected.")

    # -----------------------------
    # Tab 2: Cluster Visualization
    # -----------------------------
    with tab2:
        st.subheader("Entity Clusters (age+modifier-aware)")
        if preds:
            cluster_df = pd.DataFrame(preds)
            def make_cluster_labels(df, top_n_words=5):
                labels = {}
                for cluster_id in df['cluster'].unique():
                    cluster_texts = df[df['cluster'] == cluster_id]['text']
                    words = " ".join(cluster_texts).lower().split()
                    common_words = [w for w, _ in Counter(words).most_common(top_n_words)]
                    entities = df[df['cluster'] == cluster_id]['entity'].value_counts().to_dict()
                    entity_summary = ", ".join([f"{k}:{v}" for k, v in entities.items()])
                    labels[cluster_id] = f"Cluster {cluster_id} – {', '.join(common_words)} | {entity_summary}"
                return labels

            cluster_labels = make_cluster_labels(cluster_df)
            cluster_df['cluster_name'] = cluster_df['cluster'].map(cluster_labels)

            size_map = {"low": 6, "medium": 10, "high": 14}
            cluster_df['marker_size'] = cluster_df['modifier'].map(size_map)
            cluster_df['symbol'] = cluster_df['modifier']
            severity_map = {"low": 1, "medium": 2, "high": 3}
            cluster_df['severity_val'] = cluster_df['modifier'].map(severity_map)
            cluster_df['cluster_x'] = cluster_df['cluster'] + np.random.uniform(-0.2, 0.2, len(cluster_df))

            # Color: ADE severity or DRUG gold
            def cluster_color(row):
                if row['entity']=='ADE':
                    sev = row['modifier'].lower()
                    if sev=='high': return "#ff0000"
                    elif sev=='medium': return "#ffb6c1"
                    else: return "#b3e6ff"
                elif row['entity']=='DRUG': return "#ffd480"
                else: return "#cccccc"
            cluster_df['color'] = cluster_df.apply(cluster_color, axis=1)

            fig = px.scatter(
                cluster_df,
                x='cluster_x',
                y='severity_val',
                color='color',
                size='marker_size',
                symbol='symbol',
                hover_data={
                    "Entity": cluster_df['entity'],
                    "Text": cluster_df['text'],
                    "Age": cluster_df['age'],
                    "Age Group": cluster_df['age_group'],
                    "Severity": cluster_df['modifier']
                },
                title="Entity Clusters – Severity & Age",
                color_discrete_map="identity"
            )
            fig.update_layout(
                xaxis_title="Cluster ID (with semantic spread)",
                yaxis_title="Severity (1=low, 2=medium, 3=high)",
                legend_title_text="Cluster Color (ADE/DRUG) + Modifier (symbol)"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No entities detected.")

    # -----------------------------
    # Tab 3: Explainability
    # -----------------------------
    # -----------------------------
    with tab3:
        st.subheader("Explainability – Token-Level ADE / DRUG Importance")

        # Tokenize text
        enc = tokenizer(example_text, return_offsets_mapping=True, add_special_tokens=True)
        tokens = tokenizer.convert_ids_to_tokens(enc['input_ids'])
        offsets = enc['offset_mapping']

        # Function to predict NER scalar for SHAP
        def ner_predict_scalar(texts):
            outputs = []
            for t in texts:
                res = ner_pipeline(t)
                enc_local = tokenizer(t, return_offsets_mapping=True, add_special_tokens=False)
                token_scores = np.zeros(len(enc_local['input_ids']))
                for r in res:
                    span_start, span_end = r['start'], r['end']
                    for idx, (s,e) in enumerate(enc_local['offset_mapping']):
                        if e > span_start and s < span_end:
                            token_scores[idx] = max(token_scores[idx], r['score'])
                outputs.append(np.max(token_scores))
            return np.array(outputs)

        # SHAP explanation
        masker = shap.maskers.Text(tokenizer, mask_token="[MASK]")
        explainer = shap.Explainer(ner_predict_scalar, masker)
        shap_values = explainer([example_text])

        # Aggregate subword tokens to words
        word_map = []
        current_word = ""
        current_vals = []
        word_scores = []
        for tok, val, (s,e) in zip(tokens, shap_values[0].values, offsets):
            # Skip special tokens
            if tok in tokenizer.all_special_tokens:
                continue
            # Check if token is a continuation (starts with ##)
            if tok.startswith("##"):
                current_word += tok[2:]
                current_vals.append(val)
            else:
                if current_word:
                    word_map.append(current_word)
                    word_scores.append(np.mean(current_vals))
                current_word = tok
                current_vals = [val]
        # Add last word
        if current_word:
            word_map.append(current_word)
            word_scores.append(np.mean(current_vals))

        # Normalize scores
        norm_scores = (np.array(word_scores) - np.min(word_scores)) / (np.ptp(word_scores) + 1e-6)

        # Highlight text
        highlighted_text = ""
        for word, score in zip(word_map, norm_scores):
            color = f"rgba(255,0,0,{0.3 + 0.7*score})"
            highlighted_text += f"<span style='background-color:{color};padding:2px;margin:1px;border-radius:2px'>{word}</span> "
        st.markdown("**Token-level Highlight (Red = High Impact)**")
        st.markdown(highlighted_text, unsafe_allow_html=True)

        # SHAP bar chart
        shap_df = pd.DataFrame({"Word": word_map, "Importance": word_scores})
        st.bar_chart(shap_df.set_index("Word"))

        # -----------------------------
    # Tab 4: ADE–Drug Relevance
    # -----------------------------

    with tab4:
        st.subheader("ADE–Drug Relevance (based on weak-labeled dataset)")

        # Safety check
        if 'VAX_NAME' in df.columns or 'VAX_TYPE' in df.columns:
            # Build long form: drug / ade
            drug_cols = [c for c in ["VAX_NAME", "VAX_TYPE"] if c in df.columns]
            ade_cols = [c for c in ["SYMPTOM1","SYMPTOM2","SYMPTOM3","SYMPTOM4","SYMPTOM5"] if c in df.columns]
            long_df = []
            for _, row in df.iterrows():
                drug_val = None
                for dc in drug_cols:
                    if pd.notna(row[dc]):
                        drug_val = str(row[dc]).lower()
                        break
                for ac in ade_cols:
                    if pd.notna(row[ac]) and drug_val:
                        long_df.append({
                            "drug": drug_val,
                            "ade": str(row[ac]).lower()
                        })
            long_df = pd.DataFrame(long_df)

            # Detected entities from current text
            detected_drugs = [p['text'].lower() for p in preds if p['entity']=='DRUG']
            detected_ades = [p['text'].lower() for p in preds if p['entity']=='ADE']

            if not detected_drugs or not detected_ades:
                st.write("No DRUG or ADE entities detected to compute relevance.")
            else:
                results = []
                for d in detected_drugs:
                    for a in detected_ades:
                        # fuzzy match drug in dataset (very simple contains)
                        matches = long_df[long_df['drug'].str.contains(d)]
                        if matches.empty:
                            prob = 0.0
                            count = 0
                        else:
                            total_for_drug = len(matches)
                            count = (matches['ade'] == a).sum()
                            prob = count / total_for_drug if total_for_drug>0 else 0.0
                        results.append({
                            "Drug": d,
                            "ADE": a,
                            "Count_in_Dataset": count,
                            "Total_for_Drug": total_for_drug if not matches.empty else 0,
                            "Probability (ADE given Drug)": f"{prob*100:.1f}%"
                        })
                if results:
                    res_df = pd.DataFrame(results)
                    res_df["Effect_Flag"] = res_df["Count_in_Dataset"].apply(
    lambda x: "Possible effect of drug" if x > 0 else "No effect from drug"
)
                    display_df = res_df[["Drug", "ADE", "Effect_Flag"]]
                    st.dataframe(display_df)
                    
                    # dynamic bar chart
                    chart_df = res_df.copy()
                    chart_df["Probability"] = chart_df["Probability (ADE given Drug)"].str.rstrip('%').astype(float)
                    fig = px.bar(
                        chart_df,
                        x="ADE",
                        y="Probability",
                        color="Drug",
                        barmode="group",
                        text="Probability (ADE given Drug)",
                        title="Estimated Probability of ADE given Drug (weak-labeled data)"
                    )
                    fig.update_traces(textposition="outside")
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown(
                        """
                        *How to read:*  
                        For each DRUG detected above, the bar shows what fraction of all its records
                        in the weak-labeled dataset had the same ADE.  
                        Example: A bar at 60% means “60% of this drug’s records contained this ADE”.
                        """
                    )
                else:
                    st.write("No matching ADE–Drug records found in dataset.")
        else:
            st.write("Dataset does not contain VAX_NAME/VAX_TYPE columns to compute relevance.")



