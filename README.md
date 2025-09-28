# ADEGuard – AI-Powered Healthcare Risk Monitoring Application  
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Streamlit Demo](https://img.shields.io/badge/demo-Streamlit-brightgreen)](https://your-streamlit-app-link)

> **ADEGuard** is an end-to-end application for extracting and analyzing Adverse Drug Events (ADEs) from patient narratives.  
> It combines fine-tuned NER, semantic embeddings, unsupervised clustering, and interactive dashboards to surface safety signals and support healthcare decision-making.

---

## 📸 Application Interface
<img width="1897" height="959" alt="image" src="https://github.com/user-attachments/assets/0958f51f-6384-4031-894b-2ce4151ac36f" />


---

## 🚀 Features  
- **Fine-tuned NER Model** (BERT-based) to extract ADEs, drug names, and clinical entities  
- **Embedding + Clustering** using Sentence Transformers and HDBSCAN/Agglomerative Clustering  
- **Model Interpretability** via SHAP token contribution plots  
- **Interactive Dashboard** built with Streamlit + Plotly  
- **Age-group and severity-based ADE visualization**  

---

## 📝 Intended Uses  
- Detect and label Adverse Drug Events from unstructured clinical text or patient feedback  
- Cluster ADEs by similarity, severity, or demographic group  
- Provide interpretable outputs via dashboards for regulatory, safety, or research teams  
- Enable non-technical users to explore ADE trends and flag high-risk cases  

---

## ⚠️ Limitations & Ethical Considerations  
- Not a diagnostic or regulatory tool — outputs are for research and monitoring only  
- Model performance may degrade on out-of-domain data, non-English text, or rare ADEs  
- Clustering highlights patterns but does not imply causality  

---

## 🛠️ Tech Stack  
- **Python 3.10+**
- **Transformers / Hugging Face** for NER  
- **Sentence Transformers** for embeddings  
- **HDBSCAN & Agglomerative Clustering** for grouping ADEs  
- **SHAP** for model interpretability  
- **Streamlit + Plotly** for dashboards  
- **pandas, numpy, scikit-learn, tqdm** for data processing  

---

## 📊 Training & Data  
- **Base Model:** BERT-base (transformers `AutoModelForTokenClassification`)  
- **Fine-Tuning Data:** Manually annotated ADE + Drug spans + weak supervision  
- **Embeddings:** SentenceTransformer model  
- **Evaluation Metrics:** F1, Precision, Recall on held-out annotated test set  

<img width="769" height="327" alt="image" src="https://github.com/user-attachments/assets/bf7e7c9b-2a75-47ed-a92c-a3da1ea2bacc" />


---

## ⚡ Quick Start  

Download the model files which includes pretrained weights from link:https://drive.google.com/drive/folders/1BA2rh6-vqJxVFfnT-ek1iXGIUYF2VDhR?usp=drive_link
Store it along with the app.py file present directory and run the following command.
streamlit run app.py

Else just add the trained weights(safetensors file and add to model file)
Clone this repository and install dependencies:

```bash
git clone https://github.com/yourusername/ADEGuard.git
cd outputs
pip install -r requirements.txt
streamlit run app.py

