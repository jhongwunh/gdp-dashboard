import streamlit as st
import pandas as pd
import re
from io import StringIO

st.title("Sentence Tokenizer and Tag Extractor")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:", df.head())

    cols = df.columns.tolist()
    id_col = st.selectbox("Select ID Column", cols)
    context_col = st.selectbox("Select Text Column", cols)

    def tokenize(text):
        text = str(text).strip()
        tags = re.findall(r'#\w+', text)
        clean = re.sub(r'#\w+', '', text)
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', clean)
        parts = [p.strip() for p in parts if p.strip()]
        if tags:
            parts.append(' '.join(tags))
        return parts

    if st.button("Run Tokenization"):
        data = []
        for _, row in df.iterrows():
            sentences = tokenize(row[context_col])
            for i, s in enumerate(sentences, 1):
                data.append({
                    'ID': row[id_col],
                    'Sentence ID': i,
                    'Context': row[context_col],
                    'Statement': s
                })

        result = pd.DataFrame(data)
        st.write("Tokenized Preview:", result.head(10))

        csv = result.to_csv(index=False)
        st.download_button("Download Tokenized CSV", csv, "sentence_tokenized.csv", "text/csv")
