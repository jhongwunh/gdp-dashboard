import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from io import StringIO
import os

# Download punkt at startup if not found
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Sentence tokenizer using NLTK
def nltk_sent_tokenize(text):
    return [sent.strip() for sent in sent_tokenize(text) if sent.strip() and any(c.isalnum() for c in sent)]

st.title("📄 Text Transformation App")
st.write("Upload a CSV file, select columns, and choose how to segment the text.")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
    st.write("Preview of the data:")
    st.dataframe(df.head())

    # Column selection
    columns = df.columns.tolist()
    id_col = st.selectbox("🆔 Select ID column:", columns)
    content_col = st.selectbox("📝 Select Content column:", columns)

    # Options
    statement_cut = st.radio("Statement cut:", ['sentence', 'post'])
    context_cut = st.radio("Context cut (currently only 'whole' is supported):", ['whole'])

    if st.button("🚀 Transform Data"):
        result = []

        for _, row in df.iterrows():
            full_text = str(row[content_col])
            current_id = row[id_col]

            if statement_cut == 'sentence':
                sentences = nltk_sent_tokenize(full_text)
            else:
                sentences = [full_text]

            for i, sent in enumerate(sentences, start=1):
                result.append({
                    "ID": current_id,
                    "Sentence ID": i,
                    "Context": full_text,
                    "Statement": sent
                })

        df_out = pd.DataFrame(result)
        st.success("✅ Transformation complete!")
        st.write("Preview of the transformed data:")
        st.dataframe(df_out.head())

        csv = df_out.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Transformed CSV",
            data=csv,
            file_name='transformed_output.csv',
            mime='text/csv'
        )
