import streamlit as st
import pandas as pd
import spacy
from io import StringIO

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Sentence tokenizer using spaCy
def spacy_sent_tokenize(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip() and any(c.isalnum() for c in sent.text)]

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
                sentences = spacy_sent_tokenize(full_text)
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
