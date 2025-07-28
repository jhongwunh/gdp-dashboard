# Install spaCy and download model
!pip install spacy ipywidgets pandas --quiet
!python -m spacy download en_core_web_sm

# Load packages
import pandas as pd
import spacy
from google.colab import files
from IPython.display import display
import ipywidgets as widgets

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Sentence tokenizer using spaCy
def spacy_sent_tokenize(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip() and any(c.isalnum() for c in sent.text)]

# Upload CSV
print("📂 Please upload your CSV file:")
uploaded = files.upload()
filename = list(uploaded.keys())[0]
df = pd.read_csv(filename)
print("✅ File uploaded. Here's a preview:")
display(df.head())

# Column selection
column_options = df.columns.tolist()

id_selector = widgets.Dropdown(
    options=column_options,
    description='🆔 Select ID column:',
    style={'description_width': 'initial'}
)

content_selector = widgets.Dropdown(
    options=column_options,
    description='📝 Select Content column:',
    style={'description_width': 'initial'}
)

statement_cut = widgets.ToggleButtons(
    options=['sentence', 'post'],
    description='Statement cut:',
    style={'description_width': 'initial'}
)

context_cut = widgets.ToggleButtons(
    options=['whole'],
    description='Context cut (currently only "whole" is supported):',
    style={'description_width': 'initial'}
)

process_button = widgets.Button(description="🚀 Transform Data")
output_box = widgets.Output()

def on_process_button_click(b):
    output_box.clear_output()
    with output_box:
        ID_col = id_selector.value
        text_col = content_selector.value
        cut = statement_cut.value

        result = []

        for _, row in df.iterrows():
            full_text = str(row[text_col])
            current_id = row[ID_col]

            if cut == 'sentence':
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
        print("✅ Transformation complete! Here's a preview:")
        display(df_out.head())

        output_filename = "transformed_output.csv"
        df_out.to_csv(output_filename, index=False)
        files.download(output_filename)

process_button.on_click(on_process_button_click)

# Show interface
print("📌 Choose your columns and options below:")
display(id_selector, content_selector, statement_cut, context_cut, process_button, output_box)
