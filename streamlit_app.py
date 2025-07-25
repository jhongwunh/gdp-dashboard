# streamlit_app.py
"""Streamlit app for simple dictionary‑based text classification.

Features
========
* Upload a CSV file containing a free‑text column (e.g. marketing statements).
* Inspect & edit the tactic‑keyword dictionaries (JSON format).
* Choose which column to classify.
* Run classification and preview results in the browser.
* Download the enriched dataset as a CSV file.

Usage
-----
```bash
pip install streamlit pandas
streamlit run streamlit_app.py
```
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# 1. Default tactic dictionaries (users can edit these in the sidebar)
# -----------------------------------------------------------------------------
DEFAULT_DICTIONARIES: Dict[str, Set[str]] = {
    "urgency_marketing": {
        "limited", "limited time", "limited run", "limited edition", "order now",
        "last chance", "hurry", "while supplies last", "before they're gone",
        "selling out", "selling fast", "act now", "don't wait", "today only",
        "expires soon", "final hours", "almost gone",
    },
    "exclusive_marketing": {
        "exclusive", "exclusively", "exclusive offer", "exclusive deal",
        "members only", "vip", "special access", "invitation only",
        "premium", "privileged", "limited access", "select customers",
        "insider", "private sale", "early access",
    },
}

# JSON pretty‑print for the text editor default value
DEFAULT_DICT_JSON = json.dumps({k: sorted(v) for k, v in DEFAULT_DICTIONARIES.items()}, indent=2)

# -----------------------------------------------------------------------------
# 2. Helper — dictionary‑based classifier
# -----------------------------------------------------------------------------

def classify_statement(text: str, dicts: Dict[str, Set[str]]) -> List[str]:
    """Return list of tactic keys whose keywords appear in *text* (case‑insensitive)."""
    text_lower = text.lower()
    return [tactic for tactic, kws in dicts.items() if any(kw in text_lower for kw in kws)]


# -----------------------------------------------------------------------------
# 3. Streamlit UI layout
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Dictionary Classifier", page_icon="🗂️", layout="wide")
st.title("🗂️ Dictionary‑Based Text Classifier")
st.markdown("""
Upload a CSV file (or use the sample). Then edit the tactic keyword dictionary as needed and
click **Run Classification**.
""")

# Sidebar — dictionary editor & sample download
# --------------------------------------------
with st.sidebar:
    st.header("🔧 Tactic Dictionaries")
    dict_json = st.text_area(
        "Edit dictionaries as JSON",
        value=DEFAULT_DICT_JSON,
        height=300,
        help="Top‑level keys are tactic names; values are lists of keywords.",
    )
    st.markdown("""Need inspiration? [Download sample CSV](https://raw.githubusercontent.com/streamlit/example‑data/master/sample_data.csv)""")

# Main — file upload & parameter selection
# ---------------------------------------
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"Loaded {len(df):,} rows from **{uploaded_file.name}**")
else:
    st.info("Awaiting CSV upload … or try the sample linked in the sidebar.")
    st.stop()

# Choose text column
text_columns = [c for c in df.columns if df[c].dtype == object]
if not text_columns:
    st.error("No text columns detected. Please upload a CSV with at least one string column.")
    st.stop()

col_to_classify = st.selectbox("Select the column to classify", options=text_columns)

# Parse dictionary JSON safely
try:
    user_dict_raw = json.loads(dict_json)
    # Convert keyword lists to sets for faster lookup
    user_dictionaries: Dict[str, Set[str]] = {
        tactic: set(kws) for tactic, kws in user_dict_raw.items()
    }
except json.JSONDecodeError as e:
    st.error(f"❌ Invalid JSON in dictionary editor → {e}")
    st.stop()

# Classification trigger
if st.button("🚀 Run Classification"):
    with st.spinner("Classifying…"):
        df["_statement_clean"] = df[col_to_classify].fillna("")
        df["detected_tactics"] = df["_statement_clean"].apply(lambda s: classify_statement(s, user_dictionaries))
        # One‑hot columns for each tactic
        for tactic in user_dictionaries:
            df[tactic] = df["detected_tactics"].apply(lambda lst, t=tactic: t in lst)
        # Drop helper column
        df.drop(columns="_statement_clean", inplace=True)

    st.success("Classification complete ✅")

    # Show preview
    st.subheader("Preview")
    st.dataframe(df.head(50), use_container_width=True)

    # Download enriched CSV
    csv_buf = io.BytesIO()
    df.to_csv(csv_buf, index=False)
    csv_buf.seek(0)

    st.download_button(
        label="💾 Download classified CSV",
        data=csv_buf,
        file_name="classified_data.csv",
        mime="text/csv",
    )
