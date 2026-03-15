"""
Streamlit UI for the Legal Contract Review Agent.

Calls the Modal web endpoint to run inference.
Displays results in three sections:
  1. Risk flags (most important, shown first)
  2. Clauses found (with extracted text)
  3. Clauses missing (what wasn't found)

"""

import streamlit as st
import requests
import json

MODAL_ENDPOINT = "YOUR_MODAL_ENDPOINT_URL"



def extract_text_from_file(uploaded_file) -> str:

    if uploaded_file.type == "text/plain":
        return uploaded_file.read().decode("utf-8", errors="ignore")

    elif uploaded_file.type == "application/pdf":
        try:
            import pdfplumber
            with pdfplumber.open(uploaded_file) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except ImportError:
            st.error(
                "PDF support requires pdfplumber. "
                "Run: pip install pdfplumber"
            )
            return ""
    else:
        st.error(f"Unsupported file type: {uploaded_file.type}")
        return ""



def display_summary(summary: dict):
    """Top-level stats bar."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Clause Types Checked", summary["total_clause_types"])
    with col2:
        st.metric("Clauses Found", summary["clauses_found"])
    with col3:
        st.metric("Clauses Missing", summary["clauses_missing"])
    with col4:
        # Color the risk count red if any risks
        risk_count = summary["risk_count"]
        st.metric(
            "Risk Flags",
            risk_count,
            delta=f"{'Review needed' if risk_count > 0 else 'Clean'}",
            delta_color="inverse"
        )


def display_risks(risks: list[str]):
    """Risk flags section — shown first because most important."""
    st.subheader(" Risk Flags")

    if not risks:
        st.success("No risk flags detected.")
        return

    for risk in risks:
        if "HIGH RISK" in risk:
            st.error(risk)
        elif "MEDIUM RISK" in risk:
            st.warning(risk)
        elif "LOW RISK" in risk:
            st.info(risk)
        else:
            st.info(risk)  # NOTICE


def display_clauses(clauses: dict, summary: dict):
    """
    Two-tab view: found clauses and missing clauses.
    Separating them makes it easy to scan.
    """
    tab1, tab2 = st.tabs([
        f" Found ({summary['clauses_found']})",
        f" Missing ({summary['clauses_missing']})"
    ])

    with tab1:
        if not summary["present"]:
            st.info("No clauses found.")
        else:
            for clause_name in summary["present"]:
                texts = clauses[clause_name]
                with st.expander(f"**{clause_name}**"):
                    for i, text in enumerate(texts):
                        if len(texts) > 1:
                            st.caption(f"Occurrence {i+1}")
                        st.markdown(f"> {text}")

    with tab2:
        if not summary["missing"]:
            st.success("All clause types were found.")
        else:
            for clause_name in summary["missing"]:
                st.markdown(f"- {clause_name}")



def main():
    st.set_page_config(
        page_title="Legal Contract Review Agent",
        page_icon="⚖️",
        layout="wide"
    )

    st.title(" Legal Contract Review Agent")
    st.caption(
        "Powered by fine-tuned Mistral 7B on CUAD · "
        "Reviews contracts across 41 clause categories"
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This tool reviews legal contracts using a fine-tuned 
        language model trained on the CUAD dataset (510 contracts, 
        41 clause categories, 13,000+ expert annotations).
        
        **What it checks:**
        - Key dates and parties
        - Termination and renewal terms
        - Liability and indemnification
        - IP ownership
        - Non-compete restrictions
        - Governing law
        - And 35 more clause types
        
        **Disclaimer:** This is a research tool, not legal advice. 
        Always consult a qualified attorney before signing contracts.
        """)

        st.divider()
        st.header("Settings")
        endpoint = st.text_input(
            "Modal Endpoint URL",
            value=MODAL_ENDPOINT,
            help="Deploy serve.py to Modal and paste the URL here"
        )

    st.header("Upload Contract")

    uploaded = st.file_uploader(
        "Upload a contract to review",
        type=["pdf", "txt"],
        help="PDF or plain text files supported"
    )

    # Also allow pasting text directly
    with st.expander("Or paste contract text directly"):
        pasted_text = st.text_area(
            "Contract text",
            height=200,
            placeholder="Paste your contract text here..."
        )

    contract_text = ""

    if uploaded:
        contract_text = extract_text_from_file(uploaded)
        st.success(f"Loaded: {uploaded.name} ({len(contract_text):,} characters)")
    elif pasted_text:
        contract_text = pasted_text

    if contract_text and st.button(
        " Review Contract",
        type="primary",
        use_container_width=True
    ):
        if endpoint == "YOUR_MODAL_ENDPOINT_URL":
            st.error(
                "Please set your Modal endpoint URL in the sidebar settings."
            )
            return

        with st.spinner(
            "Reviewing contract... This takes 1-2 minutes. "
            "The model is checking 41 clause types across the full document."
        ):
            try:
                response = requests.post(
                    endpoint,
                    json={"contract_text": contract_text},
                    timeout=300  # 5 min timeout
                )
                response.raise_for_status()
                result = response.json()

            except requests.exceptions.Timeout:
                st.error(
                    "Request timed out. The contract may be too long. "
                    "Try a shorter document."
                )
                return
            except requests.exceptions.ConnectionError:
                st.error(
                    "Could not connect to Modal endpoint. "
                    "Make sure serve.py is deployed."
                )
                return
            except Exception as e:
                st.error(f"Error: {str(e)}")
                return

        st.divider()
        st.header("Review Results")

        display_summary(result["summary"])
        st.divider()

        display_risks(result["risks"])
        st.divider()

        display_clauses(result["clauses"], result["summary"])

        # Download results as JSON
        st.divider()
        st.download_button(
            label="⬇️ Download Full Report (JSON)",
            data=json.dumps(result, indent=2),
            file_name="contract_review.json",
            mime="application/json"
        )


if __name__ == "__main__":
    main()
