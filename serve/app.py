"""
Streamlit UI for the Legal Contract Review Agent.
Run locally with: streamlit run serve/app.py
"""

import streamlit as st
import requests
import json

MODAL_ENDPOINT = "https://arvind-kurapati--legal-contract-serve-contractreviewserv-6eb889.modal.run"

SAMPLE_CONTRACT = """DISTRIBUTOR AGREEMENT

THIS DISTRIBUTOR AGREEMENT (the "Agreement") is made by and between 
Acme Corporation, a Delaware corporation ("Company"), and XYZ Distributors 
Inc., a California corporation ("Distributor").

1. TERM. This Agreement shall commence on January 1, 2024 and shall 
continue for a period of two (2) years, unless earlier terminated.

2. GOVERNING LAW. This Agreement shall be governed by and construed in 
accordance with the laws of the State of Delaware.

3. TERMINATION FOR CONVENIENCE. Either party may terminate this Agreement 
upon thirty (30) days written notice to the other party.

4. NON-COMPETE. During the term of this Agreement and for a period of 
one (1) year thereafter, Distributor shall not directly or indirectly 
compete with Company in the sale of similar products.

5. INDEMNIFICATION. Distributor shall indemnify and hold harmless Company 
from any claims arising from Distributor's breach of this Agreement.

6. LIMITATION OF LIABILITY. In no event shall either party be liable for 
indirect, incidental, or consequential damages. Company's total liability 
shall not exceed the fees paid in the prior three months.

7. RENEWAL. This Agreement shall automatically renew for successive 
one-year terms unless either party provides sixty (60) days written 
notice of non-renewal prior to expiration.

8. AUDIT RIGHTS. Company shall have the right to audit Distributor's 
books and records upon reasonable notice to verify compliance.

9. ANTI-ASSIGNMENT. Distributor may not assign this Agreement without 
Company's prior written consent.

10. INSURANCE. Distributor shall maintain commercial general liability 
insurance of at least $1,000,000 per occurrence during the term."""


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
            st.error("PDF support requires pdfplumber. Run: pip install pdfplumber")
            return ""
    else:
        st.error(f"Unsupported file type: {uploaded_file.type}")
        return ""


def display_summary(summary: dict):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Clause Types Checked", summary["total_clause_types"])
    with col2:
        st.metric("Clauses Found", summary["clauses_found"])
    with col3:
        st.metric("Clauses Missing", summary["clauses_missing"])
    with col4:
        risk_count = summary["risk_count"]
        st.metric(
            "Risk Flags",
            risk_count,
            delta=f"{'Review needed' if risk_count > 0 else 'Clean'}",
            delta_color="inverse"
        )


def display_risks(risks: list):
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
            st.info(risk)


def display_clauses(clauses: dict, summary: dict):
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
                        st.markdown(f"> {text[:500]}{'...' if len(text) > 500 else ''}")
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

    st.title("Legal Contract Review Agent")
    st.caption(
        "Powered by Mistral 7B Instruct · "
        "Reviews contracts across 41 clause categories · "
        "Trained on CUAD (510 contracts, 13,000+ expert annotations)"
    )

    with st.sidebar:
        st.header("About")
        st.markdown("""
        Reviews legal contracts across 41 clause categories 
        from the CUAD benchmark. Flags missing or high-risk clauses.
        
        **What it checks:**
        - Key dates and parties
        - Termination and renewal terms
        - Liability and indemnification
        - IP ownership
        - Non-compete restrictions
        - Governing law
        - And 35 more clause types
        
        **Disclaimer:** Research tool, not legal advice. 
        Always consult a qualified attorney before signing.
        """)

        st.divider()
        st.header("Resources")
        st.markdown("""
        - [GitHub Repo](https://github.com/AravindKurapati/legal-contract-agent)
        - [CUAD Dataset](https://huggingface.co/datasets/cuad)
        - [Research Analysis](/results/analysis.md)
        """)

    # ── Upload section ────────────────────────────────────────────────────────
    st.header("Upload Contract")

    uploaded = st.file_uploader(
        "Upload a contract (PDF or TXT)",
        type=["pdf", "txt"],
        help="Max 200MB. For best results use contracts under 50k characters."
    )

    with st.expander("Or paste contract text directly"):
        if st.button(" Load sample contract"):
            st.session_state.sample_text = SAMPLE_CONTRACT
            st.success("Sample contract loaded. Click Review Contract below.")

        pasted_text = st.text_area(
            "Contract text",
            height=200,
            value=st.session_state.get("sample_text", ""),
            placeholder="Paste contract text here, or click 'Load sample contract' above..."
        )

    # ── Review ────────────────────────────────────────────────────────────────
    contract_text = ""
    if uploaded:
        contract_text = extract_text_from_file(uploaded)
        st.success(f"Loaded: {uploaded.name} ({len(contract_text):,} characters)")
    elif pasted_text:
        contract_text = pasted_text

    if contract_text:
        char_count = len(contract_text)
        est_minutes = max(3, round(char_count / 15000))
        st.caption(f"Contract size: {char_count:,} characters · Estimated review time: {est_minutes}-{est_minutes+3} minutes")

    if contract_text and st.button(
        " Review Contract",
        type="primary",
        use_container_width=True
    ):
        with st.spinner(
            "Reviewing contract... The agent is checking 41 clause types. "
            "First request takes ~15 seconds to warm up."
        ):
            try:
                response = requests.post(
                    MODAL_ENDPOINT,
                    json={"contract_text": contract_text},
                    timeout=600
                )
                response.raise_for_status()
                result = response.json()

            except requests.exceptions.Timeout:
                st.error(
                    "Request timed out. Try a shorter contract (under 50k characters) "
                    "or paste a section of the contract instead of the full document."
                )
                return
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to Modal endpoint. The service may be starting up — try again in 30 seconds.")
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

        st.divider()
        st.download_button(
            label=" Download Full Report (JSON)",
            data=json.dumps(result, indent=2),
            file_name="contract_review.json",
            mime="application/json"
        )


if __name__ == "__main__":
    main()