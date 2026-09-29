from __future__ import annotations

# import io  # PDF upload disabled
from typing import List, Dict, Any

import streamlit as st
from pypdf import PdfReader  # PDF upload disabled

# LangChain building blocks
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# Step 1: Mock Database Definition
# ============================================================

MOCK_BIS_STANDARDS: List[Dict[str, Any]] = [
    {
        "is_number": "IS 8041:1996",
        "title": "Rapid Hardening Portland Cement - Specification",
        "status": "Active (Reaffirmed 2020)",
        "latest_version": "Current",
        "amendments": "Amendment No. 1, 2",
        "scope_description": (
            "Specification for rapid hardening Portland cement used in construction "
            "projects requiring high early strength development within 24 to 72 hours. "
            "Applicable to precast concrete works, road repairs, cold weather concreting, "
            "and structural elements where early formwork removal is essential. Covers "
            "chemical composition, fineness, setting time, soundness, and compressive "
            "strength requirements for cement manufactured by grinding Portland cement "
            "clinker with gypsum."
        ),
        "normative_references": ["IS 269:2015", "IS 4031 (Part 1-25)"],
        "mandatory_certification": "BIS Mandatory",
    },
    {
        "is_number": "IS 2062:2006",
        "title": "Hot Rolled Medium and High Tensile Structural Steel",
        "status": "Superseded",
        "latest_version": "IS 2062:2011",
        "amendments": "None",
        "scope_description": (
            "Specification for hot rolled structural steel plates, strips, shapes, and "
            "sections used in the fabrication of general engineering and structural work "
            "including bridges, buildings, railway rolling stock, transmission line towers, "
            "and other load bearing structures. Defines chemical composition, mechanical "
            "properties, tensile strength, yield stress, elongation, impact resistance, "
            "and grade classification (E250, E350, E410, E450) for weldable structural steel. "
            "(Note: This is an older version used for testing version control alerts.)"
        ),
        "normative_references": ["IS 808:1989", "IS 1852:1985"],
        "mandatory_certification": "BIS Mandatory",
    },
    {
        "is_number": "IS 694:2010",
        "title": "PVC Insulated Cables for Working Voltages up to and including 1100 V",
        "status": "Active",
        "latest_version": "Current",
        "amendments": "None",
        "scope_description": (
            "Specification for polyvinyl chloride (PVC) insulated electrical cables with "
            "copper or aluminium conductors used for domestic wiring, industrial installations, "
            "control panels, and low voltage power distribution up to 1100 volts. Covers "
            "conductor construction, insulation thickness, sheathing, colour coding, "
            "flammability, insulation resistance, and electrical safety requirements for "
            "single-core and multi-core cables in fixed wiring applications."
        ),
        "normative_references": ["IS 8130:2013", "IS 5831:1984"],
        "mandatory_certification": "BIS Mandatory",
    },
    {
        "is_number": "IS 2925:1984",
        "title": "Industrial Safety Helmets - Specification",
        "status": "Active",
        "latest_version": "Current",
        "amendments": "Amendment No. 1",
        "scope_description": (
            "Specification for industrial safety helmets (hard hats) used as personal "
            "protective equipment (PPE) by workers on construction sites, factories, mines, "
            "and hazardous industrial environments to protect the head from falling objects, "
            "impact, and penetration. Defines material requirements, shell design, harness "
            "and cradle construction, shock absorption, penetration resistance, flammability, "
            "and electrical insulation performance for occupational head protection."
        ),
        "normative_references": ["IS 4151:2015"],
        "mandatory_certification": "BIS Mandatory",
    },
    {
        "is_number": "IS 10500:2012",
        "title": "Drinking Water - Specification",
        "status": "Active",
        "latest_version": "Current",
        "amendments": "Amendment No. 1, 2",
        "scope_description": (
            "Specification for the quality of drinking water supplied by municipal bodies, "
            "packaged water manufacturers, and other water supply agencies for human "
            "consumption. Defines acceptable and permissible limits for physical parameters "
            "(colour, turbidity, taste, odour), chemical parameters (pH, total dissolved solids, "
            "hardness, chlorides, fluorides, heavy metals), microbiological quality (E. coli, "
            "coliform bacteria), pesticide residues, and radioactive substances in potable water."
        ),
        "normative_references": ["IS 3025 (Part 1-65)", "IS 1622:1981"],
        "mandatory_certification": "None",
    },
    {
        "is_number": "IS 13252 (Part 1):2010",
        "title": "Information Technology Equipment - Safety - General Requirements",
        "status": "Active",
        "latest_version": "Current",
        "amendments": "Amendment No. 1, 2, 3",
        "scope_description": (
            "Specification for the safety of information technology equipment (ITE) including "
            "laptops, desktop computers, servers, printers, scanners, monitors, tablets, mobile "
            "phones, power adaptors, set-top boxes, and networking equipment intended for "
            "office and household use. Covers protection against electric shock, energy hazards, "
            "fire, mechanical and heat hazards, radiation, and chemical hazards. Applies to "
            "electronic and IT hardware sold, imported, or distributed in India and forms the "
            "technical basis for the Compulsory Registration Scheme (CRS) notified by MeitY."
        ),
        "normative_references": ["IS/IEC 60950-1", "IS 616:2017"],
        "mandatory_certification": "Compulsory Registration Scheme (CRS) via MeitY",
    },
    {
        "is_number": "IS 1417:2016",
        "title": "Gold and Gold Alloys, Jewellery/Artefacts - Fineness and Marking",
        "status": "Active",
        "latest_version": "Current",
        "amendments": "Amendment No. 1",
        "scope_description": (
            "Specification for the fineness, composition, and hallmarking of gold and gold "
            "alloy jewellery, ornaments, coins, and artefacts sold to consumers in India. "
            "Defines permissible caratage grades (14K, 18K, 20K, 22K, 23K, 24K), assaying "
            "methods, purity tolerances, and the mandatory BIS Hallmark marking, consisting "
            "of the BIS logo, purity/fineness number, assaying centre identification, and "
            "jeweller identification. Applicable to bullion dealers, jewellery manufacturers, "
            "retailers, and e-commerce sellers of gold ornaments."
        ),
        "normative_references": ["IS 1418:2009", "IS 2790:2017"],
        "mandatory_certification": "BIS Hallmarking Mandatory",
    },
]


# ============================================================
# Step 2: Vector Database Initialization
# ============================================================

@st.cache_resource(show_spinner="Loading embedding model and building FAISS index...")
def build_vector_store() -> FAISS:
    """Initialize HuggingFace embeddings and a FAISS vector store."""
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    documents: List[Document] = []

    for record in MOCK_BIS_STANDARDS:
        documents.append(
            Document(
                page_content=record["scope_description"],
                metadata={
                    "is_number": record["is_number"],
                    "title": record["title"],
                    "status": record.get("status", "Active"),
                    "latest_version": record.get("latest_version", "Current"),
                    "amendments": record.get("amendments", "None"),
                    "normative_references": record["normative_references"],
                    "mandatory_certification": record["mandatory_certification"],
                },
            )
        )

    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store


# ============================================================
# Step 3: Semantic Search Pipeline
# ============================================================

def l2_distance_to_confidence(l2_distance: float) -> float:
    """Convert a FAISS L2 (squared) distance to a cosine-similarity percentage."""
    cosine_similarity = 1.0 - (float(l2_distance) / 2.0)
    confidence_pct = max(0.0, min(1.0, cosine_similarity)) * 100.0
    return round(confidence_pct, 2)


def recommend_standards(
    vector_store: FAISS,
    query: str,
    top_k: int = 2,
) -> List[Dict[str, Any]]:
    """Return the top matching Indian Standards for a user query."""
    if not query or not query.strip():
        return []

    results = vector_store.similarity_search_with_score(query, k=top_k)
    recommendations: List[Dict[str, Any]] = []

    for doc, score in results:
        recommendations.append(
            {
                "is_number": doc.metadata.get("is_number", "N/A"),
                "title": doc.metadata.get("title", "N/A"),
                "status": doc.metadata.get("status", "Active"),
                "latest_version": doc.metadata.get("latest_version", "Current"),
                "amendments": doc.metadata.get("amendments", "None"),
                "scope_description": doc.page_content,
                "normative_references": doc.metadata.get("normative_references", []),
                "mandatory_certification": doc.metadata.get(
                    "mandatory_certification", "None"
                ),
                "l2_distance": float(score),
                "match_confidence": l2_distance_to_confidence(score),
            }
        )

    return recommendations


# ============================================================
# Multilingual Input Support (mock translation)
# ============================================================

SUPPORTED_LANGUAGES: List[str] = ["English", "Hindi", "Tamil", "Bengali"]


def translate_to_english(text: str, source_language: str) -> str:
    """Mock translator: returns text unchanged; real impl would call an MT API."""
    if not text:
        return text

    if source_language == "English":
        return text

    return text


# ============================================================
# PDF Tender Upload (Document Ingestion) [DISABLED]
# ============================================================

def extract_pdf_text(pdf_bytes: bytes, max_pages: int = 2) -> str:
    """Return concatenated text from the first max_pages pages of a PDF."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages_to_read = min(max_pages, len(reader.pages))

    chunks: List[str] = []

    for i in range(pages_to_read):
        page_text = reader.pages[i].extract_text() or ""
        if page_text.strip():
            chunks.append(page_text.strip())

    return "\n\n".join(chunks).strip()


# ============================================================
# Step 4: Streamlit UI Implementation
# ============================================================

def render_recommendation(
    rec: Dict[str, Any],
    is_primary: bool = True,
) -> None:
    """Render a single BIS recommendation card in the Streamlit UI."""
    label = "Primary Recommended Standard" if is_primary else "Related Standard"
    st.markdown(
        f'<div class="rec-label">{label}</div>',
        unsafe_allow_html=True,
    )

    status = rec.get("status", "Active")
    superseded = status == "Superseded"

    # ---- Header row: title/IS on the left, confidence on the right ----
    left, right = st.columns([4, 1])

    with left:
        st.markdown(
            f'<div class="rec-title">{rec["title"]}</div>'
            f'<div class="rec-meta">IS Number: '
            f'<span class="is-num">{rec["is_number"]}</span></div>',
            unsafe_allow_html=True,
        )

        if superseded:
            st.markdown(
                f'<div class="rec-status rec-status-warn">Superseded '
                f'&middot; Latest active version: '
                f'<b>{rec.get("latest_version")}</b></div>',
                unsafe_allow_html=True,
            )
        else:
            amendments = rec.get("amendments", "None")
            st.markdown(
                f'<div class="rec-status rec-status-ok">Status: {status} '
                f'&middot; Amendments: {amendments}</div>',
                unsafe_allow_html=True,
            )

    with right:
        st.metric(
            label="Match Confidence",
            value=f'{rec["match_confidence"]:.1f}%',
            help="How closely this standard matches the tender description.",
        )

    # ---- Superseded note (compact) ----
    if superseded:
        st.markdown(
            '<div class="rec-callout rec-callout-warn">'
            "Update your tender to reference the latest active version "
            "to avoid procurement disputes."
            "</div>",
            unsafe_allow_html=True,
        )

    with st.expander("Scope Description"):
        st.write(rec["scope_description"])

    refs = rec.get("normative_references") or []
    if refs:
        st.markdown(
            '<div class="rec-refs-label">Related References</div>'
            + "".join(
                f'<span class="ref-chip">{r}</span>' for r in refs
            ),
            unsafe_allow_html=True,
        )

    cert = rec.get("mandatory_certification", "None")
    if isinstance(cert, str) and cert.strip().lower() != "none":
        st.markdown(
            '<div class="rec-callout rec-callout-cert">'
            f"<b>Mandatory certification:</b> {cert}. "
            "Tender clauses should require a valid BIS licence (ISI mark) "
            "and product test certificates before award."
            "</div>",
            unsafe_allow_html=True,
        )


def main() -> None:
    """Entry point for the Streamlit application."""
    st.set_page_config(
        page_title="BIS AI Procurement Assistant",
        page_icon="IN",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # ========================================================
    # Government of India Portal Styling (CSS injection)
    # ========================================================

    st.markdown(
        """
        <style>
        /* ---- Global typography ---- */
        html, body, [class*="css"], .stApp,
        .stMarkdown, .stButton, .stTextInput, .stTextArea, .stSelectbox {
            font-family: "Segoe UI", Roboto, Arial, "Helvetica Neue", sans-serif !important;
            color: #333333;
        }

        /* ---- Hide Sidebar Completely ---- */
        [data-testid="collapsedControl"] {
            display: none !important;
        }

        section[data-testid="stSidebar"] {
            display: none !important;
        }

        /* ---- Remove default heavy top padding so header sits flush ---- */
        .block-container {
            padding-top: 1.2rem !important;
            padding-bottom: 2rem !important;
            max-width: 1280px;
        }

        header[data-testid="stHeader"] {
            background: transparent;
            height: 0rem;
        }

        /* ---- Tiranga top + bottom accent bars on the whole app ---- */
        .stApp {
            background-color: #FFFFFF;
            border-top: 4px solid #FF9933; /* Saffron */
            border-bottom: 4px solid #138808; /* Green */
        }

        /* ---- Official Government of India strip ---- */
        .goi-strip {
            font-size: 12px;
            color: #555555;
            letter-spacing: 0.3px;
            padding: 6px 2px 8px 2px;
            border-bottom: 1px solid #E5E7EB;
            margin-bottom: 14px;
        }

        .goi-strip strong {
            color: #003366;
        }

        .goi-strip .sep {
            color: #999999;
            margin: 0 8px;
        }

        /* ---- Main headers (title, h1-h3) ---- */
        h1, h2, h3, h4 {
            color: #003366 !important;
            font-weight: 600 !important;
            letter-spacing: 0.2px;
        }

        h1 {
            border-bottom: 2px solid #003366;
            padding-bottom: 6px;
        }

        /* ---- Custom Navbar ---- */
        .custom-navbar {
            background-color: #003366;
            padding: 12px 20px;
            display: flex;
            align-items: center;
            gap: 25px;
            border-radius: 2px;
            margin-top: 1px;
            margin-bottom: 25px;
            border-bottom: 3px solid #FF9933;
        }

        .custom-navbar a {
            color: #FFFFFF !important;
            text-decoration: none;
            font-weight: 600;
            font-size: 14px;
            transition: color 0.2s;
        }

        .custom-navbar a:hover {
            color: #FF9933 !important;
        }

        .nav-status {
            margin-left: auto;
            color: #FFFFFF;
            font-size: 12px;
            background-color: rgba(255,255,255,0.1);
            padding: 4px 10px;
            border-radius: 2px;
            border: 1px solid rgba(255,255,255,0.2);
        }

        /* ---- Primary buttons: rigid, boxy, navy ---- */
        .stButton > button {
            background-color: #003366 !important;
            color: #FFFFFF !important;
            border: 1px solid #003366 !important;
            border-radius: 2px !important;
            font-weight: 600 !important;
            padding: 0.55rem 1.2rem !important;
            box-shadow: none !important;
            transition: background-color 0.15s ease-in-out,
                        border-color 0.15s ease-in-out;
        }

        .stButton > button:hover {
            background-color: #FF9933 !important;
            border-color: #FF9933 !important;
            color: #FFFFFF !important;
        }

        .stButton > button:focus {
            outline: 2px solid #138808 !important;
        }

        /* ---- Text area / inputs: sharp corners, subtle border ---- */
        textarea, input[type="text"], .stTextInput input,
        .stTextArea textarea,
        .stSelectbox div[data-baseweb="select"] > div {
            outline-offset: 1px;
            border-radius: 2px !important;
            border: 1px solid #CBD5E1 !important;
            background-color: #FFFFFF !important;
            color: #333333 !important;
        }

        textarea:focus, input:focus {
            border-color: #003366 !important;
            box-shadow: 0 0 0 1px #003366 !important;
        }

        /* ---- File uploader: boxy government form look ---- */
        [data-testid="stFileUploader"] section {
            border: 1px dashed #003366 !important;
            border-radius: 2px !important;
            background-color: #F8F9FA !important;
        }

        /* ---- Alert boxes: sharp corners, thin solid borders ---- */
        div[data-testid="stAlert"] {
            border-radius: 2px !important;
            border-width: 1px !important;
            border-style: solid !important;
            box-shadow: none !important;
        }

        div[data-testid="stAlert"][data-baseweb="notification"]:has(svg[title="info"]) {
            border-color: #003366 !important;
            background: #F8F9FA !important;
            color: #003366 !important;
        }

        div[data-testid="stAlert"][data-baseweb="notification"]:has(svg[title="warning"]) {
            border-color: #FF9933 !important;
            background: #FFF8EE !important;
            color: #7A4A12 !important;
        }

        div[data-testid="stAlert"][data-baseweb="notification"]:has(svg[title="success"]) {
            border-color: #138808 !important;
            background: #F1F8F1 !important;
            color: #136B08 !important;
        }

        div[data-testid="stAlert"][data-baseweb="notification"]:has(svg[title="error"]) {
            border-color: #C00020 !important;
            background: #FDF3F3 !important;
            color: #7A0012 !important;
        }

        /* ---- Metric widget: subtle KPI tile ---- */
        div[data-testid="stMetric"] {
            background-color: #F8F9FA;
            border: 1px solid #E5E7EB;
            border-radius: 2px;
            padding: 8px 12px;
        }

        div[data-testid="stMetricLabel"] {
            color: #6B7280 !important;
            font-weight: 500 !important;
            font-size: 12px !important;
        }

        div[data-testid="stMetricValue"] {
            color: #003366 !important;
            font-weight: 600 !important;
            font-size: 22px !important;
        }

        /* ---- Expander: boxy government accordion ---- */
        details, div[data-testid="stExpander"] {
            border: 1px solid #E5E7EB !important;
            border-radius: 2px !important;
            background-color: #F8F9FA !important;
        }

        div[data-testid="stExpander"] summary {
            color: #003366 !important;
            font-weight: 600 !important;
        }

        /* ---- Divider: plain subtle line ---- */
        div[data-testid="stDivider"] hr {
            border: none !important;
            height: 1px !important;
            background: #E5E7EB !important;
        }

        /* ---- Recommendation card styling ---- */
        .rec-label {
            font-size: 11px;
            letter-spacing: 1px;
            text-transform: uppercase;
            color: #6B7280;
            font-weight: 600;
            margin-bottom: 4px;
        }

        .rec-title {
            font-size: 26px;
            font-weight: 600;
            color: #003366;
            line-height: 1.3;
        }

        .rec-meta {
            font-size: 13px;
            color: #555;
            margin-top: 2px;
        }

        .rec-meta .is-num {
            font-family: "Consolas", monospace;
            color: #003366;
        }

        .rec-status {
            display: inline-block;
            margin-top: 8px;
            font-size: 12px;
            padding: 3px 8px;
            border-radius: 2px;
            border: 1px solid;
        }

        .rec-status-ok {
            color: #0B4A05;
            border-color: #C8E6C9;
            background: #F1F8F1;
        }

        .rec-status-warn {
            color: #7A0012;
            border-color: #F1C6C6;
            background: #FDF3F3;
        }

        .rec-callout {
            margin-top: 10px;
            padding: 8px 12px;
            font-size: 13px;
            border-radius: 2px;
            border-left: 3px solid #CBD5E1;
            background: #F8F9FA;
            color: #333;
        }

        .rec-callout-warn {
            border-left-color: #B00020;
            background: #FDF3F3;
            color: #7A0012;
        }

        .rec-callout-cert {
            border-left-color: #FF9933;
            background: #FFF8EE;
            color: #6B4A00;
        }

        .rec-refs-label {
            font-size: 12px;
            color: #6B7280;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            margin: 14px 0 6px 0;
        }

        .ref-chip {
            display: inline-block;
            font-family: "Consolas", monospace;
            font-size: 12px;
            color: #003366;
            background: #F1F5F9;
            border: 1px solid #E2E8F0;
            border-radius: 2px;
            padding: 2px 8px;
            margin: 0 6px 0 0;
        }

        /* ---- Footer ---- */
        .goi-footer {
            margin-top: 24px;
            padding-top: 10px;
            border-top: 1px solid #E5E7EB;
            font-size: 11px;
            color: #6B7280;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ========================================================
    # Official Government of India strip
    # ========================================================

    st.markdown(
        """
        <div class="goi-strip">
            <strong>Government of India</strong>
            <span class="sep">|</span>
            Ministry of Consumer Affairs, Food and Public Distribution
            <span class="sep">|</span>
            <strong>Bureau of Indian Standards</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ========================================================
    # Custom Navbar
    # ========================================================

    st.markdown(
        """
        <div class="custom-navbar">
            <a href="https://gem.gov.in" target="_blank">GeM Portal</a>
            <a href="https://eprocure.gov.in" target="_blank">
                Central Public Procurement
            </a>
            <a href="https://www.bis.gov.in" target="_blank">
                BIS Catalogue
            </a>
            <div class="nav-status">
                ● System Online | 21,842 Active IS Codes Synced
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ========================================================
    # Header
    # ========================================================

    st.title("BIS AI Procurement Assistant")

    st.markdown(
        """
        An **AI-powered assistant** that reads your draft tender
        specification and recommends the correct **Bureau of Indian Standards (BIS)**
        code, its related references, and flags any **mandatory certification**
        requirements.
        """
    )

    vector_store = build_vector_store()

    # ========================================================
    # Input area
    # ========================================================

    st.subheader("Draft Tender Specifications")

    default_example = (
        "We require supply and installation of high early strength cement for repair "
        "of an airport runway during a 48-hour maintenance window. Product must gain "
        "significant compressive strength within 24 hours to permit reopening of the "
        "runway to aircraft traffic."
    )

    if "tender_text" not in st.session_state:
        st.session_state.tender_text = default_example
    if "last_uploaded_pdf" not in st.session_state:
        st.session_state.last_uploaded_pdf = None

    upload_col, lang_col = st.columns([2, 1])

    with upload_col:
        uploaded_pdf = st.file_uploader(
            "Upload a draft tender PDF (optional) - first 2 pages will auto-populate the text below",
            type=["pdf"],
            accept_multiple_files=False,
        )

    with lang_col:
        source_language = st.selectbox(
            "Input Language",
            options=SUPPORTED_LANGUAGES,
            index=0,
            help="Language of the tender text. Non-English input is auto-translated before search.",
        )

    if uploaded_pdf is not None:
        upload_signature = (uploaded_pdf.name, uploaded_pdf.size)
        if st.session_state.last_uploaded_pdf != upload_signature:
            try:
                extracted = extract_pdf_text(uploaded_pdf.getvalue(), max_pages=2)
                if extracted:
                    st.session_state.tender_text = extracted
                    st.session_state.last_uploaded_pdf = upload_signature
                    st.success(
                        f"Extracted {len(extracted.split())} words from the first 2 pages of "
                        f"**{uploaded_pdf.name}**."
                    )
                else:
                    st.warning(
                        "The PDF was read but no extractable text was found on the first 2 pages. "
                        "It may be a scanned image - OCR would be required."
                    )
            except Exception as exc: 
                st.error(f"Could not read the uploaded PDF: {exc}")

    tender_text = st.text_area(
        label="Paste or type the product / material description from your draft tender:",
        key="tender_text",
        height=200,
        placeholder="e.g., Supply of hot-rolled structural steel sections for bridge girders...",
    )

    # ========================================================
    # Advanced Settings
    # ========================================================

    with st.expander("⚙ Advanced Search Settings"):
        top_k = st.slider(
            "Number of standards to retrieve",
            min_value=1,
            max_value=3,
            value=2,
            help="How many matching standards to show in the results.",
        )

    col_run, _col_spacer = st.columns([1, 5])

    with col_run:
        submitted = st.button(
            "Analyze Tender",
            type="primary",
            use_container_width=True,
        )

    # ========================================================
    # Analysis & Output
    # ========================================================

    if submitted:
        if not tender_text.strip():
            st.error("Please enter a tender specification before analysis.")
            return

        query_text = tender_text

        if source_language != "English":
            st.info(
                f"Detected **{source_language}** input - translating to English before "
                "looking up Indian Standards...",
                icon="ℹ️",
            )
            query_text = translate_to_english(tender_text, source_language)

            with st.expander("Translated query"):
                st.write(query_text)

        with st.spinner("Analyzing tender and searching the BIS knowledge base..."):
            recommendations = recommend_standards(
                vector_store,
                query_text,
                top_k=top_k,
            )

        if not recommendations:
            st.error(
                "No matching Indian Standards were found. Please refine your query."
            )
            return

        st.success(
            f"Found {len(recommendations)} relevant Indian Standard(s)."
        )

        st.divider()
        render_recommendation(recommendations[0], is_primary=True)

        for extra in recommendations[1:]:
            st.divider()
            render_recommendation(extra, is_primary=False)

    # ========================================================
    # Footer
    # ========================================================

    st.markdown(
        """
        <div class="goi-footer">
            &copy; Bureau of Indian Standards &middot; Government of India &middot;
            BIS AI Procurement Assistant (Prototype) &middot; For official use only.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
