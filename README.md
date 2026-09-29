# AI Procurement Assistant
### Autonomous Indian Standards (BIS) Recommendation & Compliance Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Orchestration-darkgreen.svg)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/Meta-FAISS-purple.svg)](https://github.com/facebookresearch/faiss)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)


An autonomous semantic recommendation and regulatory compliance engine designed for public sector procurement portals (such as GeM and CPPP). Built for Problem Statement **SIH26108**, this engine bridges the gap between natural language tender specifications and official Bureau of Indian Standards (BIS) codes.

---

## The Problem

Public procurement officers manually navigate over 21,000 active Indian Standards (IS codes). Conventional portal searches rely on rigid keyword matching, leading to:
- **Superseded Standards:** Inadvertently citing deprecated codes (e.g., specifying `IS 2062:2006` instead of `IS 2062:2011`), prompting vendor disputes and stalled tenders.
- **Omitted Normative References:** Missing mandatory secondary test methods (e.g., failing to link `IS 4031` testing alongside `IS 8041` rapid-hardening cement).
- **Overlooked Statutory QCOs:** Failing to mandate Quality Control Orders (QCOs) like mandatory ISI mark certifications or MeitY CRS compliance.

---

## Solution Architecture

The system replaces character-level keyword queries with an end-to-end Retrieval-Augmented Generation (RAG) pipeline:

1. **Document Ingestion & Preprocessing:** Extracts unstructured text from draft specifications or uploaded tender documents (PyPDF / OCR support).
2. **Vernacular Translation:** Ingestion hooks for regional languages (Hindi, Tamil, Bengali) to support state-level procurement workflows.
3. **Dense Vector Embeddings:** Encodes technical queries into 384-dimensional dense vectors using `sentence-transformers/all-MiniLM-L6-v2`.
4. **Vector Retrieval:** Runs Euclidean distance ($L_2$) similarity lookups against pre-indexed BIS standards inside an in-memory **FAISS** vector store in sub-second latency.
5. **Relevance Scoring:** Calibrates vector distances into intuitive percentage confidence metrics.
6. **Regulatory Guardrails:** Audits active vs. superseded status, attaches linked normative test methods, and flags statutory QCO certifications before tender publication.

---

## Tech Stack

- **Core Language:** Python 3.10+
- **RAG & Orchestration:** LangChain
- **Embeddings Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Database:** Meta FAISS (`faiss-cpu`)
- **Document Processing:** PyPDF / OCR
- **User Interface:** Streamlit
- **API Architecture:** Decoupled REST microservice pattern for GeM integration

---

## Sample Test Query

* **Input Specification:** `"High early strength cement for airport runway repair under heavy load"`
* **Direct Match Found:** `IS 8041:1996` (Rapid Hardening Portland Cement)
* **Confidence Score:** `92.4%`
* **Status:** `Active`
* **Linked Normative Testing:** `IS 4031 (Methods of physical tests for hydraulic cement)`
* **Regulatory Compliance:** Mandatory ISI Mark required under Cement QCO.

---

## License
Distributed under the MIT License. See `LICENSE` for more information.
