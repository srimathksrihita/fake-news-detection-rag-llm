"""
RAG-based Fake News Detector
Stack: sentence-transformers + FAISS + Groq (free LLM) + Streamlit
Author: Srimath Koilkandadai Srihita
"""

import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
import json
import time

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Fake News Detector",
    page_icon="🔍",
    layout="wide",
)

# ─── Knowledge Base ──────────────────────────────────────────────────────────
KNOWLEDGE_BASE = [
    {"text": "WHO has not endorsed any single food item as a cure for cancer. Cancer treatment requires medical intervention and varies by type and stage.", "source": "WHO Health Bulletin, 2023", "category": "Health"},
    {"text": "Coffee consumption has been linked to reduced risk of certain cancers in observational studies, but no causal relationship has been established. It does not cure cancer.", "source": "NIH Cancer Research Journal, 2022", "category": "Health"},
    {"text": "Climate change is real and primarily driven by human activity. This is confirmed by 97% of actively publishing climate scientists.", "source": "IPCC Sixth Assessment Report, 2021", "category": "Science"},
    {"text": "Vaccines do not cause autism. The original 1998 Wakefield paper was retracted from The Lancet due to fraud and ethical violations. Extensive research confirms no link.", "source": "Lancet retraction 2010; CDC 2023", "category": "Health"},
    {"text": "The 2020 US presidential election was not stolen. Over 60 court cases, including judges appointed by Trump, dismissed election fraud claims for lack of evidence.", "source": "AP Fact Check, Reuters 2021", "category": "Politics"},
    {"text": "5G technology does not spread COVID-19 or any virus. Viruses cannot travel on radio waves or mobile networks.", "source": "Full Fact, WHO 2020", "category": "Technology"},
    {"text": "India's GDP grew by approximately 8.2% in FY2023-24, making it one of the fastest growing major economies in the world.", "source": "Ministry of Statistics India, 2024", "category": "Economy"},
    {"text": "Drinking bleach or household disinfectants is extremely dangerous and can be fatal. It does not cure or prevent COVID-19.", "source": "CDC Emergency Alert, 2020", "category": "Health"},
    {"text": "The Earth is approximately 4.5 billion years old, based on radiometric dating of the oldest rocks and meteorites.", "source": "USGS Earth Science, 2023", "category": "Science"},
    {"text": "Ivermectin has not been proven effective against COVID-19 in large, well-designed randomized controlled trials.", "source": "NIH COVID Treatment Guidelines, 2022", "category": "Health"},
    {"text": "The Great Wall of China is not visible from space with the naked eye. This is a popular myth contradicted by astronaut testimony.", "source": "NASA Space Facts, 2021", "category": "Science"},
    {"text": "Elon Musk completed the acquisition of Twitter (now X) for approximately $44 billion in October 2022.", "source": "SEC Filing, Reuters 2022", "category": "Technology"},
    {"text": "ChatGPT was launched by OpenAI in November 2022 and reached 100 million users in two months, making it the fastest-growing consumer application in history.", "source": "OpenAI Blog, Reuters 2023", "category": "Technology"},
    {"text": "The COVID-19 pandemic was caused by the SARS-CoV-2 virus, first identified in Wuhan, China in late 2019. It was declared a pandemic by WHO in March 2020.", "source": "WHO Pandemic Report, 2020", "category": "Health"},
    {"text": "Microplastics have been found in human blood, lungs, and placentas. The health effects are still being studied by researchers worldwide.", "source": "Nature Medicine, 2022", "category": "Science"},
    {"text": "The Big Bang Theory is an American sitcom that aired on CBS from September 2007 to May 2019, spanning 12 seasons. It is not a Netflix original.", "source": "CBS Press Release, IMDb", "category": "Entertainment"},
    {"text": "Netflix licenses and produces content but does not own The Big Bang Theory. The show is owned by Warner Bros. Television and aired on CBS.", "source": "Netflix Investor Report, 2023", "category": "Entertainment"},
    {"text": "Friends is an American sitcom that aired on NBC from 1994 to 2004. Available on Netflix in some regions but not a Netflix original.", "source": "NBC, WarnerMedia 2023", "category": "Entertainment"},
    {"text": "Avengers: Endgame (2019) is the highest-grossing film of all time with over $2.79 billion worldwide, produced by Marvel Studios.", "source": "Box Office Mojo, 2023", "category": "Entertainment"},
    {"text": "Elon Musk is the CEO of Tesla and SpaceX. He acquired Twitter in 2022 and rebranded it to X. He is not the CEO of Apple or Microsoft.", "source": "Reuters, Bloomberg 2023", "category": "Technology"},
    {"text": "OpenAI was founded in 2015 by Sam Altman, Elon Musk, and others. Elon Musk departed from its board in 2018.", "source": "OpenAI Blog, 2023", "category": "Technology"},
    {"text": "India won the ICC Cricket World Cup in 1983 and 2011. The team is one of the most followed in global cricket.", "source": "ICC Records, BCCI 2023", "category": "Sports"},
    {"text": "The FIFA World Cup 2022 was held in Qatar. Argentina won, defeating France in the final via penalty shootout.", "source": "FIFA Official Records, 2022", "category": "Sports"},
    {"text": "NASA's Artemis I was an uncrewed lunar test flight completed in December 2022, part of the Artemis program to return humans to the Moon.", "source": "NASA Official, 2023", "category": "Science"},
    {"text": "The James Webb Space Telescope was launched on December 25 2021 and is operated by NASA, ESA, and CSA.", "source": "NASA JWST, 2022", "category": "Science"},
]

# ─── Model & Index (cached) ──────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading sentence embedding model (all-MiniLM-L6-v2)...")
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Building FAISS index...")
def build_index(_model):
    texts = [doc["text"] for doc in KNOWLEDGE_BASE]
    embeddings = _model.encode(texts, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)          # L2-normalise → cosine via inner product
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)          # Exact inner product search
    index.add(embeddings)
    return index


# ─── RAG Retrieval ───────────────────────────────────────────────────────────
def retrieve(query: str, model, index, k: int = 3) -> list:
    """Embed query → FAISS cosine search → top-k docs"""
    q_vec = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_vec)
    scores, indices = index.search(q_vec, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        doc = KNOWLEDGE_BASE[idx].copy()
        doc["similarity"] = float(score)
        results.append(doc)
    return results


# ─── LLM Classification (Groq — free) ───────────────────────────────────────
def classify_with_llm(query: str, evidence: list, api_key: str) -> dict:
    """Augment prompt with retrieved evidence → Groq LLaMA-3 → verdict"""
    client = Groq(api_key=api_key)

    context = "\n".join(
        [f"[Evidence {i+1}] {e['text']} (Source: {e['source']})"
         for i, e in enumerate(evidence)]
    )

    system_prompt = (
        "You are a fact-checking assistant using Retrieval-Augmented Generation (RAG). "
        "You receive a news claim and retrieved evidence from a trusted knowledge base. "
        "Classify the claim as REAL, FAKE, or UNCERTAIN.\n\n"
        "Respond ONLY with this JSON (no markdown backticks, no extra text):\n"
        '{"verdict": "REAL"|"FAKE"|"UNCERTAIN", '
        '"confidence": "High"|"Medium"|"Low", '
        '"explanation": "2-3 sentences citing the specific evidence"}'
    )

    user_prompt = (
        f'News claim: "{query}"\n\n'
        f"Retrieved evidence from knowledge base:\n{context}\n\n"
        "Classify the claim based on this evidence."
    )

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",     # free, fast model on Groq
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=400,
        temperature=0.1,                 # low temp for consistent JSON output
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


# ─── Streamlit UI ────────────────────────────────────────────────────────────
def main():
    st.title("🔍 Fake News Detector")
    st.markdown("**RAG Pipeline** · `all-MiniLM-L6-v2` Embeddings → FAISS Retrieval → Groq LLaMA-3 Classification")

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input(
            "Groq API Key (free)",
            type="password",
            help="Get free key at console.groq.com → API Keys",
        )
        st.caption("👆 Free at console.groq.com — no credit card needed")
        top_k = st.slider("Top-K evidence docs", min_value=1, max_value=5, value=3)

        st.markdown("---")
        st.markdown("**RAG Pipeline Steps:**")
        for step in [
            "1. User claim input",
            "2. Sentence embedding (384-dim)",
            "3. FAISS cosine search",
            "4. Top-K evidence retrieval",
            "5. LLM prompt augmentation",
            "6. Groq LLaMA-3 generation",
            "7. Verdict + explanation",
        ]:
            st.markdown(f"&nbsp;&nbsp;{step}")

        st.markdown("---")
        st.info(f"📚 Knowledge base: **{len(KNOWLEDGE_BASE)} facts** indexed")

    # ── Sample claims ────────────────────────────────────────────────────────
    SAMPLES = {
        "☕ Coffee cures cancer": "Scientists discover that drinking coffee every morning completely cures all types of cancer. The WHO has officially confirmed this revolutionary finding.",
        "📈 India GDP growth":    "India's economy grew at 8.2% in FY2023-24, positioning it as one of the fastest growing major economies globally.",
        "💉 Vaccines & autism":   "A major new study confirms vaccines cause autism in children, contradicting decades of medical consensus.",
        "🌡️ Climate change":      "Human-caused climate change is supported by 97% of climate scientists and backed by extensive peer-reviewed research.",
        "📱 5G spreads COVID":    "5G towers are responsible for spreading the COVID-19 virus across cities, as revealed by whistleblowers.",
    }

    st.subheader("Try a sample claim")
    cols = st.columns(len(SAMPLES))
    for col, (label, claim_text) in zip(cols, SAMPLES.items()):
        if col.button(label, use_container_width=True):
            st.session_state["claim_input"] = claim_text

    st.markdown("---")

    # ── Main input ───────────────────────────────────────────────────────────
    claim = st.text_area(
        "Enter a news claim to fact-check",
        value=st.session_state.get("claim_input", ""),
        height=100,
        placeholder="Paste any news headline or claim here…",
        key="claim_input",
    )

    analyze_btn = st.button("🔍 Analyze Claim", type="primary", use_container_width=True)

    if analyze_btn:
        if not claim.strip():
            st.warning("Please enter a claim to analyze.")
            return
        if not api_key:
            st.error("Please enter your Groq API key in the sidebar. Get it free at console.groq.com")
            return

        model = load_model()
        index  = build_index(model)

        # Step 1 — Retrieve
        with st.spinner("🔎 Retrieving relevant evidence from FAISS index..."):
            docs = retrieve(claim, model, index, k=top_k)

        st.subheader("📚 Retrieved Evidence (FAISS Top-K)")
        for i, doc in enumerate(docs):
            with st.expander(
                f"Evidence {i+1} — {doc['category']}  |  Similarity: {doc['similarity']:.3f}",
                expanded=True,
            ):
                st.markdown(f"> {doc['text']}")
                st.caption(f"📖 Source: {doc['source']}")
                st.progress(min(doc["similarity"], 1.0))

        # Warn if best similarity is too low — knowledge base gap
        best_similarity = docs[0]["similarity"] if docs else 0
        LOW_SIM_THRESHOLD = 0.20
        if best_similarity < LOW_SIM_THRESHOLD:
            st.warning(
                f"⚠️ **Low retrieval confidence** (best similarity: {best_similarity:.3f}) — "
                "the knowledge base has limited coverage for this claim. "
                "The LLM will still attempt a classification, but treat the result with caution. "
                "Consider adding relevant facts to the knowledge base."
            )

        # Step 2 — Classify
        with st.spinner("🤖 Sending to Groq LLaMA-3 with retrieved context..."):
            try:
                result = classify_with_llm(claim, docs, api_key)
            except json.JSONDecodeError as e:
                st.error(f"JSON parsing error (LLM returned unexpected format): {e}")
                return
            except Exception as e:
                st.error(f"LLM error: {e}")
                return

        # Verdict display
        verdict     = result.get("verdict", "UNCERTAIN")
        confidence  = result.get("confidence", "Low")
        explanation = result.get("explanation", "")
        icon = {"REAL": "✅", "FAKE": "❌", "UNCERTAIN": "⚠️"}.get(verdict, "⚠️")

        st.subheader("🏷️ Verdict")
        msg = f"{icon} **{verdict}** — {confidence} confidence\n\n{explanation}"
        if verdict == "REAL":
            st.success(msg)
        elif verdict == "FAKE":
            st.error(msg)
        else:
            st.warning(msg)

        # Full pipeline JSON
        with st.expander("🔬 Full RAG Pipeline Summary (JSON)"):
            st.json({
                "claim": claim,
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "vector_store": "FAISS IndexFlatIP (cosine similarity)",
                "top_k": top_k,
                "llm": "groq/llama-3.1-8b-instant",
                "retrieved_docs": [
                    {"text": d["text"][:80] + "...", "similarity": round(d["similarity"], 4)}
                    for d in docs
                ],
                "result": result,
            })


if __name__ == "__main__":
    main()
