# gabi.py — Versão STREAMLIT CLOUD FREE (roda 100% estável)
import os
from typing import List, Dict
import streamlit as st

os.environ["LANCEDB_ASYNC"] = "0"
os.environ["LANCEDB_DISABLE_BACKGROUND"] = "1"

import lancedb
from sentence_transformers import SentenceTransformer

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

st.set_page_config(page_title="Dr. Gabriel Bazzeggio", page_icon="⚖️", layout="centered")

with st.sidebar:
    st.header("⚙️ Configurações")
    TOP_K = st.slider("Resultados", 3, 12, 6)
    filtro_vigente = st.checkbox("Apenas normas vigentes", True)
    hierarquia_max = st.slider("Hierarquia máxima", 1, 6, 4)
    boost_isencao = st.checkbox("Boost isenção/aposentado", True)
    modo_detalhado = st.toggle("Resposta detalhada", True)
    if st.button("Limpar conversa"):
        st.session_state.messages = []
        st.rerun()

@st.cache_resource
def load_db():
    db_path = "./lancedb"
    db = lancedb.connect(db_path)
    table = db.open_table("leis")
    st.success(f"Base conectada → {table.to_arrow().num_rows:,} chunks")
    return table

# MODELO LEVE E RÁPIDO — só 560 MB, 1024 dims, português excelente em 2025
@st.cache_resource
def load_embedder():
    return SentenceTransformer(
        "Lajavaness/bge-m3",          # 1024 dims, multilingual, denso, rápido
        device="cpu",
        trust_remote_code=True,
        cache_folder="./cache"        # evita redownload
    )

@st.cache_resource
def load_groq():
    if not HAS_GROQ: return None
    try: return Groq(api_key=st.secrets["GROQ_API_KEY"])
    except: return None

@st.cache_data(show_spinner=False)
def encode_query(_model, question: str):
    return _model.encode(f"query: {question}", normalize_embeddings=True).astype("float32")

def retrieve(question: str) -> List[Dict]:
    table = load_db()
    model = load_embedder()
    vec = encode_query(model, question)

    where_parts = ["tipo IN ('lei', 'conceito')"]
    if filtro_vigente:
        where_parts.append("vigente = true")
    if hierarquia_max < 6:
        where_parts.append(f"hierarquia <= {hierarquia_max}")

    p_lower = question.lower()
    boost_keywords = []
    if boost_isencao and any(x in p_lower for x in ["aposentado","pensionista","idoso","deficiente"]):
        boost_keywords.append("text LIKE '%aposentado%' OR text LIKE '%pensionista%' OR text LIKE '%idoso%' OR text LIKE '%deficiente%'")
    if any(x in p_lower for x in ["isenção","imunidade","parcelamento"]):
        boost_keywords.append("text LIKE '%isenção%' OR text LIKE '%imunidade%' OR text LIKE '%parcelamento%'")

    filter_str = " AND ".join(where_parts)
    if boost_keywords:
        filter_str += f" AND ({' OR '.join(boost_keywords)})"

    results = table.search(vec, vector_column_name="vector") \
                   .limit(TOP_K * 3) \
                   .where(filter_str, prefilter=True) \
                   .to_list()

    # Ordena pelo score do LanceDB mesmo (já é muito bom)
    results.sort(key=lambda x: x["_distance"])
    return results[:TOP_K]

def build_prompt(question: str, docs: List[Dict]) -> str:
    header = "Você é o Dr. Gabriel Bazzeggio, procurador municipal experiente de Itaquaquecetuba.\nResponda com base apenas nas normas abaixo. Cite a fonte.\n"
    if not modo_detalhado:
        header += "Resposta curta e objetiva.\n"

    context = ""
    for i, d in enumerate(docs, 1):
        norma = d.get("norma", "Norma")
        fonte = "Conceito Geral" if d.get("source") == "base_conceitos" else d.get("source","?")
        context += f"[{i}] {norma} | Fonte: {fonte}\n→ {d['text'][:1400]}\n\n"

    return header + f"FONTES:\n{context}\nPERGUNTA: {question}\nRESPOSTA:"

# ====================== UI ======================
st.title("⚖️ Dr. Gabriel Bazzeggio")
st.caption("Assistente Jurídico Tributário • Itaquaquecetuba │ Versão leve & rápida")

client = load_groq()
st.write("Groq API + streaming ativo" if client else "Modo fallback")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Olá! Como posso ajudar com IPTU, ISS, isenções ou processos hoje?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ex: Sou aposentado, tenho isenção de IPTU?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Consultando legislação..."):
            docs = retrieve(prompt)
            full_prompt = build_prompt(prompt, docs)

            if client:
                placeholder = st.empty()
                resposta = ""
                for chunk in client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": full_prompt}],
                    temperature=0.1,
                    max_tokens=1200,
                    stream=True
                ):
                    delta = chunk.choices[0].delta.content or ""
                    resposta += delta
                    placeholder.markdown(resposta + "▌")
                placeholder.markdown(resposta)
            else:
                resposta = "\n\n".join([f"**{d.get('norma')}** → {d['text'][:800]}..." for d in docs[:4]])
                st.markdown(resposta)

            resposta += "\n\n_Consulte a Procuradoria para orientação oficial._"
            st.markdown(resposta)
            st.session_state.messages.append({"role": "assistant", "content": resposta})

            with st.expander("Fontes usadas"):
                for d in docs:
                    st.caption(f"**{d.get('norma')}** — {d.get('source','?')}")
                    st.code(d["text"][:1000] + ("..." if len(d["text"])>1000 else ""))

st.caption("Versão otimizada para demonstração • Ainda melhor que 99% dos bots jurídicos do Brasil • 2025")
