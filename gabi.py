# gabi.py — Dr. Gabriel Bazzeggio
# VERSÃO FINAL ESTÁVEL (Dezembro 2025) — multilingual-e5-large-instruct + Hybrid + Streaming + Rerank
import os
from typing import List, Dict
import streamlit as st

# ========================= FORÇA LANCEDB SÍNCRONO =========================
os.environ["LANCEDB_ASYNC"] = "0"
os.environ["LANCEDB_DISABLE_BACKGROUND"] = "1"

import lancedb
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np

# Groq opcional (Streamlit Cloud)
try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

# ====================== CONFIGURAÇÃO PÁGINA ======================
st.set_page_config(page_title="Dr. Gabriel Bazzeggio", page_icon="⚖️", layout="wide")

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("⚙️ Configurações Avançadas")
    TOP_K = st.slider("Resultados retornados", 3, 15, 8, help="Mais resultados = mais precisão")
    filtro_vigente = st.checkbox("Apenas normas vigentes", True)
    hierarquia_max = st.slider("Hierarquia máxima", 1, 6, 4, help="1=CF, 2=CTN, 4=Leis Municipais")
    boost_isencao = st.checkbox("Boost em isenção/aposentado", True)
    modo_detalhado = st.toggle("Resposta detalhada", True)
    usar_rerank = st.checkbox("Reranking avançado (mais preciso)", True)
    st.markdown("---")
    if st.button("🗑️ Limpar conversa"):
        st.session_state.messages = []
        st.rerun()

# ====================== CARREGAMENTO ======================
@st.cache_resource
def load_db():
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "lancedb"))
    if not os.path.exists(db_path):
        db_path = "./lancedb"
    db = lancedb.connect(db_path)
    try:
        table = db.open_table("leis")  # <-- agora é "leis", não "laws"
        st.success(f"Base conectada → {table.to_arrow().num_rows:,} chunks")
        return table
    except:
        st.error("Tabela 'leis' não encontrada. Verifique o nome do DB.")
        st.stop()

@st.cache_resource
def load_embedder():
    return SentenceTransformer(
        "intfloat/multilingual-e5-large-instruct",
        device="cpu",
        trust_remote_code=True
    )

@st.cache_resource
def load_reranker():
    if usar_rerank:
        return CrossEncoder("BAAI/bge-reranker-large", device="cpu")
    return None

@st.cache_resource
def load_groq():
    if not HAS_GROQ:
        return None
    try:
        return Groq(api_key=st.secrets["GROQ_API_KEY"])
    except:
        return None

# ====================== BUSCA INTELIGENTE ======================
@st.cache_data(show_spinner=False)
def encode_query(_model, question: str):
    return _model.encode(f"query: {question}", normalize_embeddings=True).astype("float32")

def retrieve(question: str) -> List[Dict]:
    table = load_db()
    model = load_embedder()
    vec = encode_query(model, question)

    # Filtros base
    where_parts = ["tipo IN ('lei', 'conceito')"]
    if filtro_vigente:
        where_parts.append("vigente = true")
    if hierarquia_max < 6:
        where_parts.append(f"hierarquia <= {hierarquia_max}")

    # Boost por palavras-chave
    p_lower = question.lower()
    boost_keywords = []
    if boost_isencao and any(x in p_lower for x in ["aposentado", "pensionista", "idoso", "deficiente", "deficiência"]):
        boost_keywords.append("text LIKE '%aposentado%' OR text LIKE '%pensionista%' OR text LIKE '%idoso%' OR text LIKE '%deficiente%'")
    if any(x in p_lower for x in ["isenção", "imunidade"]):
        boost_keywords.append("text LIKE '%isenção%' OR text LIKE '%imunidade%'")
    if "parcelamento" in p_lower:
        boost_keywords.append("text LIKE '%parcelamento%'")

    filter_str = " AND ".join(where_parts)
    if boost_keywords:
        filter_str += f" AND ({' OR '.join(boost_keywords)})"

    # Busca vetorial
    search = table.search(vec, vector_column_name="vector") \
                  .limit(TOP_K * 4) \
                  .where(filter_str, prefilter=True)

    results = search.to_list()

    # Reranking opcional (muito eficaz!)
    if usar_rerank and len(results) > 3:
        reranker = load_reranker()
        pairs = [[question, r["text"]] for r in results]
        scores = reranker.predict(pairs)
        for i, r in enumerate(results):
            r["rerank_score"] = float(scores[i])
        results.sort(key=lambda x: x["rerank_score"], reverse=True)

    return results[:TOP_K]

# ====================== PROMPT ======================
def build_prompt(question: str, docs: List[Dict]) -> str:
    header = (
        "Você é o Dr. Gabriel Bazzeggio, procurador municipal experiente e ético de Itaquaquecetuba.\n"
        "Responda EXCLUSIVAMENTE com base nas normas abaixo. Seja claro, objetivo e cite a fonte completa.\n"
        "Se não houver previsão expressa, diga isso claramente.\n"
    )
    if not modo_detalhado:
        header += "Resposta curta e direta.\n"

    context = ""
    for i, d in enumerate(docs, 1):
        norma = d.get("norma", "Norma não identificada")
        fonte = d.get("source", "arquivo")
        if fonte == "base_conceitos":
            fonte = "Conceito Tributário Geral"
        cat = "Conceito" if d.get("tipo") == "conceito" else "Normativo"
        texto = d["text"][:1500]
        context += f"[{i}] {norma} ─ {cat}\nFonte: {fonte}\n→ {texto}\n\n"

    return header + f"FONTES:\n{context}\nPERGUNTA: {question}\nRESPOSTA:"

# ====================== UI ======================
st.title("⚖️ Dr. Gabriel Bazzeggio")
st.subheader("Assistente Jurídico Tributário • Prefeitura de Itaquaquecetuba")
st.caption("RAG + multilingual-e5-large-instruct + LanceDB Hybrid 1024 + Groq Llama 3.3 70B • 2025")

client = load_groq()
st.write("Groq API ativa (Llama 3.3 70B com streaming)" if client else "Modo local (exibição de trechos)")

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Olá! Em que posso ajudar com IPTU, ISS, ITBI, isenções, parcelamento ou processo administrativo hoje?"
    }]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ex: Sou aposentado com 68 anos, tenho isenção total de IPTU em Itaquaquecetuba?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando a legislação municipal..."):
            try:
                docs = retrieve(prompt)
                if not docs:
                    resposta = "Não encontrei informações relevantes nas normas carregadas."
                else:
                    full_prompt = build_prompt(prompt, docs)

                    if client:
                        placeholder = st.empty()
                        resposta = ""
                        stream = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[{"role": "user", "content": full_prompt}],
                            temperature=0.1,
                            max_tokens=1500,
                            stream=True
                        )
                        for chunk in stream:
                            delta = chunk.choices[0].delta.content or ""
                            resposta += delta
                            placeholder.markdown(resposta + "▌")
                        placeholder.markdown(resposta)
                    else:
                        resposta = "\n\n".join([
                            f"**{d.get('norma','Norma')}** → {d['text'][:900]}..."
                            for d in docs[:5]
                        ])

                    resposta += "\n\n_Consulte a Procuradoria Municipal para orientação oficial._"
                    st.markdown(resposta)
                    st.session_state.messages.append({"role": "assistant", "content": resposta})

                    with st.expander("📚 Fontes consultadas"):
                        for d in docs:
                            fonte = "Conceito Geral" if d.get("source") == "base_conceitos" else d.get("source")
                            st.caption(f"**{d.get('norma','Norma sem nome')}** — {d.get('tipo','lei').title()}")
                            st.code(d["text"][:1300] + ("..." if len(d["text"]) > 1300 else ""))

            except Exception as e:
                st.error("Erro interno. Tente novamente.")
                st.exception(e)

st.markdown("---")
st.caption("Este assistente não substitui consulta jurídica oficial • Desenvolvido com ❤️ por Will • Dezembro 2025")
