"""
gabi.py — Dr. Gabriel Bazzeggio
VERSÃO FINAL 100% FUNCIONAL NO STREAMLIT CLOUD (Dezembro 2025)
"""

import streamlit as st
import os
from typing import List, Dict

# ========================================
# FORÇA LANCEDB A RODAR 100% SÍNCRONO (OBRIGATÓRIO NO STREAMLIT CLOUD)
# ========================================
os.environ["LANCEDB_ASYNC"] = "0"           # ← ESSA LINHA É O SEGREDO
os.environ["LANCEDB_DISABLE_BACKGROUND"] = "1"

import lancedb
from sentence_transformers import SentenceTransformer

# Groq opcional
try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

st.set_page_config(page_title="Dr. Gabriel Bazzeggio", page_icon="⚖️", layout="wide")

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("Configurações")
    TOP_K = st.slider("Resultados", 3, 12, 6)
    filtro_vigente = st.checkbox("Apenas normas vigentes", True)
    hierarquia_max = st.slider("Hierarquia máxima (1=CF, 3=leis municipais)", 1, 6, 3)
    modo_detalhado = st.toggle("Resposta detalhada", True)
    st.markdown("---")
    if st.button("Limpar chat"):
        st.session_state.messages = []
        st.rerun()

# ====================== CARREGAMENTO ======================
@st.cache_resource
def load_db():
    try:
        db = lancedb.connect("./lancedb")           # ← SEM read_mode! Só a env var
        table = db.open_table("laws")
        st.success(f"Base conectada: {table.to_arrow().num_rows:,} registros")
        return table
    except Exception as e:
        st.error(f"Erro ao conectar no LanceDB: {e}")
        st.stop()

@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

@st.cache_resource
def load_groq():
    if not HAS_GROQ or "GROQ_API_KEY" not in st.secrets:
        return None
    try:
        return Groq(api_key=st.secrets["GROQ_API_KEY"])
    except:
        return None

# ====================== BUSCA ======================
def retrieve(question: str):
    table = load_db()
    vec = load_embedder().encode(question).astype("float32")

    where = []
    if filtro_vigente:
        where.append("vigente = true")
    if hierarquia_max:
        where.append(f"hierarquia <= {hierarquia_max}")
    filter_str = " AND ".join(where) if where else None

    search = table.search(vec).metric("cosine").limit(TOP_K * 2)
    if filter_str:
        search = search.where(filter_str, prefilter=True)
    try:
        search = search.text(question)   # hybrid search
    except:
        pass

    return search.to_list()[:TOP_K]

# ====================== PROMPT ======================
def build_prompt(question: str, docs: List[Dict]):
    header = (
        "Você é o Dr. Gabriel Bazzeggio, assistente jurídico tributário da Prefeitura de Itaquaquecetuba.\n"
        "Responda apenas com base nos trechos abaixo. Seja claro, objetivo e cite as fontes.\n"
        "Sempre termine com: 'Consulte a Procuradoria Municipal para orientação oficial.'\n\n"
    )
    if not modo_detalhado:
        header += "Responda de forma simples e curta.\n"

    context = ""
    for i, d in enumerate(docs, 1):
        norma = d.get("norma", "Norma")
        num_ano = f"{d.get('numero','')}/{d.get('ano','')}".strip("/")
        fonte = os.path.basename(d.get("source_file", "arquivo"))
        vigente = "Vigente" if d.get("vigente", True) else "Revogada"
        texto = d["text"][:1000] + ("..." if len(d["text"]) > 1000 else "")
        context += f"[{i}] {norma} {num_ano} ({vigente})\nFonte: {fonte}\n{texto}\n\n"

    return header + f"CONEXTO:\n{context}\nPERGUNTA: {question}\nRESPOSTA:"

# ====================== UI ======================
st.title("Dr. Gabriel Bazzeggio")
st.subheader("Assistente Jurídico Tributário • Itaquaquecetuba/SP")
st.caption("RAG + LanceDB + Groq • Dezembro 2025")

client = load_groq()
st.write("Groq API ativa" if client else "Groq indisponível → modo fallback")

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Olá! Como posso ajudar com IPTU, ISS, ITBI ou leis municipais hoje?"
    }]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

if prompt := st.chat_input("Digite sua dúvida (ex: isenção IPTU aposentado)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando as normas..."):
            try:
                docs = retrieve(prompt)
                if not docs:
                    resposta = "Não encontrei informações relevantes nas normas carregadas."
                else:
                    full_prompt = build_prompt(prompt, docs)
                    if client:
                        resp = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[{"role": "user", "content": full_prompt}],
                            temperature=0.1,
                            max_tokens=1200
                        )
                        resposta = resp.choices[0].message.content
                    else:
                        resposta = "\n\n---\n\n".join([
                            f"**{d.get('norma','Norma')}** ({os.path.basename(d.get('source_file',''))})\n{d['text'][:900]}..."
                            for d in docs
                        ])
                    resposta += "\n\n_Consulte a Procuradoria Municipal para orientação oficial._"
                st.markdown(resposta)
                st.session_state.messages.append({"role": "assistant", "content": resposta})
            except Exception as e:
                st.error("Erro interno. Tente novamente.")
                st.write(e)

st.markdown("---")
st.caption("Este assistente não substitui consulta jurídica oficial • Desenvolvido por Will • 2025")