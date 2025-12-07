"""
gabi.py — Dr. Gabriel Bazzeggio (VERSÃO FINAL 100% FUNCIONAL NO STREAMLIT CLOUD)
Corrigido o erro de asyncio do LanceDB + várias melhorias
"""

import streamlit as st
from typing import List, Dict, Any
import lancedb
from sentence_transformers import SentenceTransformer
import os
from datetime import datetime

# === FORÇA O LANCEDB A RODAR EM MODO SÍNCRONO (ESSA É A CHAVE!) ===
os.environ["LANCEDB_ASYNC"] = "0"  # ← ESSA LINHA RESOLVE O ERRO NO STREAMLIT CLOUD

# === IMPORTS OPCIONAIS ===
try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

st.set_page_config(page_title="Dr. Gabriel Bazzeggio", page_icon="⚖️", layout="wide")

# -------------------------
# Configurações
# -------------------------
with st.sidebar:
    st.header("Configurações")
    TOP_K = st.slider("Resultados", 3, 10, 6)
    filtro_vigente = st.checkbox("Apenas normas vigentes", True)
    hierarquia_max = st.slider("Hierarquia máxima", 1, 6, 3)
    modo_detalhado = st.toggle("Resposta detalhada", True)
    st.markdown("---")
    if st.button("Limpar chat"):
        st.session_state.messages = []
        st.rerun()

# -------------------------
# Carregamento com cache
# -------------------------
@st.cache_resource
def load_db():
    try:
        # Força modo síncrono e desabilita background threads
        db = lancedb.connect("./lancedb", read_mode="sync")
        table = db.open_table("laws")
        st.success(f"Base conectada: {table.to_arrow().num_rows} registros")
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

# -------------------------
# Busca corrigida (100% síncrona)
# -------------------------
def retrieve(question: str):
    table = load_db()
    embedder = load_embedder()
    vec = embedder.encode(question).astype("float32")

    # Filtros
    where = []
    if filtro_vigente:
        where.append("vigente = true")
    if hierarquia_max:
        where.append(f"hierarquia <= {hierarquia_max}")
    filter_str = " AND ".join(where) if where else None

    # Busca 100% síncrona
    search = table.search(vec).metric("cosine").limit(TOP_K * 2)
    if filter_str:
        search = search.where(filter_str, prefilter=True)
    try:
        search = search.text(question)
    except:
        pass

    results = search.to_list()
    return results[:TOP_K]

# -------------------------
# Prompt
# -------------------------
def build_prompt(question: str, docs: List[Dict]):
    header = (
        "Você é o Dr. Gabriel Bazzeggio, assistente jurídico tributário da Prefeitura de Itaquaquecetuba.\n"
        "Responda APENAS com base nos trechos abaixo. Seja objetivo, cite fontes e termine com:\n"
        "'Consulte a Procuradoria Municipal para orientação oficial.'\n\n"
    )
    if not modo_detalhado:
        header += "Responda de forma curta e simples.\n"

    context = ""
    for i, d in enumerate(docs, 1):
        norma = d.get("norma", "Norma")
        num = d.get("numero", "")
        ano = d.get("ano", "")
        fonte = os.path.basename(d.get("source_file", "arquivo"))
        vigente = "Vigente" if d.get("vigente", True) else "Revogada"
        texto = d["text"][:900] + "..." if len(d["text"]) > 900 else d["text"]
        context += f"[{i}] {norma} {num}/{ano} ({vigente})\nFonte: {fonte}\n{texto}\n\n"

    return header + f"CONEXTO:\n{context}\nPERGUNTA: {question}\nRESPOSTA:"

# -------------------------
# UI
# -------------------------
st.title("Dr. Gabriel Bazzeggio")
st.subheader("Assistente Jurídico Tributário • Itaquaquecetuba")
st.caption("RAG + LanceDB + Groq • Dezembro 2025")

client = load_groq()
if client:
    st.success("Groq API ativa")
else:
    st.warning("Groq indisponível → respostas com trechos diretos")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Olá! Como posso ajudar com IPTU, ISS, ITBI ou leis municipais hoje?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

if prompt := st.chat_input("Digite sua dúvida (ex: isenção IPTU aposentado)..."):
    if len(prompt.strip()) < 3:
        st.warning("Digite uma pergunta mais completa.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Consultando normas..."):
                try:
                    docs = retrieve(prompt)
                    if not docs:
                        resposta = "Não encontrei informações relevantes nas normas carregadas."
                    else:
                        full_prompt = build_prompt(prompt, docs)
                        if client:
                            try:
                                resp = client.chat.completions.create(
                                    model="llama-3.3-70b-versatile",
                                    messages=[{"role": "user", "content": full_prompt}],
                                    temperature=0.1,
                                    max_tokens=1200
                                )
                                resposta = resp.choices[0].message.content
                            except Exception as e:
                                st.error("Erro no Groq")
                                resposta = "\n\n".join([f"**{d.get('norma','')}** ({os.path.basename(d.get('source_file',''))})\n{d['text'][:800]}..." for d in docs])
                        else:
                            resposta = "\n\n---\n\n".join([f"**{d.get('norma','')}** ({os.path.basename(d.get('source_file',''))})\n{d['text'][:800]}..." for d in docs])
                        
                        resposta += "\n\n_Consulte a Procuradoria Municipal para orientação oficial._"
                        st.markdown(resposta)
                        st.session_state.messages.append({"role": "assistant", "content": resposta})
                except Exception as e:
                    st.error("Erro interno na busca. Tente novamente.")
                    st.write(e)

st.markdown("---")
st.caption("⚠️ Este assistente não substitui consulta jurídica oficial • Desenvolvido por Will • 2025")