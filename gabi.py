# ============================================================
# main.py — RAG Tributário no Streamlit (versão para Streamlit Cloud)
# ============================================================

import streamlit as st
import lancedb
from sentence_transformers import SentenceTransformer
from datetime import datetime
import google.generativeai as genai
import os




# ---------------------- CONFIGURAÇÃO ----------------------
st.set_page_config(
    page_title="Dr. Gabriel – Assistente Tributário",
    page_icon="⚖️",
    layout="centered"
)

# Configura a API do Gemini 
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")

# ---------------------- TÍTULO ----------------------
st.title("⚖️ Dr. Gabriel")
st.markdown(
    "**Procurador Municipal • OAB/SP**  \n"
    "Consultas tributárias sobre IPTU, ISS, ITBI e leis municipais de Itaquaquecetuba."
)

# ---------------------- CARREGAR RAG ----------------------
@st.cache_resource
def load_rag():
    st.info("Carregando base de leis e modelo de embeddings...")

    db = lancedb.connect("./lancedb")
    tbl = db.open_table("laws")
    model_emb = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    return tbl, model_emb


# Inicializa
tbl, model_emb = load_rag()


# ---------------------- FUNÇÃO DE BUSCA ----------------------
def busca_rag(pergunta, top_k=6):
    query_vec = model_emb.encode(
        pergunta, normalize_embeddings=True
    ).astype("float32")

    where_clauses = ["vigente = true", "hierarquia <= 3"]
    keyword_boost = []
    p_lower = pergunta.lower()

    search = tbl.search(query_vec).metric("cosine").limit(top_k * 5)
    search = search.where(" AND ".join(where_clauses))

    if keyword_boost:
        search = search.where(" OR ".join(keyword_boost), prefilter=True)

    return search.to_list()[:top_k]



# ---------------------- CHAT ----------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Em que posso ajudar com tributos municipais hoje?"}
    ]

# Exibe histórico
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Entrada do usuário
if prompt := st.chat_input("Ex: Aposentado tem isenção de IPTU?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando a legislação municipal..."):
            
            # 1. Busca no RAG
            docs = busca_rag(prompt, top_k=6)

            contexto = "\n\n".join([
                f"FONTE: {d['norma']} {d.get('numero','')}/{d.get('ano','')} — {d.get('source_file','')}\n{d['text'][:1500]}"
                for d in docs
            ])

            # 2. Monta prompt jurídico
            mensagem = f"""
Você é o Dr. Gabriel, procurador municipal especializado em tributação local.

Use APENAS as leis fornecidas no bloco "LEIS".  
Não invente artigos, números ou normas.

PERGUNTA DO USUÁRIO:
{prompt}

LEIS ENCONTRADAS:
{contexto}

Responda com precisão jurídica, começando com "Prezado(a) Cliente,".
"""

            # 3. Gera resposta com Gemini
            resposta = model.generate_content(mensagem)
            texto = resposta.text

            st.write(texto)

            st.session_state.messages.append(
                {"role": "assistant", "content": texto}
            )

# ---------------------- SIDEBAR ----------------------
with st.sidebar:
    st.header("ℹ️ Sobre este assistente")
    st.write("""
    • Baseado em leis municipais vigentes  
    • IA Gemini + RAG (LanceDB)  
    • Consultas sobre IPTU, ISS, ITBI, isenções e parcelamentos  
    """)
    st.caption("Criado em Dez/2025")
