# ============================================================
# Dr. Gabriel — Assistente Tributário com RAG + Groq
# Versão compatível com Streamlit Cloud
# ============================================================

import streamlit as st
import lancedb
import pyarrow as pa
from sentence_transformers import SentenceTransformer
from groq import Groq
import numpy as np


# ---------------------- CONFIG ----------------------
st.set_page_config(
    page_title="Dr. Gabriel – Assistente Tributário",
    page_icon="⚖️",
    layout="centered"
)

# Groq (configure no deploy: Secrets → GROQ_API_KEY)
client = Groq(api_key=st.secrets["GROQ_API_KEY"])


# ---------------------- CARREGAR LANCEDB + EMBEDDINGS ----------------------
@st.cache_resource
def load_rag():

    st.info("Carregando base legal e modelo de embeddings...")

    # Conecta ao LanceDB (cria pasta db/)
    db = lancedb.connect("db")

    # Schema compatível com pyarrow/streamlit cloud
    schema = pa.schema([
        ("texto", pa.string()),
        ("embedding", pa.list_(pa.float32()))
    ])

    # Cria tabela, ou abre caso já exista
    try:
        table = db.create_table(
            "docs",
            schema=schema,
            mode="create"
        )
    except:
        table = db.open_table("docs")

    # Carrega modelo de embeddings
    emb = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    return table, emb


table, emb_model = load_rag()


# ---------------------- Função de BUSCA RAG ----------------------
def busca_rag(query, k=5):

    vec = emb_model.encode(query).astype("float32")

    # Busca vetorial
    result = table.search(vec).metric("cosine").limit(k).to_list()

    return result


# ---------------------- Função LLM (Groq) ----------------------
def gerar_resposta(pergunta, leis):

    prompt = f"""
Você é o Dr. Gabriel, procurador municipal especialista em direito tributário local.

Use APENAS as leis fornecidas abaixo.  
Não invente artigos, não crie leis, não mencione normas inexistentes.

PERGUNTA:
{pergunta}

LEIS ENCONTRADAS:
{leis}

Responda como um procurador municipal, começando com:
"Prezado(a) munícipe,"

"""

    completion = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return completion.choices[0].message.content


# ---------------------- INTERFACE ----------------------
st.title("⚖️ Dr. Gabriel")
st.markdown(
    "Assistente Tributário especializado em **IPTU, ISS, ITBI**, "
    "legislação municipal e normas vigentes."
)


# ---------------------- HISTÓRICO ----------------------
if "chat" not in st.session_state:
    st.session_state.chat = [
        {"role": "assistant",
         "content": "Prezado(a) munícipe, em que posso ajudar hoje?"}
    ]


# Exibe histórico
for msg in st.session_state.chat:
    st.chat_message(msg["role"]).write(msg["content"])


# ---------------------- INPUT DO USUÁRIO ----------------------
pergunta = st.chat_input("Exemplo: aposentado tem isenção de IPTU?")

if pergunta:

    st.session_state.chat.append({"role": "user", "content": pergunta})
    st.chat_message("user").write(pergunta)

    with st.chat_message("assistant"):
        with st.spinner("Consultando legislação municipal..."):

            # 1. Busca RAG
            docs = busca_rag(pergunta, k=5)

            leis_texto = "\n\n".join(
                f"- {d['texto'][:1400]}"
                for d in docs
            )

            # 2. Resposta do Groq
            resposta = gerar_resposta(pergunta, leis_texto)

            st.write(resposta)

            st.session_state.chat.append(
                {"role": "assistant", "content": resposta}
            )


# ---------------------- SIDEBAR ----------------------
with st.sidebar:
    st.header("ℹ️ Sobre o projeto")
    st.markdown("""
    • RAG com **LanceDB + Sentence Transformers**  
    • LLM **Groq (Mixtral-8x7b)**  
    • Consultas reais sobre legislação municipal  
    • Ideal para Procuradorias, Prefeituras e atendimentos ao público  
    """)
    st.caption("Criado por William — Dez/2025")
