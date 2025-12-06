# ============================================================
# main.py — RAG Tributário com LanceDB + GROQ (Streamlit Cloud)
# ============================================================

import streamlit as st
import lancedb
import pyarrow as pa
from groq import Groq
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Dr. Gabriel – Assistente Tributário", page_icon="⚖️", layout="centered")
groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

@st.cache_resource
def init_rag():
    db = lancedb.connect("db")

    # Schema compatível com PyArrow novo
    embedding_field = pa.field("embedding", pa.list_(pa.float32(), list_size=384))
    schema = pa.schema([
        pa.field("texto", pa.string()),
        embedding_field
    ])

    try:
        table = db.create_table("docs", schema=schema, mode="create")
    except:
        table = db.open_table("docs")

    emb_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    if table.count_rows() == 0:
        st.warning("Base vazia — populando com dados iniciais...")
        textos = [
            "O contribuinte aposentado pode ter isenção de IPTU conforme legislação municipal vigente.",
            "O IPTU é calculado com base no valor venal do imóvel determinado pela Prefeitura.",
            "O ISS incide sobre a prestação de serviços e segue regras do Código Tributário Municipal.",
            "ITBI é o imposto sobre transmissão inter vivos de bens imóveis.",
            "A alíquota máxima de ISS é definida pela Lei Complementar 116/2003."
        ]
        data = [{"texto": t, "embedding": emb_model.encode(t).astype("float32").tolist()} for t in textos]
        table.add(data)
        st.success("Base populada!")

    return table, emb_model

table, emb_model = init_rag()

def busca_rag(pergunta, k=5):
    vec = emb_model.encode(pergunta).astype("float32")
    return table.search(vec).metric("cosine").limit(k).to_list()

# UI abaixo (igual ao anterior)
st.title("⚖️ Dr. Gabriel")
st.write("Assistente Tributário Municipal (IPTU, ISS, ITBI) com RAG")

if "chat" not in st.session_state:
    st.session_state.chat = [{"role": "assistant", "content": "Em que posso ajudar hoje?"}]

for msg in st.session_state.chat:
    st.chat_message(msg["role"]).write(msg["content"])

if pergunta := st.chat_input("Digite sua pergunta tributária..."):
    st.session_state.chat.append({"role": "user", "content": pergunta})
    st.chat_message("user").write(pergunta)

    with st.chat_message("assistant"):
        with st.spinner("Consultando base de leis..."):
            docs = busca_rag(pergunta, k=5)
            contexto = "\n\n".join([d["texto"] for d in docs])

            prompt = f"""Você é o Dr. Gabriel, procurador municipal especialista em IPTU, ISS e ITBI.
Use APENAS as informações abaixo.

PERGUNTA: {pergunta}

LEIS:
{contexto}

Responda com precisão e sem inventar informações."""

            resp = groq_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": prompt}]
            )
            resposta = resp.choices[0].message.content
            st.write(resposta)
            st.session_state.chat.append({"role": "assistant", "content": resposta})