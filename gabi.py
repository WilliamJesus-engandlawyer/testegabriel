# ============================================================
# main.py — RAG Tributário com LanceDB + GROQ (Streamlit Cloud)
# ============================================================

import streamlit as st
import lancedb
import pyarrow as pa
from groq import Groq
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Dr. Gabriel", page_icon="⚖️", layout="centered")
groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

@st.cache_resource
def init_rag():
    db = lancedb.connect("db")
    schema = pa.schema([
        pa.field("texto", pa.string()),
        pa.field("vector", pa.fixed_size_list(pa.float32(), 384))
    ])
    try:
        table = db.create_table("docs", schema=schema, mode="create")
    except:
        table = db.open_table("docs")

    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    if table.count_rows() == 0:
        st.warning("Populando...")
        textos = [
            "O contribuinte aposentado pode ter isenção de IPTU conforme legislação municipal vigente.",
            "IPTU é calculado pelo valor venal do imóvel.",
            "ISS incide sobre prestação de serviços.",
            "ITBI é devido na compra e venda de imóvel.",
            "Alíquota máxima de ISS é 5% pela LC 116/2003."
        ]
        data = [{"texto": t, "vector": model.encode(t).astype("float32").tolist()} for t in textos]
        table.add(data)
        table.create_index(vector_column_name="vector", metric="cosine")
        st.success("Pronto!")

    return table, model

table, emb = init_rag()

def busca(pergunta, k=5):
    vec = emb.encode(pergunta).astype("float32").tolist()
    return table.search(query_vector=vec, vector_column_name="vector").metric("cosine").limit(k).to_list()

st.title("⚖️ Dr. Gabriel")
if "chat" not in st.session_state:
    st.session_state.chat = [{"role": "assistant", "content": "Em que posso ajudar?"}]

for m in st.session_state.chat:
    st.chat_message(m["role"]).write(m["content"])

if q := st.chat_input("Sua dúvida tributária..."):
    st.session_state.chat.append({"role": "user", "content": q})
    st.chat_message("user").write(q)

    with st.spinner("Pensando..."):
        docs = busca(q)
        ctx = "\n".join([d["texto"] for d in docs])

        resp = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": f"Responda SÓ com base nisso:\n{ctx}\n\nPergunta: {q}"}]
        ).choices[0].message.content

        st.write(resp)
        st.session_state.chat.append({"role": "assistant", "content": resp})
        


