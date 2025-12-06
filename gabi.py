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
    
    schema = pa.schema([
        pa.field("texto", pa.string()),
        pa.field("vector", pa.fixed_size_list(pa.float32(), 384))
    ])

    try:
        table = db.create_table("docs", schema=schema, mode="create")
    except:
        table = db.open_table("docs")

    emb_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    if table.count_rows() == 0:
        st.warning("Populando base pela primeira vez...")
        textos = [
            "O contribuinte aposentado pode ter isenção de IPTU conforme legislação municipal vigente.",
            "O IPTU é calculado com base no valor venal do imóvel determinado pela Prefeitura.",
            "O ISS incide sobre a prestação de serviços e segue regras do Código Tributário Municipal.",
            "ITBI é o imposto sobre transmissão inter vivos de bens imóveis por ato oneroso.",
            "A alíquota máxima de ISS é definida pela Lei Complementar 116/2003.",
            "Imóveis usados como templo religioso podem ter imunidade tributária.",
            "O lançamento do IPTU é de competência exclusiva do município."
447        ]

        data = [{"texto": t, "vector": emb_model.encode(t).astype("float32").tolist()} for t in textos]
        table.add(data)
        table.create_index(metric="cosine")  # índice simples já resolve
        st.success("Pronto! Dr. Gabriel está online.")

    return table, emb_model

table, emb_model = init_rag()

def busca_rag(pergunta, k=5):
    vec = emb_model.encode(pergunta).astype("float32").tolist()
    return table.search(vec, vector_column_name="vector").metric("cosine").limit(k).to_list()

# === UI ===
st.title("⚖️ Dr. Gabriel")
st.write("Assistente Tributário Municipal com RAG")

if "chat" not in st.session_state:
    st.session_state.chat = [{"role": "assistant", "content": "Olá! Em que posso ajudar com IPTU, ISS ou ITBI?"}]

for msg in st.session_state.chat:
    st.chat_message(msg["role"]).write(msg["content"])

if pergunta := st.chat_input("Faça sua pergunta tributária..."):
    st.session_state.chat.append({"role": "user", "content": pergunta})
    st.chat_message("user").write(pergunta)

    with st.spinner("Consultando leis..."):
        docs = busca_rag(pergunta)
        contexto = "\n\n".join([d["texto"] for d in docs])

        prompt = f"""Você é o Dr. Gabriel, procurador municipal especialista em tributos municipais.
Use SOMENTE as informações abaixo para responder.

PERGUNTA: {pergunta}

INFORMAÇÕES RELEVANTES:
{contexto}

Resposta direta e precisa:"""

        resp = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        resposta = resp.choices[0].message.content
        st.write(resposta)
        st.session_state.chat.append({"role": "assistant", "content": resposta})