# gabi.py — versão CORRIGIDA para Streamlit Cloud
import streamlit as st
import lancedb
from lancedb.pydantic import LanceModel, Vector
from groq import Groq
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Dr. Gabriel", page_icon="⚖️", layout="centered")
groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

@st.cache_resource
def load_db():
    db = lancedb.connect("db")

    class Document(LanceModel):
        texto: str
        vector: Vector(384)          # ← dimensão do modelo

    try:
        tbl = db.create_table("docs", schema=Document, mode="create")
    except Exception:
        tbl = db.open_table("docs")

    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    if tbl.count_rows() == 0:
        st.warning("Populando base pela primeira vez...")
        textos = [
            "O contribuinte aposentado pode ter isenção de IPTU conforme legislação municipal vigente.",
            "O IPTU é calculado com base no valor venal do imóvel determinado pela Prefeitura.",
            "O ISS incide sobre a prestação de serviços e segue regras do Código Tributário Municipal.",
            "ITBI é o imposto sobre transmissão inter vivos de bens imóveis por ato oneroso.",
            "A alíquota máxima de ISS é definida pela Lei Complementar 116/2003.",
            "Imóveis usados como templo religioso podem ter imunidade tributária.",
            "O lançamento do IPTU é de competência exclusiva do município."
        ]
        data = [Document(texto=t, vector=model.encode(t).tolist()) for t in textos]
        tbl.add(data)
        tbl.create_index(metric="cosine")       # índice simples já basta
        st.success("Base pronta!")

    return tbl, model

table, embedder = load_db()

def busca(pergunta, k=5):
    q_vec = embedder.encode(pergunta).tolist()
    # CORREÇÃO: especificar a coluna de vetores
    return table.search(q_vec).vector_column_name("vector").metric("cosine").limit(k).to_list()

# ================== UI ==================
st.title("⚖️ Dr. Gabriel")
st.caption("Assistente Tributário Municipal com RAG")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Em que posso ajudar com IPTU, ISS ou ITBI?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Sua dúvida tributária..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando leis..."):
            resultados = busca(prompt)
            contexto = "\n".join([r["texto"] for r in resultados]) if resultados else "Sem informações relevantes."

            resposta = groq_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{
                    "role": "user",
                    "content": f"Responda APENAS com base nestas leis:\n{contexto}\n\nPergunta: {prompt}"
                }],
                temperature=0.3
            )
            texto = resposta.choices[0].message.content
            st.write(texto)
            st.session_state.messages.append({"role": "assistant", "content": texto})