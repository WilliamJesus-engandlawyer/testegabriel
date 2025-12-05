import streamlit as st
import lancedb
import pandas as pd
from pypdf import PdfReader
from groq import Groq
from sentence_transformers import SentenceTransformer

# --- Config ---
st.set_page_config("RAG Groq")
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# --- Embeddings (locais, rápidos e estáveis) ---
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def embed(text):
    return embedder.encode(text).tolist()

# --- Conectar LanceDB ---
db = lancedb.connect("db")
table = db.create_table(
    "docs",
    schema={"texto": str, "embedding": list},
    mode="create",  # cria uma vez; ajuste para "append" se quiser reusar
)

# --- Função para ler PDFs ---
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# --- Indexação ---
st.header("📄 Indexar PDF")
uploaded_file = st.file_uploader("Envie um PDF", type=["pdf"])

if uploaded_file and st.button("Indexar"):
    text = extract_text_from_pdf(uploaded_file)
    chunks = text.split("\n\n")  # chunking simples

    rows = []
    for chunk in chunks:
        if chunk.strip():
            rows.append({
                "texto": chunk,
                "embedding": embed(chunk)
            })

    table.add(rows)
    st.success(f"{len(rows)} trechos indexados!")

# --- Perguntas ---
st.header("💬 Fazer Pergunta ao RAG")

query = st.text_input("Sua pergunta:")

if query:
    q_emb = embed(query)

    results = table.search(q_emb).limit(5).to_pandas()

    if len(results) == 0:
        st.warning("Nenhum documento encontrado.")
    else:
        context = "\n".join(results["texto"].tolist())

        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "Você é um assistente jurídico."},
                {"role": "user", "content": f"Contexto:\n{context}\n\nPergunta: {query}"}
            ]
        )

        answer = response.choices[0].message["content"]

        st.subheader("Resposta:")
        st.write(answer)

        with st.expander("🔎 Contexto usado"):
            st.write(context)
