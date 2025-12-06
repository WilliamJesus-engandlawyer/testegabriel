# ============================================================
# main.py — RAG Tributário com LanceDB + GROQ (Streamlit Cloud)
# ============================================================

import streamlit as st
import lancedb
import pyarrow as pa
from groq import Groq
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------
# CONFIGURAÇÃO
# ------------------------------------------------------------
st.set_page_config(
    page_title="Dr. Gabriel – Assistente Tributário",
    page_icon="⚖️",
    layout="centered"
)

groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ------------------------------------------------------------
# INICIALIZAÇÃO DO LANCEDB + EMBEDDINGS
# ------------------------------------------------------------
@st.cache_resource
def init_rag():
    # Conecta ao banco
    db = lancedb.connect("db")

    # Schema correto
    schema = pa.schema([
        ("texto", pa.string()),
        pa.field("embedding", pa.fixed_size_list(pa.float32(), 384))  # Corrigi pra fixed_size_list (tamanho do modelo: 384 dims)
    ])

    # Cria tabela se não existir
    try:
        table = db.create_table("docs", schema=schema, mode="create")
    except:
        table = db.open_table("docs")

    # Carrega modelo de embeddings
    emb_model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # Se a tabela estiver VAZIA, faz autopopulação
    if table.count_rows() == 0:
        st.warning("Base vazia — populando tabela automaticamente...")

        textos_iniciais = [
            "O contribuinte aposentado pode ter isenção de IPTU conforme legislação municipal vigente.",
            "O IPTU é calculado com base no valor venal do imóvel determinado pela Prefeitura.",
            "O ISS incide sobre a prestação de serviços e segue regras do Código Tributário Municipal."
        ]

        # Coleta tudo numa lista e adiciona de uma vez (fix pro erro)
        data = []
        for txt in textos_iniciais:
            vec = emb_model.encode(txt).astype("float32")
            data.append({"texto": txt, "embedding": vec.tolist()})

        table.add(data)  # Agora sim: lista inteira

        st.success("Base populada com dados iniciais!")

    return table, emb_model


table, emb_model = init_rag()

# ------------------------------------------------------------
# FUNÇÃO DE BUSCA RAG
# ------------------------------------------------------------
def busca_rag(pergunta, k=5):
    vec = emb_model.encode(pergunta).astype("float32")
    result = table.search(vec).metric("cosine").limit(k).to_list()
    return result


# ------------------------------------------------------------
# CHAT UI
# ------------------------------------------------------------
st.title("⚖️ Dr. Gabriel")
st.write("Assistente Tributário Municipal (IPTU, ISS, ITBI) com RAG")

if "chat" not in st.session_state:
    st.session_state.chat = [
        {"role": "assistant", "content": "Em que posso ajudar hoje?"}
    ]

# Exibe histórico
for msg in st.session_state.chat:
    st.chat_message(msg["role"]).write(msg["content"])

# Entrada
if pergunta := st.chat_input("Digite sua pergunta tributária..."):
    st.session_state.chat.append({"role": "user", "content": pergunta})
    st.chat_message("user").write(pergunta)

    with st.chat_message("assistant"):
        with st.spinner("Consultando base de leis..."):

            # 1. Busca no RAG
            docs = busca_rag(pergunta, k=5)

            contexto = "\n\n".join([d["texto"] for d in docs])

            # 2. Prompt final
            prompt_llm = f"""
Você é o Dr. Gabriel, procurador municipal especialista em IPTU, ISS e ITBI.

Use APENAS as informações do bloco "LEIS".

PERGUNTA: {pergunta}

LEIS:
{contexto}

Responda com precisão jurídica, tom profissional e sem inventar leis inexistentes.
"""

            # 3. Gera resposta com GROQ
            resp = groq_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": prompt_llm}]
            )

            resposta_final = resp.choices[0].message["content"]

            st.write(resposta_final)

            st.session_state.chat.append(
                {"role": "assistant", "content": resposta_final}
            )