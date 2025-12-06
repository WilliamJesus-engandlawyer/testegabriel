# ============================================================
# gabi.py — RAG Tributário com LanceDB + GROQ (Streamlit Cloud)
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

    # Schema com fixed_size_list e nome 'vector'
    schema = pa.schema([
        pa.field("texto", pa.string()),
        pa.field("vector", pa.fixed_size_list(pa.float32(), 384))
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
            "O ISS incide sobre a prestação de serviços e segue regras do Código Tributário Municipal.",
            "ITBI é o imposto sobre transmissão inter vivos de bens imóveis por ato oneroso.",
            "A alíquota máxima de ISS é definida pela Lei Complementar 116/2003.",
            "Imóveis de entidades religiosas podem ter imunidade ao IPTU nos termos da Constituição Federal.",
            "O lançamento do IPTU pode ser contestado administrativamente em até 30 dias."
        ]

        data = []
        for txt in textos_iniciais:
            vec = emb_model.encode(txt).tolist()  # Converte para list diretamente
            data.append({"texto": txt, "vector": vec})

        table.add(data)

        # Cria índice explicitamente
        table.create_index(
            vector_column_name="vector",
            metric="cosine",
            index_type="IVF_PQ",
            num_partitions=2,  # Pequeno para MVP
            num_sub_vectors=4
        )

        st.success("Base populada e índice criado!")

    # Debug: Mostra o schema para verificar se 'vector' é reconhecido
    st.info(f"Schema da tabela: {str(table.schema)}")
    st.info(f"Número de rows: {table.count_rows()}")

    return table, emb_model


table, emb_model = init_rag()

# ------------------------------------------------------------
# FUNÇÃO DE BUSCA RAG
# ------------------------------------------------------------
def busca_rag(pergunta, k=5):
    vec = emb_model.encode(pergunta).tolist()
    # Sempre especifica vector_column_name para evitar erro de inferência
    try:
        result = table.search(
            vec,
            vector_column_name="vector"
        ).metric("cosine").limit(k).to_list()
        return result
    except Exception as e:
        st.error(f"Erro na busca: {str(e)}")
        return []


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

            if not docs:
                resposta_final = "Não encontrei informações relevantes na base. Tente reformular a pergunta."
            else:
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
                try:
                    resp = groq_client.chat.completions.create(
                        model="llama3-70b-8192",
                        messages=[{"role": "user", "content": prompt_llm}],
                        temperature=0.2  # Baixa para precisão
                    )
                    resposta_final = resp.choices[0].message.content
                except Exception as e:
                    resposta_final = f"Erro na API GROQ: {str(e)}"

            st.write(resposta_final)

            st.session_state.chat.append(
                {"role": "assistant", "content": resposta_final}
            )

# Botão para limpar chat e debug
if st.button("Limpar conversa e recarregar base"):
    st.session_state.chat = [{"role": "assistant", "content": "Em que posso ajudar hoje?"}]
    st.cache_resource.clear()
    st.rerun()