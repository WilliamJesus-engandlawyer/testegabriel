# ============================================================
# Assistente Tributário – Dr. Gabriel (RAG + Gemini)
# Versão otimizada para Streamlit Cloud
# ============================================================

import streamlit as st
import lancedb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai


# ---------------------- CONFIGURAÇÃO ----------------------
st.set_page_config(
    page_title="Dr. Gabriel – Assistente Tributário",
    page_icon="⚖️",
    layout="centered"
)

# Configura Gemini
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")

# ---------------------- CABEÇALHO ----------------------
st.title("⚖️ Dr. Gabriel")
st.write("**Procurador Municipal • OAB/SP**")
st.caption("Consultas tributárias sobre IPTU, ISS, ITBI e legislação municipal de Itaquaquecetuba.")

# ---------------------- CARREGAMENTO DO RAG ----------------------
@st.cache_resource
def load_rag():
    st.info("Carregando base jurídica e modelo de embeddings...")

    # 1) Conecta ao LanceDB
    try:
        db = lancedb.connect("./lancedb")
        st.success("✓ LanceDB conectado")
    except Exception as e:
        st.error(f"❌ Erro ao conectar ao LanceDB: {e}")
        return None, None

    # 2) Abre tabela
    try:
        tbl = db.open_table("laws")
        st.success("✓ Tabela 'laws' carregada")
    except Exception as e:
        st.error(f"❌ Erro ao abrir tabela 'laws': {e}")
        return None, None

    # 3) Carrega modelo de embeddings
    try:
        emb = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        st.success("✓ Modelo de embeddings carregado")
    except Exception as e:
        st.error(f"❌ Erro ao carregar modelo de embeddings: {e}")
        return None, None

    return tbl, emb


# Inicializa RAG
tbl, emb_model = load_rag()


# ---------------------- BUSCA RAG INTELIGENTE ----------------------
def busca_rag(pergunta, top_k=6):
    """Executa busca semântica com filtros e keyword-boost automático."""

    # Vetor da pergunta
    query_vec = emb_model.encode(pergunta, normalize_embeddings=True).astype("float32")

    p = pergunta.lower()
    keyword_boost = []

    # Boost para temas frequentes
    if any(x in p for x in ["isenção", "isencao", "imunidade"]):
        keyword_boost.append("text LIKE '%isen%' OR text LIKE '%imunid%'")

    if any(x in p for x in ["aposentado", "pensionista", "idoso"]):
        keyword_boost.append("text LIKE '%aposent%' OR text LIKE '%pension%' OR text LIKE '%idos%'")

    if "alíquota" in p or "aliquota" in p:
        keyword_boost.append("text LIKE '%aliquot%'")

    if "parcelamento" in p:
        keyword_boost.append("text LIKE '%parcel%'")

    # Busca semântica
    search = tbl.search(query_vec).metric("cosine").limit(top_k * 5)

    # Filtros básicos
    search = search.where("vigente = true AND hierarquia <= 3")

    # Keyword boost (prefilter rápido)
    if keyword_boost:
        search = search.where(" OR ".join(keyword_boost), prefilter=True)

    return search.to_list()[:top_k]


# ---------------------- CHATBOT ----------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Olá! Sou o Dr. Gabriel. Em que posso ajudar com tributos municipais hoje?"}
    ]

# Render histórico
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Entrada do usuário
if prompt := st.chat_input("Digite sua dúvida tributária aqui..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando legislação municipal..."):

            docs = busca_rag(prompt)

            # Monta o contexto jurídico
            contexto = "\n\n".join([
                f"FONTE: {d['norma']} {d.get('numero','')}/{d.get('ano','')} — {d.get('source_file','')}\n"
                f"{d['text'][:1200]}"
                for d in docs
            ])

            prompt_llm = f"""
Você é o Dr. Gabriel, procurador municipal especializado em tributação.

Responda APENAS com base no bloco "LEIS".
Nunca invente artigos ou números que não estejam no contexto.

PERGUNTA:
{prompt}

LEIS:
{contexto}

Escreva uma resposta formal e jurídica, iniciando com:
"Prezado(a) Cliente,"
"""

            # Gera resposta
            resposta = model.generate_content(prompt_llm)
            texto = resposta.text

            st.write(texto)

            st.session_state.messages.append(
                {"role": "assistant", "content": texto}
            )


# ---------------------- SIDEBAR ----------------------
with st.sidebar:
    st.header("ℹ️ Sobre o Assistente")
    st.write("""
    • IA com base em leis municipais  
    • RAG usando LanceDB  
    • Consultas sobre IPTU, ITBI, ISS, isenções e parcelamentos  
    """)
    st.caption("Criado em Dezembro/2025")
