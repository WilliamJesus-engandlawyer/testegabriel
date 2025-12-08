# gabi.py — Dr. Gabriel Bazzeggio
# VERSÃO FINAL 100% FUNCIONAL (local + Streamlit Cloud) — Dezembro 2025
import os
from typing import List, Dict
import streamlit as st

# ========================= FORÇA LANCEDB SÍNCRONO =========================
os.environ["LANCEDB_ASYNC"] = "0"
os.environ["LANCEDB_DISABLE_BACKGROUND"] = "1"

import lancedb
from sentence_transformers import SentenceTransformer

# Groq opcional (Streamlit Cloud)
try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

# ====================== CONFIGURAÇÃO PÁGINA ======================
st.set_page_config(page_title="Dr. Gabriel Bazzeggio", page_icon="⚖️", layout="wide")

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("⚙️ Configurações Avançadas")
    TOP_K = st.slider("Resultados retornados", 3, 15, 7)
    filtro_vigente = st.checkbox("Apenas normas vigentes", True)
    hierarquia_max = st.slider("Hierarquia máxima", 1, 6, 4, help="1=CF, 2=CTN, 4=Leis Municipais")
    boost_isencao = st.checkbox("Boost em isenção/aposentado", True)
    modo_detalhado = st.toggle("Resposta detalhada", True)
    st.markdown("---")
    if st.button("🗑️ Limpar conversa"):
        st.session_state.messages = []
        st.rerun()

# ====================== CARREGAMENTO ======================
@st.cache_resource
def load_db():
    try:
        # Funciona local e no Streamlit Cloud
        db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "lancedb"))
        if not os.path.exists(db_path):
            db_path = "./lancedb"  # fallback
        db = lancedb.connect(db_path)
        table = db.open_table("laws")
        st.success(f"Base conectada → {table.to_arrow().num_rows:,} chunks")
        return table
    except Exception as e:
        st.error(f"Erro ao carregar LanceDB: {e}")
        st.stop()

@st.cache_resource
def load_embedder():
    # MELHOR MODELO 2025 PARA DIREITO BRASILEIRO
    return SentenceTransformer("neuralmind/bert-base-portuguese-cased")

@st.cache_resource
def load_groq():
    if not HAS_GROQ:
        return None
    try:
        return Groq(api_key=st.secrets["GROQ_API_KEY"])
    except:
        return None

# ====================== BUSCA INTELIGENTE ======================
def retrieve(question: str) -> List[Dict]:
    table = load_db()
    model = load_embedder()
    vec = model.encode(question, normalize_embeddings=True).astype("float32")

    # Filtros base
    where_parts = []
    if filtro_vigente:
        where_parts.append("vigente = true")
    if hierarquia_max < 6:
        where_parts.append(f"hierarquia <= {hierarquia_max}")

    # Detecção automática de categoria
    p_lower = question.lower()
    if any(x in p_lower for x in ["recorrer", "recurso", "defesa", "prazo", "processo", "notificação", "administrativo"]):
        where_parts.append("categoria = 'Direito Formal'")
    else:
        where_parts.append("categoria = 'Direito Material'")  # default

    # Boost de palavras-chave (muito importante em tributário!)
    boost_keywords = []
    if boost_isencao and any(x in p_lower for x in ["aposentado", "pensionista", "idoso", "deficiente"]):
        boost_keywords.append("text LIKE '%aposentado%' OR text LIKE '%pensionista%' OR text LIKE '%idoso%' OR text LIKE '%deficiente%'")
    if "isenção" in p_lower or "imunidade" in p_lower:
        boost_keywords.append("text LIKE '%isenção%' OR text LIKE '%imunidade%'")
    if "parcelamento" in p_lower:
        boost_keywords.append("text LIKE '%parcelamento%'")

    filter_str = " AND ".join(where_parts)
    if boost_keywords:
        filter_str += f" AND ({' OR '.join(boost_keywords)})"

    # Busca híbrida (vector + texto)
    search = table.search(vec).metric("cosine").limit(TOP_K * 3)
    search = search.where(filter_str, prefilter=True)
    search = search.text(question)  # hybrid boost
    results = search.to_list()
    return results[:TOP_K]

# ====================== PROMPT ======================
def build_prompt(question: str, docs: List[Dict]) -> str:
    header = (
        "Você é o Dr. Gabriel Bazzeggio, procurador municipal experiente e ético de Itaquaquecetuba.\n"
        "Responda EXCLUSIVAMENTE com base nas normas abaixo. Seja claro, objetivo e cite a fonte.\n"
        "Se não houver previsão expressa, diga isso claramente.\n"
    )
    if not modo_detalhado:
        header += "Resposta curta e direta.\n"

    context = ""
    for i, d in enumerate(docs, 1):
        norma = d.get("norma", "Norma não identificada")
        fonte = os.path.basename(d.get("source_file", "arquivo"))
        cat = "Material" if d.get("categoria") == "Direito Material" else "Formal"
        texto = d["text"][:1300]
        context += f"[{i}] {norma} ─ {cat}\nFonte: {fonte}\n→ {texto}\n\n"

    return header + f"FONTES:\n{context}\nPERGUNTA: {question}\nRESPOSTA:"

# ====================== UI ======================
st.title("⚖️ Dr. Gabriel Bazzeggio")
st.subheader("Assistente Jurídico Tributário • Prefeitura de Itaquaquecetuba")
st.caption("RAG + neuralmind/bert-base-portuguese-cased + LanceDB + Groq • Dezembro 2025")

client = load_groq()
st.write("Groq API ativa (Llama 3.3 70B)" if client else "Modo local (resumo de trechos)")

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Olá! Em que posso ajudar com IPTU, ISS, ITBI, parcelamento ou isenções hoje?"
    }]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ex: Sou aposentado com 68 anos, tenho isenção total de IPTU?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando a legislação municipal..."):
            try:
                docs = retrieve(prompt)
                if not docs:
                    resposta = "Não encontrei informações relevantes nas normas carregadas."
                else:
                    full_prompt = build_prompt(prompt, docs)
                    if client:
                        resp = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[{"role": "user", "content": full_prompt}],
                            temperature=0.1,
                            max_tokens=1500
                        )
                        resposta = resp.choices[0].message.content
                    else:
                        resposta = "\n\n".join([
                            f"**{d.get('norma','Norma')}** → {d['text'][:900]}..."
                            for d in docs[:5]
                        ])

                    resposta += "\n\n_Consulte a Procuradoria Municipal para orientação oficial._"

                st.markdown(resposta)
                st.session_state.messages.append({"role": "assistant", "content": resposta})

                # Fontes no expander (transparência total)
                with st.expander("📚 Fontes consultadas"):
                    for d in docs:
                        st.caption(f"**{d.get('norma','Norma sem nome')}** — {d.get('categoria','Material')}")
                        st.code(d["text"][:1200] + ("..." if len(d["text"]) > 1200 else ""))

            except Exception as e:
                st.error("Erro interno. Tente novamente.")
                st.write(e)

st.markdown("---")
st.caption("Este assistente não substitui consulta jurídica oficial • Desenvolvido com ❤️ por Will • 2025")
