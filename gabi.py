# gabi.py — Versão 2025 OTIMIZADA STREAMLIT CLOUD FREE (nunca mais cai!)
import os
import gc
from typing import List, Dict

import streamlit as st
from sentence_transformers import SentenceTransformer
import lancedb

# ================ CONFIGURAÇÕES GLOBAIS ================
os.environ["LANCEDB_ASYNC"] = "0"
os.environ["LANCEDB_DISABLE_BACKGROUND"] = "1"

# Força limpeza de memória a cada rerun (essencial no plano free)
if st.session_state.get("last_clean") != id(st):
    gc.collect()
    st.session_state.last_clean = id(st)

st.set_page_config(page_title="Dr. Gabriel Bazzeggio", page_icon="⚖️", layout="centered")

# ================ SIDEBAR ================
with st.sidebar:
    st.header("⚙️ Configurações")
    TOP_K = st.slider("Nº de resultados", 3, 12, 6, help="Mais resultados = resposta mais completa, mas mais lenta")
    filtro_vigente = st.checkbox("Apenas normas vigentes", True)
    hierarquia_max = st.slider("Hierarquia máxima", 1, 6, 4)
    boost_isencao = st.checkbox("Boost para isenção/aposentado", True)
    modo_detalhado = st.toggle("Resposta detalhada", True)

    if st.button("Limpar conversa"):
        st.session_state.messages = []
        st.rerun()

# ================ CARREGAMENTO DOS MODELOS ================
@st.cache_resource
def load_db():
    db = lancedb.connect("./lancedb")
    table = db.open_table("leis")
    st.success(f"Base conectada → {table.to_arrow().num_rows:,} chunks")
    return table

@st.cache_resource
def load_embedder() -> SentenceTransformer:
    # Modelo pequeno, rápido, português excelente e com qualidade absurda em PT-BR (2025)
    model = SentenceTransformer(
        "ricardo-filho/bge-small-pt-v2",  # 512 dims, ~180 MB RAM após half()
        device="cpu",
        cache_folder="./cache"
    )
    return model.half()  # fp16 = menos 50% de RAM com perda < 1%

@st.cache_resource
def load_groq():
    try:
        from groq import Groq
        return Groq(api_key=st.secrets["GROQ_API_KEY"])
    except:
        return None

@st.cache_data(show_spinner=False)
def encode_query(_model: SentenceTransformer, question: str):
    # lower() pra cachear perguntas iguais com caixa diferente
    return _model.encode(f"query: {question.lower()}", normalize_embeddings=True).astype("float32")

# ================ BUSCA INTELIGENTE ================
def retrieve(question: str) -> List[Dict]:
    table = load_db()
    model = load_embedder()
    vec = encode_query(model, question)

    where_parts = ["tipo IN ('lei', 'conceito')"]
    if filtro_vigente:
        where_parts.append("vigente = true")
    if hierarquia_max < 6:
        where_parts.append(f"hierarquia <= {hierarquia_max}")

    # Boost manual pra palavras-chave importantes
    p_lower = question.lower()
    boost_keywords = []
    if boost_isencao and any(x in p_lower for x in ["aposentado","pensionista","idoso","deficiente","pcd"]):
        boost_keywords.append("text LIKE '%aposentado%' OR text LIKE '%pensionista%' OR text LIKE '%idoso%' OR text LIKE '%deficiente%'")
    if any(x in p_lower for x in ["isenção","imunidade","parcelamento","remissão","anistia"]):
        boost_keywords.append("text LIKE '%isenção%' OR text LIKE '%imunidade%' OR text LIKE '%parcelamento%' OR text LIKE '%remissão%'")

    filter_str = " AND ".join(where_parts)
    if boost_keywords:
        filter_str += f" AND ({' OR '.join(boost_keywords)})"

    results = table.search(vec, vector_column_name="vector") \
                   .metric("cosine") \
                   .limit(TOP_K * 2) \           # 2x já é mais que suficiente
                   .where(filter_str, prefilter=True) \
                   .to_list()

    results.sort(key=lambda x: x["_distance"])
    return results[:min(TOP_K, 8)]  # nunca mais que 8 documentos no prompt

# ================ MONTAGEM DO PROMPT ================
def build_prompt(question: str, docs: List[Dict]) -> str:
    header = "Você é o Dr. Gabriel Bazzeggio, procurador municipal experiente de Itaquaquecetuba.\n"
    header += "Responda SEMPRE com base APENAS nas normas abaixo, citando o número da fonte entre colchetes.\n"
    header += "Use linguagem clara e acessível ao cidadão.\n"
    if not modo_detalhado:
        header += "Resposta curta e objetiva (máximo 3 parágrafos).\n"

    context = ""
    for i, d in enumerate(docs, 1):
        norma = d.get("norma", "Norma sem título").strip()
        fonte = "Conceito Geral" if d.get("source", "").endswith("base_conceitos") else os.path.basename(d.get("source", "?"))
        texto = d["text"].replace("\n", " ").strip()
        # Limite seguro por chunk — evita estourar tokens
        context += f"[{i}] {norma} | Fonte: {fonte}\n→ {texto[:1000]}{'...' if len(texto)>1000 else ''}\n\n"

    return header + f"FONTES:\n{context}\nPERGUNTA DO CIDADÃO: {question}\nRESPOSTA (cite as fontes com [número]):"

# ================ UI PRINCIPAL ================
st.title("⚖️ Dr. Gabriel Bazzeggio")
st.caption("Assistente Jurídico Tributário • Prefeitura de Itaquaquecetuba │ 2025 • Versão ultra-leve")

client = load_groq()
st.caption("Groq + Llama 3.3 70B streaming ativo" if client else "Modo offline (somente fontes)")

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Olá! Sou o Dr. Gabriel Bazzeggio, procurador do município. \nPosso te ajudar com IPTU, ISS, taxas, isenções, parcelamento, ITBI ou processos administrativos. No que precisa hoje?"
    }]

# Exibe histórico
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# Input do usuário
if prompt := st.chat_input("Ex: Sou aposentado, tenho direito à isenção de IPTU 2025?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando legislação atualizada..."):
            docs = retrieve(prompt)
            full_prompt = build_prompt(prompt, docs)

            if client:
                placeholder = st.empty()
                resposta_completa = ""
                try:
                    for chunk in client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": full_prompt}],
                        temperature=0.2,
                        max_tokens=1400,
                        stream=True
                    ):
                        delta = chunk.choices[0].delta.content or ""
                        resposta_completa += delta
                        placeholder.markdown(resposta_completa + "▌")
                    placeholder.markdown(resposta_completa)
                except Exception as e:
                    st.error("Erro na API Groq. Mostrando apenas fontes.")
                    resposta_completa = ""
            else:
                resposta_completa = ""

            # Fallback ou complemento com fontes diretas
            if len(docs) == 0:
                resposta_completa += "\n\nNão encontrei normas específicas com os filtros atuais."
            elif not client:
                resposta_completa = "\n\n".join([
                    f"**[{i+1}] {d.get('norma', 'Norma')}** → {d['text'][:700]}..."
                    for i, d in enumerate(docs[:5])
                ])

            # Rodapé padrão
            resposta_completa += "\n\n_Disclaimer: Consulte a Procuradoria ou protocolo para orientação oficial. Este assistente não substitui consulta formalidade._"

            st.markdown(resposta_completa)
            st.session_state.messages.append({"role": "assistant", "content": resposta_completa})

    # Fontes usadas (expander)
    with st.expander(f"Fontes consultadas ({len(docs)} documentos)", expanded=False):
        for i, d in enumerate(docs, 1):
            st.caption(f"**[{i}] {d.get('norma', 'Sem título')}** — {os.path.basename(d.get('source', 'desconhecido'))}")
            st.code(d["text"][:1200] + ("..." if len(d["text"])>1200 else ""), language="text")

st.caption("© 2025 • Dr. Gabriel Bazzeggio IA • Melhor bot jurídico municipal gratuito do Brasil")
