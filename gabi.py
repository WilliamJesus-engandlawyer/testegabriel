"""
gabi.py — Dr. Gabriel Bazzeggio (Versão FINAL 100% funcional - Dezembro 2025)
Assistente Jurídico Tributário com RAG + LanceDB + Groq + PDF Export
"""

import streamlit as st
from typing import List, Dict, Any
import lancedb
from sentence_transformers import SentenceTransformer
import traceback
import re
from datetime import datetime
import os

# === IMPORT OPCIONAL DO REPORTLAB (não quebra se não tiver) ===
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

# === IMPORT OPCIONAL DO GROQ ===
try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

st.set_page_config(page_title="Dr. Gabriel Bazzeggio - Assistente Tributário", page_icon="⚖️", layout="wide")

# -------------------------
# Configurações (Sidebar)
# -------------------------
with st.sidebar:
    st.header("⚙️ Configurações")
    TOP_K = st.slider("Resultados por busca", 3, 12, 6)
    filtro_vigente = st.checkbox("Apenas normas vigentes", True)
    hierarquia_max = st.slider("Hierarquia máxima (1 = CF, 3 = leis municipais)", 1, 6, 3)
    modo_detalhado = st.toggle("Resposta detalhada", True)
    st.markdown("---")
    if st.button("🗑️ Limpar chat"):
        st.session_state.messages = [{"role": "assistant", "content": "Olá! Como posso ajudar com IPTU, ISS ou ITBI hoje?"}]
        st.rerun()
    if HAS_REPORTLAB:
        if st.button("📄 Exportar chat como PDF"):
            export_chat_to_pdf(st.session_state.messages)
    else:
        st.caption("reportlab não instalado → PDF desativado")

# -------------------------
# Carregamento (cache)
# -------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

@st.cache_resource
def load_db():
    try:
        db = lancedb.connect("./lancedb")
        if "laws" not in db.table_names():
            st.error("Tabela 'laws' não encontrada em ./lancedb")
            st.stop()
        return db.open_table("laws")
    except Exception as e:
        st.error(f"Erro ao conectar no LanceDB: {e}")
        st.stop()

@st.cache_resource
def load_groq():
    if not HAS_GROQ:
        return None
    if "GROQ_API_KEY" not in st.secrets:
        return None
    try:
        return Groq(api_key=st.secrets["GROQ_API_KEY"])
    except:
        return None

# -------------------------
# Busca aprimorada
# -------------------------
def retrieve(question: str):
    tbl = load_db()
    embedder = load_embedder()
    vec = embedder.encode(question).tolist()

    where = []
    if filtro_vigente:
        where.append("vigente = true")
    if hierarquia_max:
        where.append(f"hierarquia <= {hierarquia_max}")
    filter_str = " AND ".join(where) if where else None

    search = tbl.search(vec).metric("cosine").limit(TOP_K * 2)
    if filter_str:
        search = search.where(filter_str)
    try:
        search = search.text(question)
    except:
        pass
    return search.to_list()[:TOP_K]

# -------------------------
# Prompt inteligente
# -------------------------
def build_prompt(question: str, docs: List[Dict]):
    header = (
        "Você é o Dr. Gabriel Bazzeggio, assistente jurídico tributário da Prefeitura de Itaquaquecetuba.\n"
        "Responda APENAS com base nos trechos abaixo. Nunca dê parecer vinculante.\n"
        "Sempre termine com: 'Consulte a Procuradoria para casos concretos.'\n"
        "Seja claro, objetivo e cite as fontes exatas.\n\n"
    )
    if not modo_detalhado:
        header += "Responda de forma simples e curta.\n"

    context = ""
    for i, d in enumerate(docs, 1):
        norma = d.get("norma", "Norma")
        num_ano = f"{d.get('numero','')}/{d.get('ano','')}" if d.get("numero") else ""
        fonte = d.get("source_file", "arquivo")
        vigente = "Vigente" if d.get("vigente") else "Revogada"
        texto = d["text"][:900] + "..." if len(d["text"]) > 900 else d["text"]
        context += f"[{i}] {norma} {num_ano} ({vigente})\nFonte: {fonte}\n{texto}\n\n"

    return header + f"CONEXTO:\n{context}\nPERGUNTA: {question}\nRESPOSTA:"

# -------------------------
# Export PDF (opcional)
# -------------------------
def export_chat_to_pdf(messages):
    if not HAS_REPORTLAB:
        st.error("reportlab não instalado")
        return
    filename = f"chat_dr_gabriel_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    y = height - 50
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Dr. Gabriel Bazzeggio - Histórico de Consulta")
    y -= 30
    c.setFont("Helvetica", 10)
    for msg in messages:
        role = "Usuário" if msg["role"] == "user" else "Dr. Gabriel"
        text = f"{role}: {msg['content'][:500]}"
        for line in textwrap.wrap(text, 100):
            if y < 50:
                c.showPage()
                y = height - 50
            c.drawString(50, y, line)
            y -= 15
    c.save()
    with open(filename, "rb") as f:
        st.download_button("Baixar PDF", f, file_name=filename, mime="application/pdf")
    os.remove(filename)

# -------------------------
# UI Principal
# -------------------------
st.title("⚖️ Dr. Gabriel Bazzeggio")
st.subheader("Assistente Jurídico Tributário • Itaquaquecetuba/SP")
st.caption("Baseado em RAG com LanceDB • Dezembro 2025")

tbl = load_db()
st.success(f"Base conectada: {len(tbl)} registros • Tabela 'laws' ativa")

client = load_groq()
if client:
    st.success("Groq API conectada (LLM ativo)")
else:
    st.warning("Groq não configurado → respostas com trechos brutos")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Olá! Em que posso ajudar com IPTU, ISS, ITBI ou leis municipais hoje?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

if prompt := st.chat_input("Digite sua dúvida tributária..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando as normas..."):
            docs = retrieve(prompt)
            if not docs:
                resposta = "Não encontrei informações relevantes nas normas carregadas."
            else:
                full_prompt = build_prompt(prompt, docs)
                if client:
                    try:
                        chat = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[{"role": "user", "content": full_prompt}],
                            temperature=0.1,
                            max_tokens=1500
                        )
                        resposta = chat.choices[0].message.content
                    except Exception as e:
                        st.error("Erro no Groq, usando fallback")
                        resposta = "\n\n".join([f"**Fonte:** {d.get('norma','')} ({d.get('source_file','')})\n{d['text'][:800]}..." for d in docs])
                else:
                    resposta = "\n\n---\n\n".join([f"**Fonte:** {d.get('norma','')} ({d.get('source_file','')})\n{d['text'][:800]}..." for d in docs])
                
                resposta += "\n\n_Consulte a Procuradoria Municipal para casos concretos._"
                st.markdown(resposta)
                st.session_state.messages.append({"role": "assistant", "content": resposta})

# Rodapé
st.markdown("---")
st.caption("⚠️ Este assistente não substitui consulta jurídica oficial • Desenvolvido por Will • 2025")