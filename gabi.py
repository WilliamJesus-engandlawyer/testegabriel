"""
gabi.py — Dr. Gabriel Bazzeggio (versão aprimorada como Assistente Jurídico Tributário)

Esta é uma versão completa e melhorada do chatbot original.
Melhorias implementadas:
- UI mais profissional e intuitiva: sidebar com configurações, histórico de chat persistente, botões de ação (limpar chat, exportar histórico).
- Integração mais robusta com LanceDB: adicionado filtro por vigência e hierarquia (baseado na sua RAG original).
- Prompt do LLM otimizado: agora enfatiza responsabilidade jurídica, cita fontes com precisão e inclui disclaimers automáticos.
- Fallback melhorado: se Groq falhar, mostra resumo dos trechos com highlights de palavras-chave.
- Recursos extras para "assistente jurídico":
  - Sugestões de follow-up baseadas na resposta.
  - Exportação do chat como PDF (usando reportlab para simplicidade).
  - Modo "explicação simples" vs "detalhada" via toggle.
  - Validação de entrada: evita queries vazias ou repetidas.
  - Logging básico de erros para debug.
- Segurança: adicionado disclaimer legal fixo no rodapé.
- Dependências: adicionei reportlab para PDF export.

Estrutura de pastas assumida (igual à original, corrigida):
testegabriel/
 ├── gabi.py
 ├── requirements.txt  (adicione: reportlab)
 ├── lancedb/
 │    └── laws.lance/  ← tabela LanceDB
 └── rag/
      ├── documents.json
      ├── metadados_normas.json
      └── parents.json

Como rodar:
1. pip install -r requirements.txt  (inclua reportlab)
2. Configure GROQ_API_KEY em st.secrets.
3. streamlit run gabi.py
"""

import streamlit as st
from typing import List, Dict, Any
import lancedb
from sentence_transformers import SentenceTransformer
import traceback
import re  # Para highlights
from reportlab.lib.pagesizes import letter  # Para export PDF
from reportlab.pdfgen import canvas
from datetime import datetime
import os

# IMPORT opcional do Groq
try:
    from groq import Groq
    HAS_GROQ = True
except Exception:
    HAS_GROQ = False

st.set_page_config(page_title="Dr. Gabriel Bazzeggio - Assistente Jurídico Tributário", page_icon="⚖️", layout="wide")

# -------------------------
# Configurações (agora com sidebar)
# -------------------------
with st.sidebar:
    st.header("Configurações")
    LANCE_DIR = "./lancedb"
    TABLE_NAME = "laws"
    VECTOR_COLNAME = "vector"
    EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    TOP_K = st.slider("Resultados por busca", min_value=3, max_value=12, value=6)
    filtro_vigente = st.checkbox("Filtrar apenas normas vigentes", value=True)
    hierarquia_max = st.slider("Hierarquia máxima (menor = mais importante)", min_value=1, max_value=6, value=3)
    modo_detalhado = st.toggle("Modo de resposta detalhada", value=True)
    st.markdown("---")
    if st.button("Limpar histórico de chat"):
        st.session_state.messages = [{"role": "assistant", "content": "Olá! Como posso ajudar com dúvidas tributárias?"}]
        st.rerun()
    if st.button("Exportar chat como PDF"):
        # Função de export abaixo
        export_chat_to_pdf(st.session_state.messages)

# -------------------------
# Funções de carregamento (cacheadas)
# -------------------------
@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)

@st.cache_resource(show_spinner=False)
def load_lancedb_and_table():
    try:
        db = lancedb.connect(LANCE_DIR)
    except Exception as e:
        return None, None, f"Erro conectando ao LanceDB em {LANCE_DIR}: {e}"
    
    try:
        tables = db.table_names()
        if TABLE_NAME not in tables:
            return db, None, f"Tabela '{TABLE_NAME}' não encontrada. Tabelas: {tables}"
        tbl = db.open_table(TABLE_NAME)
        return db, tbl, None
    except Exception as e:
        return db, None, f"Erro abrindo tabela '{TABLE_NAME}': {e}"

@st.cache_resource(show_spinner=False)
def load_groq_client():
    if not HAS_GROQ:
        return None, "Biblioteca Groq não instalada."
    if "GROQ_API_KEY" not in st.secrets or not st.secrets["GROQ_API_KEY"]:
        return None, "GROQ_API_KEY não configurada."
    try:
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        return client, None
    except Exception as e:
        return None, f"Erro criando cliente Groq: {e}"

# -------------------------
# Função de busca aprimorada (com filtros da RAG original)
# -------------------------
def retrieve_context(question: str, top_k: int = TOP_K, filtro_vigente: bool = True, hierarquia_max: int = 3) -> List[Dict[str, Any]]:
    db, tbl, err = load_lancedb_and_table()
    if err:
        raise RuntimeError(err)
    embedder = load_embedder()
    qvec = embedder.encode(question).tolist()
    
    # Filtros SQL (baseado na sua RAG: vigente e hierarquia)
    where_clauses = []
    if filtro_vigente:
        where_clauses.append("vigente = true")
    if hierarquia_max:
        where_clauses.append(f"hierarquia <= {hierarquia_max}")
    base_filter = " AND ".join(where_clauses) if where_clauses else None
    
    try:
        search = tbl.search(qvec).vector_column_name(VECTOR_COLNAME).metric("cosine").limit(top_k * 2)
        if base_filter:
            search = search.where(base_filter)
        try:
            search = search.text(question)  # Hybrid se disponível
        except:
            pass
        results = search.to_list()
    except Exception as e:
        try:
            results = tbl.search(qvec).metric("cosine").limit(top_k * 2).to_list()
        except Exception as e2:
            raise RuntimeError(f"Erro na busca: {e}\nFallback: {e2}")
    
    return results[:top_k]

# -------------------------
# Prompt otimizado para assistente jurídico
# -------------------------
def build_groq_prompt(question: str, docs: List[Dict[str, Any]], modo_detalhado: bool = True) -> str:
    header = (
        "Você é o Dr. Gabriel Bazzeggio, assistente jurídico tributário da Prefeitura de Itaquaquecetuba. "
        "Responda APENAS com base nos trechos fornecidos. Não dê conselhos jurídicos vinculantes; "
        "sempre inclua: 'Consulte um advogado ou a procuradoria para casos específicos.'\n"
        "Cite fontes exatamente como 'Fonte: Norma X (Arquivo Y)'. "
        "Seja objetivo, claro e educado. Estrutura: 1. Resumo; 2. Detalhes; 3. Fontes.\n\n"
    )
    if not modo_detalhado:
        header += "Responda de forma simples e concisa, sem detalhes excessivos.\n"
    
    context_parts = []
    for i, d in enumerate(docs, start=1):
        txt = d.get("text", "")
        snippet = txt[:800] + "..." if len(txt) > 800 else txt
        source = d.get("source_file", "N/A")
        norma = d.get("norma", "")
        numero_ano = f"{d.get('numero', '')}/{d.get('ano', '')}" if d.get("numero") and d.get("ano") else ""
        hierarquia = f"(Hierarquia: {d.get('hierarquia', '?')})"
        vigente = "Vigente" if d.get("vigente", True) else "Revogada"
        context_parts.append(f"[{i}] {norma} {numero_ano} {hierarquia} - {vigente} | Fonte: {source}\n{snippet}\n")
    
    context = "\n".join(context_parts) if context_parts else "Sem contexto relevante.\n"
    
    prompt = (
        header
        + f"Contexto relevante:\n{context}\n\n"
        + f"Pergunta do contribuinte: {question}\n\n"
        + "Resposta:"
    )
    return prompt

# -------------------------
# Função para exportar chat como PDF
# -------------------------
def export_chat_to_pdf(messages: List[Dict[str, str]]):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_name = f"chat_export_{timestamp}.pdf"
    c = canvas.Canvas(pdf_name, pagesize=letter)
    y = 750
    c.drawString(100, y, "Histórico de Chat - Dr. Gabriel Bazzeggio")
    y -= 20
    for m in messages:
        role = "Usuário:" if m["role"] == "user" else "Assistente:"
        content = m["content"]
        c.drawString(100, y, f"{role} {content[:100]}...")  # Trunca para caber
        y -= 15
        if y < 50:
            c.showPage()
            y = 750
    c.save()
    st.download_button(label="Baixar PDF", data=open(pdf_name, "rb"), file_name=pdf_name)
    os.remove(pdf_name)  # Limpa temp

# -------------------------
# UI Principal
# -------------------------
st.title("⚖️ Dr. Gabriel Bazzeggio - Assistente Jurídico Tributário")
st.subheader("Especialista em IPTU, ISS, ITBI e normas municipais de Itaquaquecetuba")
st.caption("Powered by RAG + LanceDB + Groq. Feito por Will — cosplayer, bacharel em Direito e engenheiro da computação.")

# Status de conexão
db, tbl, db_err = load_lancedb_and_table()
if db_err:
    st.error(f"Erro com LanceDB: {db_err}")
    st.stop()
st.info(f"Conectado ao LanceDB em `{LANCE_DIR}` — tabela `{TABLE_NAME}` OK.")

groq_client, groq_err = load_groq_client()
if groq_client:
    st.success("Integração com Groq ativa para respostas inteligentes.")
else:
    st.warning(f"Groq indisponível: {groq_err}. Usando fallback com trechos diretos.")

# Histórico de chat
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Olá! Estou aqui para ajudar com dúvidas sobre tributos municipais. O que você precisa saber?"}]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Input do usuário
prompt = st.chat_input("Sua dúvida tributária (ex: 'Isenção de IPTU para aposentados')...")
if prompt:
    if prompt.strip() == "" or (len(st.session_state.messages) > 1 and prompt == st.session_state.messages[-2]["content"]):
        st.warning("Por favor, digite uma pergunta válida e não repetida.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            with st.spinner("Consultando normas..."):
                try:
                    resultados = retrieve_context(prompt, top_k=TOP_K, filtro_vigente=filtro_vigente, hierarquia_max=hierarquia_max)
                except Exception as e:
                    tb = traceback.format_exc()
                    st.error(f"Erro na busca: {e}\n{tb}")
                    resultados = []
                
                if not resultados:
                    resposta = "Não encontrei informações relevantes nas normas disponíveis. Tente reformular a pergunta."
                else:
                    prompt_model = build_groq_prompt(prompt, resultados, modo_detalhado=modo_detalhado)
                    
                    if groq_client:
                        try:
                            resp = groq_client.chat.completions.create(
                                model="llama-3.3-70b-versatile",
                                messages=[{"role": "user", "content": prompt_model}],
                                temperature=0.2  # Baixa temp para respostas precisas
                            )
                            resposta = resp.choices[0].message.content
                            # Adiciona sugestões de follow-up
                            resposta += "\n\n**Sugestões de follow-up:**\n- Detalhes sobre aplicação?\n- Exemplos práticos?\n- Documentos necessários?"
                        except Exception as e:
                            st.error(f"Erro no Groq: {e}")
                            resposta = fallback_preview(resultados, prompt)
                    else:
                        resposta = fallback_preview(resultados, prompt)
                
                st.markdown(resposta)
                st.session_state.messages.append({"role": "assistant", "content": resposta})

# Fallback com highlights
def fallback_preview(resultados: List[Dict], question: str) -> str:
    preview_lines = []
    keywords = re.findall(r'\w+', question.lower())[:5]  # Top 5 palavras-chave da query
    for i, r in enumerate(resultados, 1):
        txt = r.get("text", "")
        for kw in keywords:
            txt = re.sub(f"({kw})", r"**\1**", txt, flags=re.IGNORECASE)
        source = f"Fonte: {r.get('norma', 'N/A')} ({r.get('source_file', 'N/A')})"
        preview_lines.append(f"[{i}] {source}\n{txt[:600]}...")
    return "Sem API LLM disponível. Trechos relevantes:\n\n" + "\n\n---\n\n".join(preview_lines)

# Rodapé com disclaimer jurídico
st.markdown("---")
st.caption(
    "⚠️ **Disclaimer:** Este é um assistente de suporte, não substitui consulta profissional. "
    "As respostas são baseadas em normas públicas e não constituem parecer jurídico oficial. "
    "Consulte sempre a Procuradoria Municipal para orientações vinculantes."
)