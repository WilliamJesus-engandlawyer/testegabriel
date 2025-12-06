"""
gabi.py — Dr. Gabriel (versão final integrada com sua RAG / LanceDB)

IMPORTANTE LEMBRAR DA ORGANIZAÇÃO DAS PASTAS, QUANDO FOR MODIFICAR ALGO NO CÓDIGO 
testegabriel/
 ├── gabi.py
 ├── requirements.txt
 ├── rag/
 │    ├── laws.lance/       ← tabela LanceDB
 │    │     ├── data/
 │    │     ├── index/
 │    │     └── version
 │    ├── documents.json
 │    ├── metadados_normas.json
 │    └── parents.json



Como usar:
1. Tenha a pasta ./lancedb no mesmo nível deste arquivo (exportada do Colab).
2. Certifique-se de que a tabela dentro do lanceDB chama 'laws' (este script tenta abrir 'laws').
3. Configure a chave GROQ_API_KEY em Streamlit Cloud (opcional). Se não houver chave, o app mostra apenas os trechos encontrados.
4. rodar: streamlit run gabi.py
"""

import streamlit as st
from typing import List, Dict, Any, Tuple
import lancedb
from sentence_transformers import SentenceTransformer
import traceback

# IMPORT opcional do Groq — só se for usar a API para gerar texto final
try:
    from groq import Groq
    HAS_GROQ = True
except Exception:
    HAS_GROQ = False

st.set_page_config(page_title="Dr. Gabriel", page_icon="⚖️", layout="centered")

# -------------------------
# Configurações simples
# -------------------------
LANCE_DIR = "./lancedb"   # pasta exportada do Colab
TABLE_NAME = "laws"       # tabela criada pela Célula 6 do código da rag
VECTOR_COLNAME = "vector" # conforme pipeline da (Célula 6)
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 6                 # quantos resultados usar no prompt / exibir

# -------------------------
# Caches / carregamentos
# -------------------------
@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)

@st.cache_resource(show_spinner=False)
def load_lancedb_and_table():
    """
    Tenta conectar ao lanceDB local e abrir a tabela certa.
    Retorna (db, table) ou (None, None) em caso de erro.
    """
    try:
        db = lancedb.connect("./rag")
    except Exception as e:
        return None, None, f"db = lancedb.connect(/rag)"

    try:
        tables = db.table_names()
    except Exception as e:
        return db, None, f"Erro lendo tabelas: {e}"

    if TABLE_NAME not in tables:
        return db, None, f"Tabela '{TABLE_NAME}' não encontrada em {LANCE_DIR}. Tabelas presentes: {tables}"

    try:
        tbl = db.open_table(TABLE_NAME)
    except Exception as e:
        return db, None, f"Erro abrindo tabela '{TABLE_NAME}': {e}"

    return db, tbl, None

# Carrega Groq client se disponível e chave estiver presente
@st.cache_resource(show_spinner=False)
def load_groq_client():
    if not HAS_GROQ:
        return None, "biblioteca groq não instalada"
    if "GROQ_API_KEY" not in st.secrets or not st.secrets["GROQ_API_KEY"]:
        return None, "GROQ_API_KEY não configurada em st.secrets"
    try:
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        return client, None
    except Exception as e:
        return None, f"Erro criando cliente Groq: {e}"

# -------------------------
# Função de busca (usa o embedder para consulta)
# -------------------------
def retrieve_context(question: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """
    Cria embedding da pergunta e realiza busca vetorial na tabela.
    Retorna lista de dicts (cada dict ~ uma linha/documento do lanceDB).
    """
    db, tbl, err = load_lancedb_and_table()
    if err:
        raise RuntimeError(err)
    embedder = load_embedder()
    qvec = embedder.encode(question).tolist()

    # Monta a busca - especificamos o nome da coluna de vetor
    try:
        search = tbl.search(qvec).vector_column_name(VECTOR_COLNAME).metric("cosine").limit(top_k*4)
        # tenta usar full-text se suportado (melhora relevância)
        try:
            search = search.text(question)
        except Exception:
            pass
        results = search.to_list()
    except Exception as e:
        # API do LanceDB pode variar — tenta fallback sem especificar vector_column_name
        try:
            results = tbl.search(qvec).metric("cosine").limit(top_k*4).to_list()
        except Exception as e2:
            raise RuntimeError(f"Erro durante busca no LanceDB: {e}\nFallback: {e2}")

    # Retorna os top_k finais (já que pegamos mais para rerank/filtrar)
    return results[:top_k]

# -------------------------
# Função para montar o prompt para o modelo
# -------------------------
def build_groq_prompt(question: str, docs: List[Dict[str, Any]]) -> str:
    """
    Constrói um prompt conciso contendo trechos e referências.
    O Groq (ou outro LLM) deve responder "APENAS com base" no contexto.
    """
    header = (
        "Você é um assistente jurídico. Responda APENAS com base nos trechos das normas "
        "abaixo; não invente leis nem informações. Quando citar, indique 'Fonte: <arquivo>' "
        "e, se possível, o número/ano.\n\n"
    )
    context_parts = []
    for i, d in enumerate(docs, start=1):
        txt = d.get("text") or d.get("texto") or d.get("content") or ""
        # trunca para não explodir o prompt
        snippet = (txt[:1200] + "...") if len(txt) > 1200 else txt
        source = d.get("source_file") or d.get("arquivo") or d.get("norma") or "Fonte desconhecida"
        meta = []
        if d.get("norma"):
            meta.append(d.get("norma"))
        if d.get("numero") and d.get("ano"):
            meta.append(f"{d.get('numero')}/{d.get('ano')}")
        meta_str = " | ".join(meta) if meta else ""
        context_parts.append(f"[{i}] {meta_str} — Fonte: {source}\n{snippet}\n")

    context = "\n".join(context_parts) if context_parts else "Sem contexto disponível.\n"

    prompt = (
        header
        + f"Contexto (trechos relevantes):\n{context}\n\n"
        + f"Pergunta: {question}\n\n"
        + "Responda de forma clara e objetiva. Se não houver informação suficiente no contexto, diga que "
        "não é possível responder com segurança com base nas normas fornecidas."
    )
    return prompt

# -------------------------
# UI Streamlit
# -------------------------
st.title("⚖️ Dr. Gabriel — Assistente Tributário (RAG)")
st.caption("Base oficial: `./lancedb` (tabela 'laws'). Usando embeddings pré-gerados.")

# mostra status de conexão com a base
db, tbl, db_err = load_lancedb_and_table()
if db_err:
    st.error(f"Erro com LanceDB: {db_err}")
    st.stop()

st.info(f"Conectado ao LanceDB em `{LANCE_DIR}` — tabela `{TABLE_NAME}` encontrada.")

# carrega Groq (opcional)
groq_client, groq_err = load_groq_client()
if groq_client:
    st.success("Groq client pronto (respostas via API).")
else:
    st.warning(f"Groq indisponível: {groq_err}. O app irá retornar apenas os trechos encontrados como fallback.")

# estado do chat
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Em que posso ajudar sobre IPTU, ISS ou ITBI?"}]

for m in st.session_state.messages:
    st.chat_message(m["role"]).write(m["content"])

# input do usuário
prompt = st.chat_input("Sua dúvida tributária...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Buscando na base..."):
            try:
                resultados = retrieve_context(prompt, top_k=TOP_K)
            except Exception as e:
                tb = traceback.format_exc()
                st.error(f"Erro ao buscar contexto: {e}\n\n{tb}")
                resultados = []

            if not resultados:
                fallback_text = "Não encontrei trechos relevantes na base."
                st.write(fallback_text)
                st.session_state.messages.append({"role": "assistant", "content": fallback_text})
            else:
                prompt_for_model = build_groq_prompt(prompt, resultados)

                if groq_client:
                    try:
                        # usa a API de chat/completions do Groq
                        resp = groq_client.chat.completions.create(
                          model="llama-3.3-70b-versatile",
                            messages=[{"role": "user", "content": prompt_for_model}],
                            temperature=0.0
                        )
                        # extrai o texto da resposta
                        texto = resp.choices[0].message.content
                        st.write(texto)
                        st.session_state.messages.append({"role": "assistant", "content": texto})
                    except Exception as e:
                        # se der erro na API, mostra fallback com os trechos
                        st.error(f"Erro ao chamar Groq: {e}")
                        preview = "\n\n---\n\n".join([f"Fonte: {r.get('source_file', 'N/A')}\n{(r.get('text') or r.get('texto') or '')[:1200]}" for r in resultados])
                        st.write("FALHA AO GERAR RESPOSTA VIA API — mostrando trechos retornados:")
                        st.write(preview)
                        st.session_state.messages.append({"role": "assistant", "content": preview})
                else:
                    # fallback simples: mostra os snippets relevantes e indica que não há modelo configurado
                    preview_lines = []
                    for i, r in enumerate(resultados, 1):
                        src = r.get("source_file", "N/A")
                        num = r.get("numero") or ""
                        ano = r.get("ano") or ""
                        header = f"[{i}] Fonte: {src} {num}/{ano}"
                        snippet = (r.get("text") or "")[:1200]
                        preview_lines.append(f"{header}\n{snippet}...")
                    preview = "\n\n---\n\n".join(preview_lines)
                    st.write("Nenhuma API de LLM configurada — mostrando trechos relevantes:")
                    st.write(preview)
                    st.session_state.messages.append({"role": "assistant", "content": preview})

# rodapé com info técnica
st.markdown("---")
st.caption("Dr. Gabriel • RAG baseada no LanceDB exportado da minha pipeline.")

