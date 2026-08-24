# MCP RAG com Python, LangChain, LangGraph e LlamaIndex

Projeto de servidor **Model Context Protocol (MCP)** para indexar documentos e responder perguntas com RAG (Retrieval-Augmented Generation). O servidor usa FastMCP via SSE, LlamaIndex + ChromaDB para recuperação, LangGraph para orquestração e LangChain para geração.

## Arquitetura

```text
Cliente MCP -> SSE (/sse) -> FastMCP
                           -> ingest_documents(paths)
                           -> ask_rag(question)
                                  -> LangGraph
                                     -> recuperar contexto (LlamaIndex/Chroma)
                                     -> gerar resposta (LangChain/OpenAI)
                                     -> validar citações
```

## Início rápido

```bash
cp .env.example .env
# Edite .env e informe OPENAI_API_KEY
docker compose up --build
```

O endpoint MCP SSE fica em `http://localhost:8000/sse`.

## Documentos

Para tornar uma pasta local disponível ao contêiner, descomente o volume `./docs:/workspace/docs:ro` no `docker-compose.yml`. Em seguida, chame a ferramenta com um caminho como `/workspace/docs/manual.md`.

## Ferramentas MCP

### ingest_documents

Indexa arquivos `.txt`, `.md` e `.pdf`.

```json
{"paths": ["/workspace/docs/manual.md"]}
```

### ask_rag

Recupera os trechos mais relevantes e responde com referências `[S1]`, `[S2]`.

```json
{"question": "Quais são os procedimentos de backup?", "top_k": 5}
```

## Variáveis de ambiente

| Variável | Uso |
|---|---|
| `OPENAI_API_KEY` | Chave da API compatível com OpenAI |
| `OPENAI_MODEL` | Modelo de geração LangChain |
| `OPENAI_EMBEDDING_MODEL` | Modelo de embeddings LlamaIndex |
| `OPENAI_BASE_URL` | URL-base opcional de endpoint compatível |
| `RAG_DATA_DIR` | Diretório persistente do ChromaDB |
| `RAG_COLLECTION` | Nome da coleção vetorial |
| `RAG_CHUNK_SIZE` | Tamanho dos trechos indexados |
| `RAG_CHUNK_OVERLAP` | Sobreposição entre trechos |

## Segurança

Não exponha o SSE publicamente sem TLS e autenticação. Monte somente diretórios de documentos autorizados, pois a ferramenta de ingestão lê os caminhos informados.
