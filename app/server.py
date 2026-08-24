"""Ponto de entrada do servidor FastMCP com transporte SSE."""

from fastmcp import FastMCP

from app.graph import answer_question
from app.rag import get_rag_store

# Metadados e instruções são exibidos a clientes MCP compatíveis.
mcp = FastMCP(
    name="MCP RAG LangChain LangGraph LlamaIndex",
    instructions=(
        "Use ingest_documents para indexar documentos e ask_rag para responder "
        "perguntas exclusivamente a partir dos trechos indexados."
    ),
)


@mcp.tool()
def ingest_documents(paths: list[str]) -> dict:
    """Indexa arquivos .txt, .md e .pdf acessíveis ao processo.

    Args:
        paths: Caminhos absolutos, por exemplo /workspace/docs/manual.md.
    """
    result = get_rag_store().ingest(paths)
    result["message"] = (
        "Documentos indexados com sucesso."
        if result["indexed_chunks"]
        else "Nenhum arquivo válido foi indexado. Verifique caminhos e extensões."
    )
    return result


@mcp.tool()
def ask_rag(question: str, top_k: int = 5) -> dict:
    """Responde com RAG e lista as fontes recuperadas.

    Args:
        question: Pergunta em linguagem natural.
        top_k: Número de trechos recuperados, limitado ao intervalo de 1 a 10.
    """
    if not question.strip():
        return {"answer": "A pergunta não pode estar vazia.", "sources": []}

    # Impede consultas muito amplas e custo excessivo de contexto.
    return answer_question(question, max(1, min(top_k, 10)))


if __name__ == "__main__":
    # SSE é apropriado para clientes MCP remotos e ferramentas de desenvolvimento.
    mcp.run(transport="sse", host="0.0.0.0", port=8000)
