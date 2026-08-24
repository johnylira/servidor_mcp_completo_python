"""Grafo LangGraph que recupera contexto, gera e valida uma resposta RAG."""

from __future__ import annotations

from functools import lru_cache
from typing import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from app.config import get_settings
from app.rag import get_rag_store


class RagState(TypedDict, total=False):
    """Estado compartilhado entre os nós do fluxo LangGraph."""

    question: str
    top_k: int
    contexts: list[dict]
    answer: str


def retrieve(state: RagState) -> RagState:
    """Obtém trechos relevantes no banco vetorial."""
    return {
        "contexts": get_rag_store().retrieve(
            state["question"], state.get("top_k", 5)
        )
    }


def generate(state: RagState) -> RagState:
    """Gera resposta limitada ao contexto recebido e pede citações explícitas."""
    contexts = state.get("contexts", [])
    if not contexts:
        return {"answer": "Não encontrei trechos relevantes no índice para responder a essa pergunta."}

    context_text = "\n\n".join(
        f"[{item['id']}] Fonte: {item['source']}\n{item['text']}"
        for item in contexts
    )
    settings = get_settings()
    model_args = {
        "model": settings.openai_model,
        "api_key": settings.openai_api_key,
        "temperature": 0,
    }
    if settings.openai_base_url:
        model_args["base_url"] = settings.openai_base_url

    model = ChatOpenAI(**model_args)
    prompt = (
        "Você é um assistente RAG. Responda em português exclusivamente com base no CONTEXTO. "
        "Se o contexto não contiver a resposta, diga que não há informação suficiente. "
        "Cada afirmação factual deve ter uma referência [S1] correspondente. "
        "Não invente fontes ou identificadores.\n\n"
        f"PERGUNTA:\n{state['question']}\n\nCONTEXTO:\n{context_text}"
    )
    response = model.invoke(prompt)
    content = response.content
    return {"answer": content if isinstance(content, str) else str(content)}


def validate(state: RagState) -> RagState:
    """Sinaliza respostas que não tenham qualquer citação recuperada válida."""
    answer = state.get("answer", "")
    contexts = state.get("contexts", [])
    valid_citations = {f"[{item['id']}]" for item in contexts}
    has_citation = any(citation in answer for citation in valid_citations)
    insufficient = "não há informação suficiente" in answer.lower()
    if contexts and not has_citation and not insufficient:
        answer += "\n\nObservação: a resposta não pôde ser vinculada de forma confiável aos trechos recuperados."
    return {"answer": answer}


@lru_cache
def get_rag_graph():
    """Compila e guarda o fluxo: retrieve -> generate -> validate."""
    workflow = StateGraph(RagState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    workflow.add_node("validate", validate)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "validate")
    workflow.add_edge("validate", END)
    return workflow.compile()


def answer_question(question: str, top_k: int = 5) -> dict:
    """Executa o grafo e retorna a resposta com metadados das fontes."""
    result = get_rag_graph().invoke({"question": question, "top_k": top_k})
    return {
        "answer": result["answer"],
        "sources": [
            {"id": item["id"], "path": item["source"], "score": item["score"]}
            for item in result.get("contexts", [])
        ],
    }
