"""Camada de ingestão e recuperação usando LlamaIndex e ChromaDB."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import chromadb
from llama_index.core import Document, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from app.config import Settings, get_settings


class RagStore:
    """Encapsula o índice vetorial persistente e seus métodos de consulta."""

    def __init__(self, settings: Settings) -> None:
        # Os embeddings são criados pelo LlamaIndex ao inserir ou consultar nós.
        embedding_args = {
            "model": settings.openai_embedding_model,
            "api_key": settings.openai_api_key,
        }
        if settings.openai_base_url:
            embedding_args["api_base"] = settings.openai_base_url
        self.embed_model = OpenAIEmbedding(**embedding_args)

        # O cliente persistente mantém a coleção no volume /data.
        client = chromadb.PersistentClient(path=str(settings.rag_data_dir))
        collection = client.get_or_create_collection(settings.rag_collection)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
            embed_model=self.embed_model,
        )

        # O splitter transforma documentos em unidades recuperáveis de tamanho controlado.
        self.splitter = SentenceSplitter(
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
        )

    def ingest(self, paths: list[str]) -> dict:
        """Lê arquivos permitidos, cria nós e os adiciona ao índice existente."""
        allowed_extensions = {".txt", ".md", ".pdf"}
        candidate_paths = [Path(value).resolve() for value in paths]
        valid_paths = [
            path for path in candidate_paths
            if path.is_file() and path.suffix.lower() in allowed_extensions
        ]
        skipped_paths = [str(path) for path in candidate_paths if path not in valid_paths]

        if not valid_paths:
            return {"indexed_chunks": 0, "files": [], "skipped": skipped_paths}

        documents: list[Document] = []
        for path in valid_paths:
            # Texto e Markdown são lidos diretamente para preservar UTF-8.
            if path.suffix.lower() in {".txt", ".md"}:
                documents.append(
                    Document(text=path.read_text(encoding="utf-8"), metadata={"source": str(path)})
                )
                continue

            # O leitor do LlamaIndex usa o suporte pypdf instalado para PDFs.
            pdf_documents = SimpleDirectoryReader(input_files=[str(path)]).load_data()
            for document in pdf_documents:
                document.metadata["source"] = str(path)
            documents.extend(pdf_documents)

        nodes = self.splitter.get_nodes_from_documents(documents)
        self.index.insert_nodes(nodes)
        return {
            "indexed_chunks": len(nodes),
            "files": [str(path) for path in valid_paths],
            "skipped": skipped_paths,
        }

    def retrieve(self, question: str, top_k: int) -> list[dict]:
        """Recupera os nós semanticamente mais próximos e normaliza a saída."""
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        results = retriever.retrieve(question)
        return [
            {
                "id": f"S{position}",
                "text": result.node.get_content(),
                "source": result.node.metadata.get("source", "desconhecida"),
                "score": round(float(result.score or 0), 4),
            }
            for position, result in enumerate(results, start=1)
        ]


@lru_cache
def get_rag_store() -> RagStore:
    """Inicializa o índice uma vez por processo para reutilizar o cliente Chroma."""
    return RagStore(get_settings())
