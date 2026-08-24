# Imagem enxuta com Python para executar o servidor MCP.
FROM python:3.12-slim

# Evita arquivos .pyc e faz logs chegarem imediatamente ao Docker.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Instala dependências antes de copiar o código para aproveitar o cache de camadas.
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copia somente o pacote da aplicação.
COPY app ./app

# Executa como usuário não privilegiado e prepara volumes de dados/documentos.
RUN useradd --create-home --uid 10001 appuser && \
    mkdir -p /data /workspace/docs && \
    chown -R appuser:appuser /app /data /workspace

USER appuser
EXPOSE 8000

# Inicia FastMCP com transporte SSE.
CMD ["python", "-m", "app.server"]
