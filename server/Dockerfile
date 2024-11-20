# Dockerfile para o servidor 1
FROM python:3.10

WORKDIR /app

# Copie o código-fonte do servidor 1 para o contêiner
COPY app.py /app/

# Copie o arquivo requirements.txt e instale as dependências
COPY requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install -r ./requirements.txt

# EXPOSE 8080

# Comando para iniciar o servidor
CMD ["uvicorn", "app:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80"]