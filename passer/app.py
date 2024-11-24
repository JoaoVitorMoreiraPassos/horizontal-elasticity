"""
This app is a reverse proxy for the server.
"""

import asyncio
import httpx
import docker
import psutil
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Sevice:
    """
    
    """

    def __init__(self, name, url, container_id=None):
        self.name = name
        self.url = url
        self.container_id = container_id
        self.requisicoes_ativas = 0
        self.requisicoes_totais = 0
        self.requisicoes_concluidas = 0
        logger.info(f"Server {self.name} initialized.")

    def __str__(self):
        return f"""
        name: {self.name}
        requisicoes_ativas: {self.requisicoes_ativas}
        requisicoes_totais: {self.requisicoes_totais}
        requisicoes_concluidas: {self.requisicoes_concluidas}
        sobre_carga: {self.sobre_carga}
        """


class MatricesRequest(BaseModel):
    """
    Request model for matrix multiplication.

    Attributes:
    - matrix_a (List[List[float]]): The first matrix for multiplication.
    - matrix_b (List[List[float]]): The second matrix for multiplication.
    - cache (bool): Flag to indicate whether to use caching for results.
    """

    matrix_a: list[list[float]]
    matrix_b: list[list[float]]
    cache: bool


service = Sevice("topicos_servidor", "http://topicos_servidor/")


proxy_cpu_use = 0.0

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

@app.post("/")
async def multiply_matrices(request: MatricesRequest):
    """
    Handles matrix multiplication requests and forwards them to available servers.

    Args:
    - request (MatricesRequest): The matrix multiplication request.

    Returns:
    - JSON response containing the result of matrix multiplication.
    """

    # Input validation
    matrix_a = request.matrix_a
    matrix_b = request.matrix_b

    if len(matrix_a[0]) != len(matrix_b):
        logger.error("Invalid input: matrices cannot be multiplied due to incompatible dimensions.")
        raise HTTPException(status_code=400, detail="Invalid input: matrices cannot be multiplied")

    if not all(isinstance(x, (int, float)) for row in matrix_a for x in row):
        logger.error("Invalid input: matrix_a contains non-numeric values.")
        raise HTTPException(status_code=400, detail="Invalid input: matrix_a contains non-numeric values")

    if not all(isinstance(x, (int, float)) for row in matrix_b for x in row):
        logger.error("Invalid input: matrix_b contains non-numeric values.")
        raise HTTPException(status_code=400, detail="Invalid input: matrix_b contains non-numeric values")

    try:
        service.requisicoes_ativas += 1
        service.ocupado = True
        logger.info(f"Forwarding request to loadbalancer {service.name}. Active requests: {service.requisicoes_ativas}")
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url=f"{service.url}multiply-matrices/",
                json=request.dict(),
                headers={"Content-Type": "application/json"},
                timeout=60.0
            )
            response.raise_for_status()
            data = response.json()
        service.requisicoes_ativas -= 1
        service.requisicoes_totais += 1
        service.requisicoes_concluidas += 1
        service.ocupado = False
        logger.info(f"Request completed successfully by loadbalancer {service.name}.")
        return data
    except httpx.HTTPError as e:
        service.requisicoes_ativas -= 1
        service.requisicoes_totais += 1
        service.ocupado = False
        logger.error(f"Error forwarding request to {service.name}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


async def monitor_proxy_cpu():
    """
    Monitors and updates the proxy's own CPU usage.
    """
    global proxy_cpu_use
    logger.info("Starting proxy CPU monitoring.")
    while True:
        proxy_cpu_use = psutil.cpu_percent(interval=1)
        logger.info(f"Proxy CPU usage: {proxy_cpu_use:.2f}%")
        await asyncio.sleep(1)
