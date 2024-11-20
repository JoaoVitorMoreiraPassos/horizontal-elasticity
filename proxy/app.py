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

class Server:
    """
    Represents a server with information about its status.

    Attributes:
    - name (str): The name of the server.
    - container_id (str): The container ID of the server.
    - cpu_percent (float): The CPU percentage of the server.
    - memory_used (int): The amount of used memory on the server.
    - memory_total (int): The total memory of the server.
    - memory_percent (float): The memory usage percentage of the server.
    - requisicoes_ativas (int): The number of active requests on the server.
    - requisicoes_totais (int): The total number of requests made to the server.
    - requisicoes_concluidas (int): The number of completed requests on the server.
    - sobre_carga (bool): Flag indicating whether the server is overloaded.
    - ocupado (bool): Flag indicating whether the server is busy.

    Methods:
    - monitor_overload(): Monitors and updates the server's overload status.
    """

    def __init__(self, name, url, container_id=None):
        self.name = name
        self.url = url
        self.container_id = container_id
        self.cpu_percent = 0.0
        self.memory_used = 0
        self.memory_total = 0
        self.memory_percent = 0.0
        self.requisicoes_ativas = 0
        self.requisicoes_totais = 0
        self.requisicoes_concluidas = 0
        self.sobre_carga = False
        self.ocupado = False
        self.docker_client = docker.from_env()
        logger.info(f"Server {self.name} initialized.")

    async def get_stats(self):
        """
        Retrieves the CPU and memory usage stats of the container.
        """
        try:
            # logger.info(f"Getting stats for server {self.name}.")
            container = self.docker_client.containers.get(self.name)
            stats = container.stats(stream=False)

            # Calculating CPU percentage
            cpu_delta = stats.get('cpu_stats', {}).get('cpu_usage', {}).get('total_usage', 0) - \
                        stats.get('precpu_stats', {}).get('cpu_usage', {}).get('total_usage', 0)
            system_delta = stats.get('cpu_stats', {}).get('system_cpu_usage', 0) - \
                           stats.get('precpu_stats', {}).get('system_cpu_usage', 0)

            if system_delta > 0 and cpu_delta > 0:
                num_cpus = stats.get('cpu_stats', {}).get('online_cpus', 1)  # Default to 1 CPU if 'online_cpus' is missing
                self.cpu_percent = (cpu_delta / system_delta) * num_cpus * 100.0
            else:
                self.cpu_percent = 0.0

            # Memory usage
            memory_stats = stats.get('memory_stats', {})
            self.memory_used = memory_stats.get('usage', 0) / 1024 / 1024  # Convert to MB
            self.memory_total = memory_stats.get('limit', 1) / 1024 / 1024  # Convert to MB
            self.memory_percent = (self.memory_used / self.memory_total) * 100 if self.memory_total > 0 else 0.0
        
        except Exception as e:
            logger.error(f"Error getting stats for {self.name}: {e}")
            self.cpu_percent = 0.0
            self.memory_used = 0
            self.memory_total = 0
            self.memory_percent = 0.0

    async def monitor_overload(self):
        """
        Monitors and updates the server's overload status.
        """
        logger.info(f"Starting overload monitoring for server {self.name}.")
        while True:
            await self.get_stats()
            logger.info(f"server: {self.name} - cpu_percent: {self.cpu_percent} - memory_percent: {self.memory_percent}")
            self.sobre_carga = self.cpu_percent > 80 or self.memory_percent > 80
            if self.sobre_carga:
                logger.warning(f"Server {self.name} is overloaded.")
            await asyncio.sleep(1)

    def __str__(self):
        return f"""
        name: {self.name}
        cpu_percent: {self.cpu_percent:.2f} %
        memory_used: {self.memory_used:.2f} MB
        memory_total: {self.memory_total:.2f} MB
        memory_percent: {self.memory_percent:.2f} %
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


servers = [
    Server("servidor1", "http://servidor1/"),
    Server("servidor2", "http://servidor2/"),
    Server("servidor3", "http://servidor3/"),
]

proxy_cpu_use = 0.0

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    """
    Initializes and starts monitoring tasks for server overload.
    """
    logger.info("Starting up and initializing server monitoring tasks.")
    for svr in servers:
        asyncio.create_task(svr.monitor_overload())
    asyncio.create_task(monitor_proxy_cpu())


origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


def select_server():
    """
    Selects the best available server based on CPU usage and overload status.
    """
    logger.info("Selecting the best available server.")
    available_servers = [svr for svr in servers if not svr.sobre_carga]
    if not available_servers:
        logger.warning("All servers are overloaded. Selecting the server with the lowest CPU usage.")
        return min(servers, key=lambda svr: svr.cpu_percent)
    selected_server = min(available_servers, key=lambda svr: svr.cpu_percent)
    logger.info(f"Selected server: {selected_server.name} with CPU usage: {selected_server.cpu_percent:.2f}%")
    return selected_server


@app.post("/")
async def multiply_matrices(request: MatricesRequest):
    """
    Handles matrix multiplication requests and forwards them to available servers.

    Args:
    - request (MatricesRequest): The matrix multiplication request.

    Returns:
    - JSON response containing the result of matrix multiplication.
    """
    server = select_server()

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
        server.requisicoes_ativas += 1
        server.ocupado = True
        logger.info(f"Forwarding request to server {server.name}. Active requests: {server.requisicoes_ativas}")
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url=f"{server.url}multiply-matrices/",
                json=request.dict(),
                headers={"Content-Type": "application/json"},
                timeout=60.0
            )
            response.raise_for_status()
            data = response.json()
        server.requisicoes_ativas -= 1
        server.requisicoes_totais += 1
        server.requisicoes_concluidas += 1
        server.ocupado = False
        logger.info(f"Request completed successfully by server {server.name}.")
        return data
    except httpx.HTTPError as e:
        server.requisicoes_ativas -= 1
        server.requisicoes_totais += 1
        server.ocupado = False
        logger.error(f"Error forwarding request to {server.name}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/get_infos")
async def get_cpu_info():
    """
    Retrieves CPU information of all servers and the proxy.

    Returns:
    - JSON response containing CPU information for all servers and the proxy.
    """
    logger.info("Retrieving CPU information for all servers and the proxy.")
    return {
        "server1": str(servers[0]),
        "server2": str(servers[1]),
        "server3": str(servers[2]),
        "proxy": proxy_cpu_use,
    }


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
