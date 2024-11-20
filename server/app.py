"""
Matrix Multiplication API

This FastAPI application provides an API endpoint for multiplying matrices with different processing types.
Matrices multiplication can be done sequentially, in parallel using threads, or in parallel using multiprocessing.
Results can be cached for faster retrieval.

"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from prometheus_client import Gauge, generate_latest, start_http_server
from typing import List
import numpy as np
import asyncio
from functools import lru_cache
import time
import concurrent.futures


app = FastAPI()

# Allow all origins
origins = ["http://proxy/"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class MatricesRequest(BaseModel):
    """
    Request model for matrix multiplication.

    Attributes:
    - matrix_a (List[List[float]]): The first matrix for multiplication.
    - matrix_b (List[List[float]]): The second matrix for multiplication.
    - cache (bool): Flag to indicate whether to use caching for results.
    """

    matrix_a: List[List[float]]
    matrix_b: List[List[float]]
    cache: bool



def matrices_to_tuple(matrix_a: np.ndarray, matrix_b: np.ndarray) -> tuple:
    """
    Convert matrices to a tuple of bytes for caching.

    Args:
    - matrix_a (np.ndarray): The first matrix.
    - matrix_b (np.ndarray): The second matrix.

    Returns:
    - tuple: A tuple containing the matrices as bytes.
    """
    return (matrix_a.tobytes(), matrix_b.tobytes())


@lru_cache(maxsize=128)
def cached_multiply(matrix_a_bytes: bytes, matrix_b_bytes: bytes) -> np.ndarray:
    """
    Multiply matrices with caching.

    Args:
    - matrix_a_bytes (bytes): The first matrix in bytes.
    - matrix_b_bytes (bytes): The second matrix in bytes.

    Returns:
    - np.ndarray: The result of the matrix multiplication.
    """
    matrix_a = np.frombuffer(matrix_a_bytes, dtype=np.float64).reshape(-1, matrix_b.shape[0])
    matrix_b = np.frombuffer(matrix_b_bytes, dtype=np.float64).reshape(matrix_b.shape)
    return np.dot(matrix_a, matrix_b)


async def multiply_matrices(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    cache: bool,
) -> np.ndarray:
    """
    Multiply matrices based on the specified processing type.

    Args:
    - matrix_a (np.ndarray): The first matrix.
    - matrix_b (np.ndarray): The second matrix.
    - cache (bool): Flag to indicate whether to use caching for results.

    Returns:
    - np.ndarray: The result of the matrix multiplication.
    """
    if cache:
        key = matrices_to_tuple(matrix_a, matrix_b)
        try:
            result = cached_multiply(key[0], key[1])
            return result
        except Exception:
            pass  # If not in cache, proceed to compute

    result = np.dot(matrix_a, matrix_b)

    if cache:
        cached_multiply.cache_clear()  # Clear cache to prevent memory overflow
        cached_multiply(key[0], key[1])  # Store result in cache

    return result


@app.post("/multiply-matrices/")
async def calculate_matrices(request: MatricesRequest):
    """
    API endpoint to receive a request for matrix multiplication and return the result.

    Args:
    - request (MatricesRequest): The request object containing matrices and processing parameters.

    Returns:
    - response (dict): JSON-formatted dict representing the result and execution time.
    """
    start_time = time.time()

    # Convert lists to NumPy arrays
    try:
        matrix_a = np.array(request.matrix_a, dtype=np.float64)
        matrix_b = np.array(request.matrix_b, dtype=np.float64)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid matrix data") from e

    # Perform matrix multiplication
    try:
        result = await multiply_matrices(
            matrix_a, matrix_b, request.cache
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve)) from ve
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error") from e

    execution_time = time.time() - start_time

    # Convert result to list for JSON serialization
    response = {
        "result": result.tolist(),
        "execution_time": execution_time,
    }
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
