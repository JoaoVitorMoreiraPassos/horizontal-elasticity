"""
Matrix Multiplication API with Flask and Prometheus

This Flask application provides an API endpoint for multiplying matrices
with caching for optimization and Prometheus metrics for monitoring.
"""

import time
import logging
import numpy as np
from functools import lru_cache
from asgiref.wsgi import WsgiToAsgi
from flask import Flask, request, jsonify
from flask_cors import CORS
from prometheus_flask_exporter import PrometheusMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
asgi_app = WsgiToAsgi(app)

# Enable CORS
CORS(app, resources={r"/multiply-matrices/": {"origins": ["http://topicos_nginx/", "http://topicos_proxy/", "http://topicos_prometheus/"]}})

# Enable Prometheus metrics
metrics = PrometheusMetrics(app)
metrics.info("app_info", "Matrix Multiplication API", version="1.0.0")

# Custom metrics
REQUEST_COUNT = metrics.counter(
    "requests_total", "Total number of requests", labels={"endpoint": lambda: request.endpoint}
)
LATENCY = metrics.histogram(
    "request_latency_seconds",
    "Request latency in seconds",
    labels={"endpoint": lambda: request.endpoint},
)


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
def cached_multiply(matrix_a_bytes: bytes, matrix_b_bytes: bytes, shape_b: tuple) -> np.ndarray:
    """
    Multiply matrices with caching.

    Args:
    - matrix_a_bytes (bytes): The first matrix in bytes.
    - matrix_b_bytes (bytes): The second matrix in bytes.
    - shape_b (tuple): Shape of the second matrix.

    Returns:
    - np.ndarray: The result of the matrix multiplication.
    """
    matrix_a = np.frombuffer(matrix_a_bytes, dtype=np.float64).reshape(-1, shape_b[0])
    matrix_b = np.frombuffer(matrix_b_bytes, dtype=np.float64).reshape(shape_b)
    return np.dot(matrix_a, matrix_b)


def multiply_matrices(matrix_a: np.ndarray, matrix_b: np.ndarray, cache: bool) -> np.ndarray:
    """
    Multiply matrices with optional caching.

    Args:
    - matrix_a (np.ndarray): The first matrix.
    - matrix_b (np.ndarray): The second matrix.
    - cache (bool): Flag to indicate whether to use caching for results.

    Returns:
    - np.ndarray: The result of the matrix multiplication.
    """
    logger.info("Starting matrix multiplication.")
    if cache:
        logger.info("Using caching for matrix multiplication.")
        key = matrices_to_tuple(matrix_a, matrix_b)
        try:
            result = cached_multiply(key[0], key[1], matrix_b.shape)
            logger.info("Result retrieved from cache.")
            return result
        except Exception as e:
            logger.warning(f"Cache miss or error: {e}. Computing result.")

    result = np.dot(matrix_a, matrix_b)
    logger.info("Matrix multiplication completed.")

    if cache:
        logger.info("Storing result in cache.")
        cached_multiply.cache_clear()  # Clear cache to prevent memory overflow
        cached_multiply(matrix_a.tobytes(), matrix_b.tobytes(), matrix_b.shape)  # Cache result

    return result


@app.route("/multiply-matrices/", methods=["POST"])
@REQUEST_COUNT
@LATENCY
def calculate_matrices():
    """
    API endpoint to receive a request for matrix multiplication and return the result.

    Returns:
    - response (dict): JSON-formatted dict representing the result and execution time.
    """
    start_time = time.time()
    logger.info("Received request for matrix multiplication.")

    data = request.get_json()

    if not data or "matrix_a" not in data or "matrix_b" not in data:
        logger.error("Invalid request. Missing 'matrix_a' or 'matrix_b'.")
        return jsonify({"error": "Invalid request. Provide 'matrix_a' and 'matrix_b'."}), 400

    # Extract matrices and cache flag
    try:
        matrix_a = np.array(data["matrix_a"], dtype=np.float64)
        matrix_b = np.array(data["matrix_b"], dtype=np.float64)
        cache = data.get("cache", False)
        logger.info("Matrices and cache flag extracted successfully.")
    except Exception as e:
        logger.error(f"Error parsing matrices: {e}")
        return jsonify({"error": "Invalid matrix data"}), 400

    # Validate matrix dimensions
    if matrix_a.shape[1] != matrix_b.shape[0]:
        logger.error("Matrix dimensions do not match for multiplication.")
        return jsonify({"error": "Matrix dimensions do not match for multiplication."}), 400

    # Perform matrix multiplication
    try:
        result = multiply_matrices(matrix_a, matrix_b, cache)
        logger.info("Matrix multiplication successful.")
    except Exception as e:
        logger.error(f"Internal server error during multiplication: {e}")
        return jsonify({"error": "Internal server error"}), 500

    execution_time = time.time() - start_time
    logger.info(f"Matrix multiplication completed in {execution_time:.6f} seconds.")

    # Prepare response
    response = {
        "result": result.tolist(),
        "execution_time": execution_time,
    }
    return jsonify(response)


if __name__ == "__main__":
    logger.info("Starting Matrix Multiplication API...")
    app.run(host="0.0.0.0", port=8001)
