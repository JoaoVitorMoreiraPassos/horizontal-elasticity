from locust import HttpUser, TaskSet, task, between
import random

class UserBehavior(TaskSet):
    @task
    def multiply_matrices(self):
        # Define the request payload
        
        size = random.randint(50, 100)  # Matrices of size between 500x500 and 1000x1000

        # Generate random matrices with the specified size
        matrix_a = [[random.random() for _ in range(size)] for _ in range(size)]
        matrix_b = [[random.random() for _ in range(size)] for _ in range(size)]
        payload = {
            "matrix_a": matrix_a,
            "matrix_b": matrix_b,
            "cache": False
        }
        # Send a POST request to the matrix multiplication endpoint
        self.client.post("/", json=payload)

class WebsiteUser(HttpUser):
    tasks = [UserBehavior]
    wait_time = between(1, 5)  # Simulate user wait time between tasks
