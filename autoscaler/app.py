import time
import requests
import docker
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
PROMETHEUS_URL = 'http://topicos_prometheus:9090'
SERVICE_NAME = 'topicos_servidor'
CPU_UPPER_THRESHOLD = 50  # Percent
CPU_LOWER_THRESHOLD = 20  # Percent
MAX_REPLICAS = 50
MIN_REPLICAS = 3
SCALE_UP_STEP = 1
SCALE_DOWN_STEP = 1
CHECK_INTERVAL = 5  # Seconds

def get_cpu_usage():
    query = '''
    sum(rate(container_cpu_usage_seconds_total{container_label_com_docker_swarm_service_name="%s"}[1m])) * 100
    ''' % SERVICE_NAME

    response = requests.get(f'{PROMETHEUS_URL}/api/v1/query', params={'query': query})
    results = response.json()['data']['result']

    if results:
        return float(results[0]['value'][1])
    else:
        return 0.0

def get_current_replicas(client):
    service = client.services.get(SERVICE_NAME)
    return service.attrs['Spec']['Mode']['Replicated']['Replicas']

def set_replicas(client, replicas):
    service = client.services.get(SERVICE_NAME)
    service.update(mode={'Replicated': {'Replicas': replicas}})

def main():
    client = docker.DockerClient(base_url='unix://var/run/docker.sock')
    while True:
        try:
            cpu_usage = get_cpu_usage()
            logging.info(f'Current CPU usage: {cpu_usage:.2f}%')

            current_replicas = get_current_replicas(client)
            logging.info(f'Current replicas: {current_replicas}')

            current_average_usage = cpu_usage / current_replicas
            logging.info(f'Current average usage: {current_average_usage}')
            if current_average_usage > CPU_UPPER_THRESHOLD and current_replicas < MAX_REPLICAS:
                new_replicas = min(current_replicas + SCALE_UP_STEP, MAX_REPLICAS)
                set_replicas(client, new_replicas)
                logging.info(f'Scaled up to {new_replicas} replicas')

            elif current_average_usage < CPU_LOWER_THRESHOLD and current_replicas > MIN_REPLICAS:
                new_replicas = max(current_replicas - SCALE_DOWN_STEP, MIN_REPLICAS)
                set_replicas(client, new_replicas)
                logging.info(f'Scaled down to {new_replicas} replicas')

            else:
                logging.info('No scaling action needed.')

            time.sleep(CHECK_INTERVAL)
        except Exception as error:
            logging.error(error)

logging.info("isso foi chamado.")
if __name__ == '__main__':
    logging.info("chamou a main")
    main()
    