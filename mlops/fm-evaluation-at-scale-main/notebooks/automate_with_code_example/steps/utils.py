from datetime import datetime
import boto3
import os

def endpoint_exists(endpoint_name):
    endpoint_exist = False

    client = boto3.client('sagemaker')
    response = client.list_endpoints()
    endpoints = response["Endpoints"]

    for endpoint in endpoints:
        if endpoint_name == endpoint["EndpointName"]:
            endpoint_exist = True
            break

    return endpoint_exist

def create_training_job_name(model_id):
    return f"{model_id}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')[:-3]}"


def safe_open_w(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w')


def write_to_file(input_str, file_path):
    with safe_open_w(file_path) as text_file:
        text_file.write(input_str)
