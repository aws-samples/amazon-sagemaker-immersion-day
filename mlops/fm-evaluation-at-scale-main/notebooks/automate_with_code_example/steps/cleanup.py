import boto3


def cleanup(endpoint_name, *args):
    client = boto3.client('sagemaker')
    client.delete_endpoint(EndpointName=endpoint_name)
    client.delete_endpoint_config(EndpointConfigName=endpoint_name)
    return {"cleanup_done": True}
