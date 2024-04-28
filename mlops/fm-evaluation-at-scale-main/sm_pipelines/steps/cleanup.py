import boto3


def cleanup(model_id, endpoint_name):
    client = boto3.client('sagemaker')
    client.delete_endpoint(EndpointName=endpoint_name)
    client.delete_endpoint_config(EndpointConfigName=endpoint_name)
    return model_id
