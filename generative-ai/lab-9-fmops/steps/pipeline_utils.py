import boto3
import botocore
import json
import time
from datetime import datetime


PIPELINE_INSTANCE_TYPE = "ml.r5.xlarge"


SYSTEM_PROMPT = """You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. 
Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response."""


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


# template dataset to add prompt to each sample
def convert_to_messages(sample, system_prompt=""):
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": sample["Question"]},
        {"role": "assistant", "content": f"{sample['Complex_CoT']}\n\n{sample['Response']}"}
    ]

    sample["messages"] = messages
    
    return sample


def invoke_sagemaker_endpoint(payload, endpoint_name):
    """
    Invoke a SageMaker endpoint with the given payload.

    Args:
        payload (dict): The input data to send to the endpoint
        endpoint_name (str): The name of the SageMaker endpoint

    Returns:
        dict: The response from the endpoint
    """
    sm_client = boto3.client('sagemaker-runtime')
    try:
        start_time = time.time()
        response = sm_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        inference_time = time.time() - start_time
        
        response_body = response['Body'].read().decode('utf-8')
        return json.loads(response_body), inference_time
    except Exception as e:
        print(f"Error invoking endpoint {endpoint_name}: {str(e)}")
        return None, -1


def get_or_create_guardrail():
    guardrail_client = boto3.client('bedrock')
    guardrail_name = "ExampleMedicalGuardrail"

    guardrail_id = None
    guardrail_version = None
    
    # Try to get the guardrail
    response = guardrail_client.list_guardrails()
    for guardrail in response.get('guardrails', []):
        if guardrail['name'] == guardrail_name:
            guardrail_id = guardrail['id']
            response = guardrail_client.get_guardrail(
                guardrailIdentifier=guardrail_id
            )
            guardrail_version = response["version"]
            print(f"Found Guardrail {guardrail_id}:{guardrail_version}")
            break

    if not guardrail_id:
        try:
            guardrail = guardrail_client.create_guardrail(
                name="ExampleMedicalGuardrail",
                description='Example of a Guardrail for Medical Use Cases',
                topicPolicyConfig={
                    'topicsConfig': [{
                        'name': 'Block Pharmaceuticals',
                        'definition': 'This model cannot recommend one pharmaceutical over another. Generic prescriptions consistent with medical expertise and clinical diagnoses only.',
                        'type': 'DENY',
                        'inputAction': 'BLOCK',
                        'outputAction': 'BLOCK',
                    }]        
                },
                sensitiveInformationPolicyConfig={
                    'piiEntitiesConfig': [
                        {
                            'type': 'UK_NATIONAL_HEALTH_SERVICE_NUMBER',
                            'action': 'BLOCK',
                            'inputAction': 'BLOCK',
                            'outputAction': 'BLOCK'
                        },
                    ]
                },
                contextualGroundingPolicyConfig={
                    'filtersConfig': [
                        {
                            'type': 'RELEVANCE',
                            'threshold': 0.9,
                            'action': 'BLOCK',
                            'enabled': True
                        },
                    ]
                },
                blockedInputMessaging="ExampleMedicalGuardrail has blocked this input.",
                blockedOutputsMessaging="ExampleMedicalGuardrail has blocked this output."
            )
            
            guardrail_id = guardrail['guardrailId']
            guardrail_version = guardrail['version']
            
            print(f"Created new guardrail '{guardrail_id}:{guardrail_version}'")
        except botocore.exceptions.ClientError as create_error:
            print(f"Error creating guardrail: {create_error}")
            
    return guardrail_id, guardrail_version