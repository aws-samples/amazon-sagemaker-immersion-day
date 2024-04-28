import sagemaker
from sagemaker.jumpstart.model import JumpStartModel
from lib.utils import endpoint_exists
import boto3


def deploy(model_id, model_version, endpoint_name, instance_type="ml.g5.2xlarge", initial_instance_count=1):

    endpoint_exist = endpoint_exists(endpoint_name)
        
    if endpoint_exist:
       print("Endpoint already exists")
    else:
        my_model = JumpStartModel(model_id=model_id, model_version=model_version)
        predictor = my_model.deploy(initial_instance_count=initial_instance_count,
                                    instance_type=instance_type,
                                    serializer=sagemaker.serializers.JSONSerializer(),
                                    deserializer=sagemaker.deserializers.JSONDeserializer(), 
                                    endpoint_name=endpoint_name,
                                    accept_eula = True)
        #endpoint_name = predictor.endpoint_name
    
    return endpoint_name