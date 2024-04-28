import sagemaker
from sagemaker.jumpstart.model import JumpStartModel
from steps.utils import endpoint_exists


def deploy_llama7b(model):

    model_id = model["model_id"]
    model_version = model["model_version"]
    endpoint_name = model["endpoint_name"]
    instance_type = model["instance_type"]
    num_instances = model["num_instances"]

    if not endpoint_exists(endpoint_name):
        model = JumpStartModel(model_id=model_id, model_version=model_version)
        predictor = model.deploy(initial_instance_count=num_instances,
                                    instance_type=instance_type,
                                    serializer=sagemaker.serializers.JSONSerializer(),
                                    deserializer=sagemaker.deserializers.JSONDeserializer(),
                                    endpoint_name=endpoint_name,
                                    accept_eula=True)

    return {"model_endpoint": endpoint_name}
