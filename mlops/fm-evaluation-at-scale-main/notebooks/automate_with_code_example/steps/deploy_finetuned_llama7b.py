import sagemaker
from sagemaker.jumpstart.estimator import JumpStartEstimator
from steps.utils import endpoint_exists


def deploy_finetuned_llama7b(model, finetune_step_ret):

    training_job_name = finetune_step_ret["training_job_name"]

    model_id = model["model_id"]
    endpoint_name = model["endpoint_name"]
    instance_type = model["instance_type"]
    num_instances = model["num_instances"]

    if not endpoint_exists(endpoint_name):
        estimator = JumpStartEstimator.attach(training_job_name, model_id=model_id)
        estimator.logs()
        predictor = estimator.deploy(initial_instance_count=num_instances,
                                     instance_type=instance_type,
                                     serializer=sagemaker.serializers.JSONSerializer(),
                                     deserializer=sagemaker.deserializers.JSONDeserializer(),
                                     endpoint_name=endpoint_name)

    return {"model_endpoint": endpoint_name, "model_deployed": True}
