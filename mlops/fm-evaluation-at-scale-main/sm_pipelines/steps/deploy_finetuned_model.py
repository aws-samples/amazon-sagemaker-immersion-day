import sagemaker
from sagemaker.jumpstart.estimator import JumpStartEstimator
from lib.utils import endpoint_exists

def deploy_finetuned_model(training_job_name, model_id, endpoint_name, instance_type="ml.g5.2xlarge", initial_instance_count=1):

    endpoint_exist = endpoint_exists(endpoint_name)

    if endpoint_exist:
        print("Endpoint already exists")
    else:
        estimator = JumpStartEstimator.attach(training_job_name, model_id=model_id)
        estimator.logs()
        predictor = estimator.deploy(initial_instance_count=initial_instance_count,
                                     instance_type=instance_type,
                                     serializer=sagemaker.serializers.JSONSerializer(),
                                     deserializer=sagemaker.deserializers.JSONDeserializer(),
                                     endpoint_name=endpoint_name,
                                     accept_eula = True)

    return endpoint_name