# ### 6. Deploy Step
# This step deploys the model for evaluation

from sagemaker.core.helper.session_helper import Session, get_execution_role, _wait_until, _deploy_done
from sagemaker.core.resources import Model, Endpoint, EndpointConfig
from sagemaker.core.shapes import ContainerDefinition, ProductionVariant
from sagemaker.core.common_utils import name_from_base
import boto3
import mlflow
import time
from sagemaker.mlops.workflow.function_step import step
from .pipeline_utils import PIPELINE_INSTANCE_TYPE


@step(
    name="ModelDeploy",
    instance_type=PIPELINE_INSTANCE_TYPE,
    display_name="Model Deploy",
    keep_alive_period_in_seconds=900
)
def deploy(
    tracking_server_arn: str,
    model_artifacts_s3_path: str,
    model_id: str,
    experiment_name: str,
    run_id: str,
):    

    mlflow.set_tracking_uri(tracking_server_arn)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_id=run_id):
        with mlflow.start_run(run_name="DeployStep", nested=True) as deploy_run:
            deployment_start_time = time.time()
            
            sagemaker_session = Session()
            role = get_execution_role(sagemaker_session, use_default=True)
            instance_count = 1
            instance_type = "ml.g5.2xlarge"
            health_check_timeout = 3600
            model_data_download_timeout = 3600

            model_config = {
                'HF_MODEL_ID': "/opt/ml/model",
                'OPTION_TRUST_REMOTE_CODE': 'true',
                'OPTION_ROLLING_BATCH': "vllm",
                'OPTION_DTYPE': 'bf16',
                'OPTION_QUANTIZE': 'fp8',
                'OPTION_TENSOR_PARALLEL_DEGREE': 'max',
                'OPTION_MAX_ROLLING_BATCH_SIZE': '32',
                'OPTION_MODEL_LOADING_TIMEOUT': '3600',
                'OPTION_MAX_MODEL_LEN': '4096'
            }
            
            endpoint_name = f"{model_id.split('/')[-1].replace('.', '-').replace('_','-')}-sft-djl"

            mlflow.log_params({
                "model_id": model_id,
                "instance_type": instance_type,
                "instance_count": instance_count,
                "endpoint_name": endpoint_name,
                "health_check_timeout": health_check_timeout,
                "model_data_download_timeout": model_data_download_timeout
            })
            mlflow.log_params({"model_config_" + k: v for k, v in model_config.items()})
            
            # Delete existing endpoint if it exists
            print(f"Checking for existing endpoint: {endpoint_name}")
            sm_client = boto3.client('sagemaker')
            try:
                sm_client.describe_endpoint(EndpointName=endpoint_name)
                print(f"Endpoint {endpoint_name} exists, deleting it before deployment")
                sm_client.delete_endpoint(EndpointName=endpoint_name)
                sm_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
                
                wait_seconds = 10
                total_wait_time = 0
                max_wait_time = 300
                endpoint_deleted = False
                while total_wait_time < max_wait_time and not endpoint_deleted:
                    try:
                        sm_client.describe_endpoint(EndpointName=endpoint_name)
                        time.sleep(wait_seconds)
                        total_wait_time += wait_seconds
                    except sm_client.exceptions.ClientError:
                        endpoint_deleted = True
            except sm_client.exceptions.ClientError:
                print(f"Endpoint {endpoint_name} does not exist, proceeding with deployment")
            
            region = sagemaker_session.boto_session.region_name
            inference_image_uri = f"763104351884.dkr.ecr.{region}.amazonaws.com/djl-inference:0.33.0-lmi15.0.0-cu128"
            mlflow.log_param("inference_image_uri", inference_image_uri)
            
            deploy_model_name = name_from_base("pipeline-model")
            core_model = Model.create(
                model_name=deploy_model_name,
                execution_role_arn=role,
                primary_container=ContainerDefinition(
                    image=inference_image_uri,
                    model_data_url=model_artifacts_s3_path,
                    environment=model_config,
                ),
            )

            EndpointConfig.create(
                endpoint_config_name=endpoint_name,
                production_variants=[
                    ProductionVariant(
                        variant_name="AllTraffic",
                        model_name=deploy_model_name,
                        initial_instance_count=instance_count,
                        instance_type=instance_type,
                        container_startup_health_check_timeout_in_seconds=health_check_timeout,
                        model_data_download_timeout_in_seconds=model_data_download_timeout,
                    )
                ],
            )

            Endpoint.create(
                endpoint_name=endpoint_name,
                endpoint_config_name=endpoint_name,
            )

            _wait_until(lambda: _deploy_done(sagemaker_session.sagemaker_client, endpoint_name), poll=30)
            
            deployment_time = time.time() - deployment_start_time
            mlflow.log_param("deployment_time_seconds", deployment_time)
            mlflow.log_param("deployment_success", 1)

            mlflow.set_tags({
                "endpoint_status": "deployed",
                "deployment_type": "sagemaker",
                "framework": "djl-lmi"
            })
    
    return endpoint_name
