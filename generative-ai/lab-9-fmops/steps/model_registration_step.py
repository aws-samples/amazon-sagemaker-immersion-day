# ### 7. Model Registration Step
# This step registers the fine-tuned model in MLflow model registry and SageMaker AI model registry

from sagemaker.mlops.workflow.function_step import step
from .pipeline_utils import PIPELINE_INSTANCE_TYPE


@step(
    name="ModelRegistration",
    instance_type=PIPELINE_INSTANCE_TYPE,
    display_name="Model Registration", 
    keep_alive_period_in_seconds=900
)
def register_model(
    tracking_server_arn: str,
    experiment_name: str,
    run_id: str,
    model_artifacts_s3_path: str,
    model_id: str,
    model_name: str,
    endpoint_name: str,
    evaluation_score: float,
    pipeline_name: str,
    model_description: str
):
    import json
    import mlflow
    import boto3
    import os
    import tempfile
    import time
    from datetime import datetime
    
    print(f"Registering model: {model_name}")
    
    # Set up MLflow tracking
    mlflow.set_tracking_uri(tracking_server_arn)
    mlflow.set_experiment(experiment_name)
    
    # Connect to MLflow with the specific run
    with mlflow.start_run(run_id=run_id):
        # Create model metadata
        tags = {
            "model_id": model_id,
            "base_model": model_id.split('/')[-1],
            "task": "medical_qa",
            "framework": "pytorch",
            "endpoint_name": endpoint_name,
            "model_artifacts_s3_path": model_artifacts_s3_path,
            "deployment_timestamp": datetime.now().isoformat(),
            "description": model_description,
            "registered_by": pipeline_name
        }
            
        # Log model info as parameters
        mlflow.log_param("registered_model_name", model_name)
        mlflow.log_param("model_artifacts_path", model_artifacts_s3_path)
        mlflow.log_param("evaluation_score", evaluation_score)
        mlflow.log_param("endpoint_name", endpoint_name)
        mlflow.log_param("registration_timestamp", datetime.now().isoformat())
        
        # Log endpoint information as an artifact
        model_info = {
            "model_name": model_name,
            "model_id": model_id,
            "endpoint_name": endpoint_name,
            "model_artifacts_s3_path": model_artifacts_s3_path,
            "evaluation_score": float(evaluation_score),
            "registration_timestamp": datetime.now().isoformat()
        }
        
        with open("/tmp/model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
        mlflow.log_artifact("/tmp/model_info.json")
        
        # Create model card
        model_card = f"""
        # {model_name}
        
        ## Model Information
        - **Base Model**: {model_id}
        - **Task**: Medical Question Answering
        - **Evaluation Score**: {evaluation_score:.4f}
        - **Endpoint**: {endpoint_name}
        
        ## Description
        {model_description}
        
        ## Registration Details
        - Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        - Pipeline: {pipeline_name}
        """
        
        with open("/tmp/model_card.md", "w") as f:
            f.write(model_card)
        mlflow.log_artifact("/tmp/model_card.md")
        
        # PART 1: REGISTER WITH MLFLOW MODEL REGISTRY
        mlflow_version = None
        try:
            client = mlflow.tracking.MlflowClient()
            
            # Check if model exists and create if it doesn't
            try:
                client.get_registered_model(model_name)
                print(f"Model {model_name} already exists in MLflow registry")
            except mlflow.exceptions.MlflowException:
                client.create_registered_model(
                    name=model_name,
                    description=f"Fine-tuned medical LLM based on {model_id}"
                )
                print(f"Created new registered model: {model_name}")
            
            # Create empty model directory with artifacts
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Create a minimal model file to log
                os.makedirs(os.path.join(tmp_dir, "model"), exist_ok=True)
                
                # Copy model info and card to directory
                with open(os.path.join(tmp_dir, "model", "model_info.json"), "w") as f:
                    json.dump(model_info, f, indent=2)
                    
                with open(os.path.join(tmp_dir, "model", "model_card.md"), "w") as f:
                    f.write(model_card)
                
                # Create a model reference file pointing to the S3 artifacts
                model_ref = {
                    "artifact_path": model_artifacts_s3_path,
                    "flavors": {
                        "pytorch": {
                            "model_data": model_artifacts_s3_path,
                            "pytorch_version": "2.0+"
                        }
                    },
                    "run_id": run_id,
                    "model_class": "LLM",
                    "model_format": "PyTorch"
                }
                
                with open(os.path.join(tmp_dir, "model", "MLmodel"), "w") as f:
                    json.dump(model_ref, f, indent=2)
                
                # Log artifacts directory as model
                mlflow.log_artifacts(tmp_dir, artifact_path="")
            
            # Now register the model - try both methods
            try:
                # Method 1: Use direct registration with source as run URI
                model_uri = f"runs:/{run_id}/model"
                model_details = mlflow.register_model(
                    model_uri=model_uri,
                    name=model_name,
                    tags=tags
                )
                mlflow_version = model_details.version
                
            except Exception as e1:
                print(f"Method 1 registration failed: {str(e1)}")
                
                try:
                    # Method 2: Create version with client API
                    model_version = client.create_model_version(
                        name=model_name,
                        source=f"runs:/{run_id}/model",  # Use run URI instead of direct S3
                        run_id=run_id,
                        description=f"Fine-tuned LLM deployed at endpoint: {endpoint_name}"
                    )
                    mlflow_version = model_version.version
                    
                    # Wait for model registration to complete
                    for _ in range(10):  # Try for up to ~50 seconds
                        version_details = client.get_model_version(model_name, model_version.version)
                        if version_details.status == "READY":
                            break
                        time.sleep(5)
                    
                    # Add tags to the registered model version
                    for key, value in tags.items():
                        client.set_model_version_tag(model_name, model_version.version, key, value)
                except Exception as e2:
                    print(f"Method 2 registration failed: {str(e2)}")
                    mlflow_version = "unknown"
            
            if mlflow_version and mlflow_version != "unknown":
                # Transition model to Production/Staging based on evaluation score
                if evaluation_score >= 0.3:  # Example threshold
                    client.transition_model_version_stage(
                        name=model_name,
                        version=mlflow_version,
                        stage="Production",
                        archive_existing_versions=True
                    )
                    print(f"Model {model_name} version {mlflow_version} promoted to Production")
                else:
                    client.transition_model_version_stage(
                        name=model_name,
                        version=mlflow_version,
                        stage="Staging",
                        archive_existing_versions=False
                    )
                    print(f"Model {model_name} version {mlflow_version} added to Staging due to lower evaluation score")
            
            print(f"Successfully registered model in MLflow: {model_name}, version: {mlflow_version}")
            
        except Exception as e:
            print(f"Error registering model in MLflow: {str(e)}")
            mlflow_version = "unknown"
            
        # PART 2: REGISTER WITH SAGEMAKER MODEL REGISTRY
        sm_model_version = "unknown"
        try:
            sm_client = boto3.client('sagemaker')
            
            # Create a normalized name for SageMaker resources
            sm_model_name = model_name.replace(".", "-").replace("_", "-")
            
            # Create or update model package group
            try:
                sm_client.describe_model_package_group(ModelPackageGroupName=sm_model_name)
                print(f"SageMaker model package group {sm_model_name} already exists")
            except sm_client.exceptions.ClientError:
                sm_client.create_model_package_group(
                    ModelPackageGroupName=sm_model_name,
                    ModelPackageGroupDescription=f"Fine-tuned LLM model: {model_name}"
                )
                print(f"Created SageMaker model package group: {sm_model_name}")
            
            # Create a model package and register it
            try:
                # Create model package
                response = sm_client.create_model_package(
                    ModelPackageGroupName=sm_model_name,
                    ModelPackageDescription=model_description,
                    SourceAlgorithmSpecification={
                        'SourceAlgorithms': [
                            {
                                'AlgorithmName': 'pytorch-llm',
                                'ModelDataUrl': model_artifacts_s3_path
                            }
                        ]
                    },
                    ValidationSpecification={
                        'ValidationRole': 'dummy-role-dummy-role',  # Required but not used
                        'ValidationProfiles': [
                            {
                                'ProfileName': 'ValidationProfile1',
                                'TransformJobDefinition': {
                                    'TransformInput': {
                                        'DataSource': {
                                            'S3DataSource': {
                                                'S3DataType': 'S3Prefix',
                                                'S3Uri': 's3://dummy-bucket/dummy-prefix'  # Required but not used
                                            }
                                        }
                                    },
                                    'TransformOutput': {
                                        'S3OutputPath': 's3://dummy-bucket/dummy-output'  # Required but not used
                                    },
                                    'TransformResources': {
                                        'InstanceType': 'ml.m5.large',  # Required but not used
                                        'InstanceCount': 1
                                    }
                                }
                            }
                        ]
                    },
                    ModelApprovalStatus='Approved',
                    MetadataProperties={
                        'GeneratedBy': pipeline_name,
                        'Repository': model_id,
                    },
                    CustomerMetadataProperties={
                        'EvaluationScore': str(evaluation_score)
                    },
                    ModelMetrics={
                        'ModelQuality': {
                            'Statistics': {
                                'ContentType': 'application/json',
                                'S3Uri': f"s3://{model_artifacts_s3_path.split('/', 3)[2]}/{run_id}/artifacts/model_info.json"
                            }
                        }
                    }
                )
                
                sm_model_version = response['ModelPackageArn'].split('/')[-1]
                print(f"Created SageMaker model package: {sm_model_version}")
                
            except Exception as e_package:
                print(f"Error creating model package: {str(e_package)}")
            
            # Log SageMaker details
            mlflow.log_param("sagemaker_model_group", sm_model_name)
            mlflow.log_param("sagemaker_model_version", sm_model_version)
            
            print(f"Successfully integrated with SageMaker model registry")
            
        except Exception as e:
            print(f"Warning: Error in SageMaker model registry integration: {str(e)}")
            
    return model_name, str(mlflow_version)