import sagemaker
from model_providers.model_pipeline_steps import ModelPipelineSteps
from sagemaker.jumpstart.model import JumpStartModel
from sagemaker.jumpstart.estimator import JumpStartEstimator
from sagemaker import ModelMetrics, MetricsSource
from sagemaker import get_execution_role
from sagemaker import image_uris, model_uris, Model, script_uris
from lib.utils import endpoint_exists, find_model_by_name, is_finetuning
import s3fs as s3fs
import boto3
import json


class ModelPipelineStepsJumpStart(ModelPipelineSteps):

    def __init__(self, config):
        super().__init__(config=config)

    @staticmethod
    def deploy_step(model, *args):
        model_id = model.config["model_id"]
        model_version = model.config["model_version"]
        endpoint_name = model.config["endpoint_name"]

        endpoint_exist = endpoint_exists(endpoint_name)

        if endpoint_exist:
            print("Endpoint already exists")
        else:
            my_model = JumpStartModel(model_id=model_id, model_version=model_version)
            predictor = my_model.deploy(initial_instance_count=model.config["deployment_config"]["num_instances"],
                                        instance_type=model.config["deployment_config"]["instance_type"],
                                        serializer=sagemaker.serializers.JSONSerializer(),
                                        deserializer=sagemaker.deserializers.JSONDeserializer(),
                                        endpoint_name=endpoint_name,
                                        accept_eula=True)

        return {"model_deployed": True}

    @staticmethod
    def finetune_step(model, *args):

        endpoint_name = model.config["endpoint_name"]
        endpoint_exist = endpoint_exists(endpoint_name)

        if endpoint_exist:
            print("Endpoint already exists")
            training_job_name = None
        else:
            model_id = model.config["model_id"]
            model_version = model.config["model_version"]

            model_fine_tuning_config = model.config["finetuning_config"]
            train_data_path = model_fine_tuning_config["train_data_path"]
            validation_data_path = model_fine_tuning_config["validation_data_path"]
            epoch = model_fine_tuning_config["parameters"]["epoch"]
            max_input_length = model_fine_tuning_config["parameters"]["max_input_length"]
            instance_count = model_fine_tuning_config["parameters"]["num_instances"]
            instance_type = model_fine_tuning_config["parameters"]["instance_type"]
            instruction_tuned = model_fine_tuning_config["parameters"]["instruction_tuned"]
            chat_dataset = model_fine_tuning_config["parameters"]["chat_dataset"]

            training_job_name = model_fine_tuning_config["training_job_name"]

            estimator = JumpStartEstimator(
                model_id=model_id,
                model_version=model_version,
                instance_count=instance_count,
                instance_type=instance_type,
                environment={"accept_eula": "true"},
                disable_output_compression=False)  # For Llama-2-70b, add instance_type = "ml.g5.48xlarge"

            # By default, instruction tuning is set to false. Thus, to use instruction tuning dataset you use
            estimator.set_hyperparameters(instruction_tuned=instruction_tuned,
                                          chat_dataset=chat_dataset,
                                          epoch=epoch,
                                          max_input_length=max_input_length)
            #estimator.fit({"training": train_data_path, "validation": validation_data_path})
            estimator.fit(inputs={"training": train_data_path}, job_name=training_job_name)

            training_job_name = estimator.latest_training_job.name

        return {"training_job_name": training_job_name}

    @staticmethod
    def register(model, model_registry_config, eval_output):
        sagemaker_session = sagemaker.Session()
        #  role = get_execution_role()
        sm_client = boto3.client("sagemaker")

        # Search model:
        #if selection_step_ret is None:
        #    best_model_name = model_list[0].config["name"]
        #    model = model_list[0]
        #else:
        #    best_model_name = selection_step_ret["best_model_name"]
        #    model = find_model_by_name(model_list, best_model_name)

        #eval_output = None
        #for results in args:
        #    if results["model_name"] == best_model_name:
        #        eval_output = results["evaluation_output"]
        #        break

        eval_result = eval_output[0][0]

        eval_report_s3_uri = model.config["output_data_path"] + "/evaluation-report/" + model.config["name"] + ".json"

        # Upload evaluation report to s3
        #eval_file_name = unique_name_from_base("evaluation")
        #eval_report_s3_uri = s3_path_join(
        #    "s3://",
        #    bucket,
        #    model_package_group_name,
        #    f"evaluation-report/{eval_file_name}.json",
        #)

        s3_fs = s3fs.S3FileSystem()
        eval_report_str = json.dumps(
            {
                "score": eval_result.dataset_scores[0].value,
                "algorithm": eval_result.dataset_scores[0].name,
            }
        )
        with s3_fs.open(eval_report_s3_uri, "wb") as file:
            file.write(eval_report_str.encode("utf-8"))

        # Create model_metrics as per evaluation report in s3
        model_metrics = ModelMetrics(
            model_statistics=MetricsSource(
                s3_uri=eval_report_s3_uri,
                content_type="application/json",
            )
        )

        model_package_group_name = model_registry_config["model_package_group_name"]
        model_package_group_description = model_registry_config["model_package_group_description"]

        try:
            sm_client.describe_model_package_group(
                ModelPackageGroupName=model_package_group_name
            )
        except:
            model_package_group_input_dict = {
                "ModelPackageGroupName": model_package_group_name,
                "ModelPackageGroupDescription": model_package_group_description,
            }
            create_model_package_group_response = sm_client.create_model_package_group(
                **model_package_group_input_dict
            )
            print(
                "ModelPackageGroup Arn : {}".format(
                    create_model_package_group_response["ModelPackageGroupArn"]
                )
            )

        model_id = model.config["model_id"]
        model_version = model.config["model_version"]

        if is_finetuning(model):
            training_job_name = model.config["finetuning_config"]["training_job_name"]
            model_id = model.config["model_id"]

            estimator = JumpStartEstimator.attach(training_job_name, model_id=model_id)
            model_package = estimator.register(
                model_package_group_name=model_package_group_name,
                content_types=["application/json"],
                response_types=["application/json"],
                customer_metadata_properties={
                    "score": str(eval_result.dataset_scores[0].value),
                    "algorithm": eval_result.dataset_scores[0].name,
                },
                model_metrics=model_metrics,
            )
        else:
            js_model = JumpStartModel(model_id=model_id, model_version=model_version)

            model_package = js_model.register(
                model_package_group_name=model_package_group_name,
                image_uri=js_model.image_uri,
                content_types=["application/json"],
                response_types=["application/json"],
                customer_metadata_properties={
                    "score": str(eval_result.dataset_scores[0].value),
                    "algorithm": eval_result.dataset_scores[0].name,
                },
                skip_model_validation="All",
                model_metrics=model_metrics,
            )

        model_package_arn = model_package.model_package_arn

        return {"model_package_arn", model_package_arn}

    @staticmethod
    def deploy_finetuned_step(model, finetune_step_ret, *args):
        model_id = model.config["model_id"]
        endpoint_name = model.config["endpoint_name"]
        training_job_name = finetune_step_ret["training_job_name"]

        endpoint_exist = endpoint_exists(endpoint_name)

        if endpoint_exist:
            print("Endpoint already exists")
        else:
            estimator = JumpStartEstimator.attach(training_job_name, model_id=model_id)
            estimator.logs()
            predictor = estimator.deploy(initial_instance_count=model.config["deployment_config"]["num_instances"],
                                         instance_type=model.config["deployment_config"]["instance_type"],
                                         serializer=sagemaker.serializers.JSONSerializer(),
                                         deserializer=sagemaker.deserializers.JSONDeserializer(),
                                         endpoint_name=endpoint_name)

        return {"model_deployed": True}

    @staticmethod
    def cleanup_step(model, *args):
        client = boto3.client('sagemaker')
        client.delete_endpoint(EndpointName=model.config["endpoint_name"])
        client.delete_endpoint_config(EndpointConfigName=model.config["endpoint_name"])
        return {"cleanup_done": True}

    @staticmethod
    def get_model_runner(model, content_template):

        from fmeval.model_runners.sm_jumpstart_model_runner import JumpStartModelRunner

        endpoint_name = model.config["endpoint_name"]

        js_model_runner = JumpStartModelRunner(
            endpoint_name=endpoint_name,
            model_id=model.config["model_id"],
            model_version=model.config["model_version"],
            output=model.config["evaluation_config"]["output"],
            content_template=content_template,
            custom_attributes="accept_eula=true"
        )

        return js_model_runner

