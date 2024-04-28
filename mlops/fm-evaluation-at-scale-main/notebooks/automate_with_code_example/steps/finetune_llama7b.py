from steps.utils import endpoint_exists
from sagemaker.jumpstart.estimator import JumpStartEstimator


def finetune_llama7b(model, preprocess_step_ret):

    model_id = model["model_id"]
    model_version = model["model_version"]
    endpoint_name = model["endpoint_name"]
    instance_type = model["finetune_instance_type"]
    num_instances = model["finetune_num_instances"]
    epoch = model["epoch"]
    max_input_length = model["max_input_length"]
    per_device_train_batch_size = model["per_device_train_batch_size"]
    instruction_tuned = model["instruction_tuned"]
    chat_dataset = model["chat_dataset"]
    training_job_name = model["training_job_name"]

    if instruction_tuned == "True":
        train_data_path = preprocess_step_ret["fine_tune_data_ist_location"]
    else:
        train_data_path = preprocess_step_ret["fine_tune_data_daft_location"]

    if endpoint_exists(endpoint_name):
        print("Endpoint already exists")
        training_job_name = None
    else:
        estimator = JumpStartEstimator(
            model_id=model_id,
            model_version=model_version,
            instance_count=num_instances,
            instance_type=instance_type,
            environment={"accept_eula": "true"},
            disable_output_compression=False)

        estimator.set_hyperparameters(instruction_tuned=instruction_tuned,
                                      chat_dataset=chat_dataset,
                                      epoch=epoch,
                                      per_device_train_batch_size=per_device_train_batch_size,
                                      max_input_length=max_input_length)
        
        # estimator.fit({"training": train_data_path, "validation": validation_data_path}) # if there is a validation dataset
        estimator.fit(inputs={"training": train_data_path}, job_name=training_job_name)

        training_job_name = estimator.latest_training_job.name

    return {"training_job_name": training_job_name}
