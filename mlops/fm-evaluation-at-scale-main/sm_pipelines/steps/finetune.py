from sagemaker.jumpstart.estimator import JumpStartEstimator
from lib.utils import endpoint_exists

def finetune(model_id, endpoint_name, train_data_location, validation_data_location, instance_count=1, instance_type="ml.m5.xlarge", instruction_tuned="False", chat_dataset="False", epoch="1", max_input_length="100"):

    endpoint_exist = endpoint_exists(endpoint_name)
    estimator = None

    if endpoint_exist:
        print("Endpoint already exists")
    else:
        estimator = JumpStartEstimator(
            model_id=model_id,
            instance_count=instance_count,
            instance_type=instance_type,
            environment={"accept_eula": "true"},
            disable_output_compression=False)  # For Llama-2-70b, add instance_type = "ml.g5.48xlarge"

        # By default, instruction tuning is set to false. Thus, to use instruction tuning dataset you use
        estimator.set_hyperparameters(instruction_tuned=instruction_tuned,
                                      chat_dataset=chat_dataset,
                                      epoch=epoch,
                                      max_input_length=max_input_length,
                                      )

        #estimator.fit({"training": train_data_location, "validation": validation_data_location} )
        estimator.fit({"training": train_data_location})

    if estimator is None:
        return endpoint_name
    else:
        return estimator.latest_training_job.name