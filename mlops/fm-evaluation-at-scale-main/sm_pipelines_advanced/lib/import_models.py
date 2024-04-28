from model_providers.model_pipeline_steps_jumpstart import ModelPipelineStepsJumpStart


def import_models(config):
    models = []

    # Import models
    for model_config in config["models"]:
        model_provider = model_config["model_provider"]

        if model_provider == "jumpstart":
            model = ModelPipelineStepsJumpStart(model_config)
            models.append(model)
        # You can implement other model providers by extending the class ModelPipelineSteps.py and add them here

    return models


