import boto3
import yaml
from datetime import datetime


def is_finetuning(model):
    if "finetuning_config" in model.config.keys():
        return True
    else:
        return False

def find_model_by_name(model_list, model_name):
    for model in model_list:
        if model.config["name"] == model_name:
            return model

    return None

def create_training_job_name(model):
    model_id = model.config["model_id"]
    return f"{model_id}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')[:-3]}"


def get_step_name(name, model):
    return name+"_"+model.config["name"]

def get_deploy_step_name(model):
    return "deploy_"+model.config["name"]


def get_finetune_step_name(model):
    return "finetune_"+model.config["name"]


def get_deploy_finetuned_step_name(model):
    return "deploy_finetuned_"+model.config["name"]


def get_register_step_name(model):
    return "register_"+model.config["name"]


def get_cleanup_step_name(model):
    return "cleanup_"+model.config["name"]


def endpoint_exists(endpoint_name):
    endpoint_exist = False

    client = boto3.client('sagemaker')
    response = client.list_endpoints()
    endpoints = response["Endpoints"]

    for endpoint in endpoints:
        if endpoint_name == endpoint["EndpointName"]:
            endpoint_exist = True
            break

    return endpoint_exist


class ConfigParser:
    """
    Provides a simple interface for reading and validating a evaluation configuration file
    @param config_file: The path to the evaluation configuration file
    """
    REQUIRED_KEYS = {
        "models": ["model_id", "model_version"],
        "datasets": [],
        "evaluation": []
    }

    def __init__(self, config_file):
        self.config_file = config_file
        self.config = {}
        self.load_config()
        self.validate_config()

    def load_config(self):
        with open(self.config_file, "r") as f:
            self.config = yaml.safe_load(f)
            return self.config

    # TODO: Implement validation function using yaml schema definition
    def validate_config(self):
        # pass
        # check if the config file has all the required keys
        # iterate over all keys and subkeys in required_configs and validate that they exist in self.config
        # for key, subkeys in self.REQUIRED_KEYS.items():
        #     if key not in self.config:
        #         raise ValueError(f"Config file is missing required key: {key}")
        #     for subkey in subkeys:
        #         if subkey not in self.config[key] and subkey:
        #             raise ValueError(f"Config file is missing required key: {subkey}")
        pass

    def __get__(self, key):
        return self.config[key]

    # return the config file as a dictionary
    def get_config(self):
        return self.config

    # return a specific key from the config file
    def get_config_key(self, key):
        return self.config[key]