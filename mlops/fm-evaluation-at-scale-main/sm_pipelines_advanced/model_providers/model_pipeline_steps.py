from abc import ABC, abstractmethod
import boto3
from datetime import datetime


class ModelPipelineSteps(ABC):
    def __init__(self, config):
        self.config = config

    @staticmethod
    @abstractmethod
    def deploy_step(model, *args):
        pass

    @staticmethod
    @abstractmethod
    def finetune_step(model, *args):
        pass

    @staticmethod
    @abstractmethod
    def deploy_finetuned_step(model, finetune_step_ret, *args):
        pass

    @staticmethod
    @abstractmethod
    def register(model, model_registry_config, eval_output):
        pass

    @staticmethod
    @abstractmethod
    def cleanup_step(model, *args):
        pass

    @staticmethod
    @abstractmethod
    def get_model_runner(model, content_template, *args):
        pass




