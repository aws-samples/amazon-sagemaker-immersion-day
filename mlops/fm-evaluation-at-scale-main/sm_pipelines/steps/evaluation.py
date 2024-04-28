import sagemaker
import boto3
import json
from sagemaker.s3_utils import parse_s3_url
from sagemaker.serializers import JSONSerializer
import importlib
import markdown


def evaluation(
    data_s3_path,
    endpoint_name,
    data_config,
    model_config,
    algorithm_config,
    output_data_path,
):
    from fmeval.data_loaders.data_config import DataConfig
    from fmeval.model_runners.sm_jumpstart_model_runner import (
        JumpStartModelRunner,
    )
    from fmeval.reporting.eval_output_cells import EvalOutputCell
    from fmeval.constants import MIME_TYPE_JSONLINES

    # from amazon_fmeval.eval_algorithms.factual_knowledge import FactualKnowledge, FactualKnowledgeConfig

    s3 = boto3.client("s3")

    bucket, object_key = parse_s3_url(data_s3_path)
    s3.download_file(bucket, object_key, "dataset.jsonl")

    config = DataConfig(
        dataset_name=data_config["dataset_name"],
        dataset_uri="dataset.jsonl",
        dataset_mime_type=MIME_TYPE_JSONLINES,
        model_input_location=data_config["model_input_key"],
        target_output_location=data_config["target_output_key"],
    )

    evaluation_config = model_config["evaluation_config"]

    content_dict = {
        "inputs": evaluation_config["content_template"],
        "parameters": evaluation_config["inference_parameters"],
    }
    serializer = JSONSerializer()
    serialized_data = serializer.serialize(content_dict)

    content_template = serialized_data.replace('"PROMPT_PLACEHOLDER"', "$prompt")
    print(content_template)

    js_model_runner = JumpStartModelRunner(
        endpoint_name=endpoint_name,
        model_id=model_config["model_id"],
        model_version=model_config["model_version"],
        #output=evaluation_config["output"], #deprecated
        #content_template=content_template, #deprecated
        custom_attributes="accept_eula=true",
    )

    eval_output_all = []
    s3 = boto3.resource("s3")
    output_bucket, output_index = parse_s3_url(output_data_path)

    for algorithm in algorithm_config:
        # TODO: handle algorithm type
        algorithm_name = algorithm["algorithm"]
        module = importlib.import_module(algorithm["module"])
        algorithm_class = getattr(module, algorithm_name)
        algorithm_config_class = getattr(module, algorithm["config"])
        eval_algo = algorithm_class(
            algorithm_config_class(
                target_output_delimiter=algorithm["target_output_delimiter"]
            )
        )
        eval_output = eval_algo.evaluate(
            model=js_model_runner,
            dataset_config=config,
            prompt_template=evaluation_config["prompt_template"],
            save=True,
        )
        print(f"eval_output: {eval_output}")
        eval_output_all.append(eval_output)
        html = markdown.markdown(str(EvalOutputCell(eval_output[0])))
        file_index = (
            output_index
            + "/"
            + model_config["name"]
            + "_"
            + eval_algo.eval_name
            + ".html"
        )

        s3_object = s3.Object(bucket_name=output_bucket, key=file_index)

        s3_object.put(Body=html)

    eval_result = {"model_config": model_config, "eval_output": eval_output_all}
    print(f"eval_result: {eval_result}")

    return eval_result
