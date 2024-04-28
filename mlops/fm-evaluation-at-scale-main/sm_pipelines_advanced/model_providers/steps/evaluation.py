import boto3
import markdown
from sagemaker.s3_utils import parse_s3_url
from sagemaker.serializers import JSONSerializer
import importlib

from fmeval.data_loaders.data_config import DataConfig
from fmeval.reporting.eval_output_cells import EvalOutputCell
from fmeval.constants import MIME_TYPE_JSONLINES


def evaluation(model, data_config, algorithm_config, preprocess_step_ret, deploy_step_ret):

    s3 = boto3.client("s3")

    data_s3_path = preprocess_step_ret["output_s3_path"]

    bucket, object_key = parse_s3_url(data_s3_path)
    s3.download_file(bucket, object_key, "dataset.jsonl")

    config = DataConfig(
        dataset_name="dataset",
        dataset_uri="dataset.jsonl",
        dataset_mime_type=MIME_TYPE_JSONLINES,
        model_input_location=data_config["model_input_key"],
        target_output_location=data_config["target_output_key"],
    )

    evaluation_config = model.config["evaluation_config"]

    content_dict = {
        "inputs": evaluation_config["content_template"],
        "parameters": evaluation_config["inference_parameters"]
    }
    serializer = JSONSerializer()
    serialized_data = serializer.serialize(content_dict)

    content_template = serialized_data.replace('"PROMPT_PLACEHOLDER"', '$prompt')
    print(content_template)

    model_runner = model.get_model_runner(model, content_template)

    eval_output_all = []
    s3 = boto3.resource("s3")
    output_bucket, output_index = parse_s3_url(model.config["output_data_path"])

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
            model=model_runner,
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
                + model.config["name"]
                + "_"
                + eval_algo.eval_name
                + ".html"
        )

        s3_object = s3.Object(bucket_name=output_bucket, key=file_index)

        s3_object.put(Body=html)

    return {"evaluation_output": eval_output_all, "model_name": model.config["name"]}

