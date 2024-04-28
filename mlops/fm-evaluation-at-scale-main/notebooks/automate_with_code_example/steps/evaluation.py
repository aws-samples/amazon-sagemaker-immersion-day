import boto3
import markdown
from sagemaker.s3_utils import parse_s3_url
from fmeval.data_loaders.data_config import DataConfig
from fmeval.reporting.eval_output_cells import EvalOutputCell
from fmeval.constants import MIME_TYPE_JSONLINES
from fmeval.model_runners.sm_jumpstart_model_runner import JumpStartModelRunner
from fmeval.eval_algorithms.factual_knowledge import FactualKnowledge, FactualKnowledgeConfig


def evaluation(model, preprocess_step_ret, deploy_step_ret):
    s3 = boto3.client("s3")

    model_id = model["model_id"]
    model_version = model["model_version"]
    model_name = model["model_name"]
    
    # FMEval library needs three components:
    # - The evaluation dataset
    # - A model runner
    # - An algorithm to use
    # We will configure each of this components in the following lines:

    # Get the dataset
    data_s3_path = preprocess_step_ret["evaluation_data_location"]
    bucket, object_key = parse_s3_url(data_s3_path)
    print(bucket)
    print(object_key)
    s3.download_file(bucket, object_key, "dataset.jsonl")

    # Configure FMEval for reading the dataset
    config = DataConfig(
        dataset_name="dataset",
        dataset_uri="dataset.jsonl",
        dataset_mime_type=MIME_TYPE_JSONLINES,
        model_input_location="model_input",
        target_output_location="target_output",
    )

    # Create a JumpStartModelRunner that will be used by FMEval library to perform the call to the model 
    # and the evaluation of each entry of the evaluation dataset
    endpoint_name = deploy_step_ret["model_endpoint"]
    js_model_runner = JumpStartModelRunner(
        endpoint_name=endpoint_name,
        model_id=model_id,
        model_version=model_version,
        custom_attributes="accept_eula=true"
    )

    # Configure and launch FactualKnowledge evaluation algorithm
    eval_output_all = []
    eval_algo = FactualKnowledge(FactualKnowledgeConfig("<OR>"))
    eval_output = eval_algo.evaluate(
        model=js_model_runner,
        dataset_config=config,
        prompt_template="$feature",
        save=True,
    )
    eval_output_all.append(eval_output)

    # Save results to S3
    s3 = boto3.resource("s3")
    output_bucket, output_index = parse_s3_url(preprocess_step_ret["output_data_path"])
    
    html = markdown.markdown(str(EvalOutputCell(eval_output[0])))
    file_index = (
            output_index
            + "/"
            + model_name
            + "_"
            + eval_algo.eval_name
            + ".html"
    )

    s3_object = s3.Object(bucket_name=output_bucket, key=file_index)
    s3_object.put(Body=html)

    return {"evaluation_output": eval_output_all, "model_name": model_name}
