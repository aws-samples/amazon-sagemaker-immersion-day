import boto3
from sagemaker.s3_utils import parse_s3_url

def preprocess(data_s3_path, output_s3_folder):
    s3 = boto3.client("s3")
    
    bucket, object_key = parse_s3_url(data_s3_path)
    s3.download_file(bucket, object_key, "dataset.jsonl")

    # Some preprocessing
    output_s3_path = output_s3_folder + "/processed-dataset.jsonl"
    s3.upload_file("dataset.jsonl", *parse_s3_url(output_s3_path))

    return output_s3_path