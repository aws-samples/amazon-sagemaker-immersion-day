# ### 7. Quantitative Evaluation Step

# After fine-tuning, this step assesses the model's quantitative performance.

from sagemaker.mlops.workflow.function_step import step
from .pipeline_utils import PIPELINE_INSTANCE_TYPE
from .pipeline_utils import SYSTEM_PROMPT
from .pipeline_utils import convert_to_messages


@step(
    name="QuantitativeModelEvaluation",
    instance_type=PIPELINE_INSTANCE_TYPE,
    display_name="Quantitative Model Evaluation",
    keep_alive_period_in_seconds=900,
    dependencies="./eval/requirements.txt"
)
def quantitative_evaluate(
    tracking_server_arn: str,
    experiment_name: str,
    run_id: str,
    endpoint_name: str,
    mlflow_trace_attributes: dict
)-> dict:
    import os
    import json
    import time
    import boto3
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm.notebook import tqdm
    from datasets import load_dataset
    import mlflow
    import uuid
    import traceback
    from datetime import datetime
    from rouge_score import rouge_scorer
    from mlflow.entities import SpanType
        
    @mlflow.trace(
        name="call-local-llm", span_type=SpanType.LLM, attributes={
            "model": mlflow_trace_attributes["model_id"],
            "guardrail_id": mlflow_trace_attributes["guardrail_id"],
            "guardrail_version": mlflow_trace_attributes["guardrail_version"]
        }
    )
    def invoke_sagemaker_endpoint(payload, endpoint_name):
        """
        Invoke a SageMaker endpoint with the given payload.
    
        Args:
            payload (dict): The input data to send to the endpoint
            endpoint_name (str): The name of the SageMaker endpoint
    
        Returns:
            dict: The response from the endpoint
        """
        bedrock_runtime = boto3.client('bedrock-runtime')
        guardrail_id = mlflow_trace_attributes["guardrail_id"]
        guardrail_version = mlflow_trace_attributes["guardrail_version"]
        guardrail_response_input = bedrock_runtime.apply_guardrail(
            guardrailIdentifier=guardrail_id,
            guardrailVersion=guardrail_version,
            source='INPUT',
            content=[{'text': {'text': payload["messages"][0]["content"]}}]
        )
        guardrailResult = guardrail_response_input["action"]
    
        if guardrailResult == "GUARDRAIL_INTERVENED":
            reason = guardrail_response_input["assessments"]
            return guardrail_response_input["outputs"][0]["text"], -1
        
        try:
            start_time = time.time()
            # response = sm_client.invoke_endpoint(
            #     EndpointName=endpoint_name,
            #     ContentType='application/json',
            #     Body=json.dumps(payload)
            # )
            # inference_time = time.time() - start_time
            
            # response_body = response['Body'].read().decode('utf-8')
            # return json.loads(response_body), inference_time

            from sagemaker.core.resources import Endpoint
            
            ep = Endpoint.get(endpoint_name=endpoint_name)
            resp = ep.invoke(
                body=json.dumps(payload),
                content_type="application/json",
                accept="application/json",
            )
            result = json.loads(resp.body.read().decode("utf-8"))
            response = result['choices'][0]['message']['content']
            inference_time = time.time() - start_time
            return response, inference_time
        except Exception as e:
            print(f"Error invoking endpoint {endpoint_name}: {str(e)}")
            return None, -1
    
    def calculate_metrics(predictions, references):
        """
        Calculate all evaluation metrics for summarization using LightEval.
    
        Args:
            predictions (list): List of generated summaries
            references (list): List of reference summaries
    
        Returns:
            dict: Dictionary containing all metric scores
        """
        metrics = {}
        
        # Initialize the Rouge scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
        # Calculate ROUGE scores for each prediction-reference pair
        rouge_scores = {
            'rouge1_f': [],
            'rouge2_f': [],
            'rougeL_f': [],
            'rouge1_precision': [],
            'rouge1_recall': [],
            'rouge2_precision': [],
            'rouge2_recall': [],
            'rougeL_precision': [],
            'rougeL_recall': []
        }
    
        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)

            # Extract all metrics
            rouge_scores['rouge1_f'].append(scores['rouge1'].fmeasure)
            rouge_scores['rouge2_f'].append(scores['rouge2'].fmeasure)
            rouge_scores['rougeL_f'].append(scores['rougeL'].fmeasure)
            
            rouge_scores['rouge1_precision'].append(scores['rouge1'].precision)
            rouge_scores['rouge1_recall'].append(scores['rouge1'].recall)
            rouge_scores['rouge2_precision'].append(scores['rouge2'].precision)
            rouge_scores['rouge2_recall'].append(scores['rouge2'].recall)
            rouge_scores['rougeL_precision'].append(scores['rougeL'].precision)
            rouge_scores['rougeL_recall'].append(scores['rougeL'].recall)
            
        # Average ROUGE scores
        for key in rouge_scores:
            metrics[key] = sum(rouge_scores[key]) / len(rouge_scores[key])
        
        # Calculate prediction statistics
        metrics['avg_prediction_length'] = np.mean([len(pred.split()) for pred in predictions])
        metrics['min_prediction_length'] = min([len(pred.split()) for pred in predictions])
        metrics['max_prediction_length'] = max([len(pred.split()) for pred in predictions])
        
        # Calculate reference statistics
        metrics['avg_reference_length'] = np.mean([len(ref.split()) for ref in references])
        metrics['min_reference_length'] = min([len(ref.split()) for ref in references])
        metrics['max_reference_length'] = max([len(ref.split()) for ref in references])
        
        # Calculate length ratio
        metrics['avg_length_ratio'] = np.mean([len(pred.split()) / len(ref.split()) if len(ref.split()) > 0 else 0 
                                              for pred, ref in zip(predictions, references)])
    
        print(f"Metrics: {metrics}")
    
        return metrics
    
    def generate_summaries_with_model(endpoint_name, dataset):
        """
        Generate summaries using a model deployed on SageMaker.
    
        Args:
            endpoint_name (str): SageMaker endpoint name
            dataset: Dataset containing dialogues
    
        Returns:
            list: Generated summaries
            list: Inference times for each summary
        """
        predictions = []
        inference_times = []
        failed_generations = 0
    
        for example in tqdm(dataset, desc="Generating Responses"):
            payload = {}
            messages_prompt = convert_to_messages(example, SYSTEM_PROMPT)
            payload["messages"] = messages_prompt["messages"]
            payload["parameters"] = {
                "max_new_tokens": 512,
                "top_p": 0.9,
                "temperature": 0.6,
                "return_full_text": False
            }
    
            # Call the model endpoint
            try:
                response, inference_time = invoke_sagemaker_endpoint(payload, endpoint_name)
                
                # Extract the generated text
                if response is None:
                    prediction = "Error generating response."
                    failed_generations += 1
                elif isinstance(response, list):
                    prediction = response[0].get('generated_text', '').strip()
                elif isinstance(response, dict):
                    prediction = response.get('generated_text', '').strip()
                else:
                    prediction = str(response).strip()
    
                prediction = prediction.split("<|eot_id|>")[0] if "<|eot_id|>" in prediction else prediction
                
                # Log individual inference metrics
                mlflow.log_metric(f"inference_time_sample_{len(predictions)}", inference_time)
                
                inference_times.append(inference_time)
                
            except Exception as e:
                print(f"Error invoking SageMaker endpoint {endpoint_name}: {e}")
                prediction = "Error generating response."
                failed_generations += 1
                inference_times.append(-1)
    
            predictions.append(prediction)
    
        # Log failure rate
        mlflow.log_metric("failed_generations", failed_generations)
        mlflow.log_metric("failure_rate", failed_generations / len(dataset) if len(dataset) > 0 else 0)
    
        return predictions, inference_times
    
    def evaluate_model_on_dataset(model_config, dataset):
        """
        Evaluate a fine-tuned model on a dataset using both automated and human metrics.
    
        Args:
            model_config (dict): Model configuration with name and endpoint
            dataset: dataset for evaluation
    
        Returns:
            dict: Evaluation results
        """
        model_name = model_config["name"]
        endpoint_name = model_config["endpoint"]
    
        print(f"\nEvaluating model: {model_name} on endpoint: {endpoint_name}")
    
        # Get references
        references = ["\n".join([example["Complex_CoT"], example["Response"]]) for example in dataset]
    
        # Generate summaries
        print("\nGenerating Responses...")
        predictions, inference_times = generate_summaries_with_model(endpoint_name, dataset)
        
        # Log inference time metrics
        valid_times = [t for t in inference_times if t > 0]
        if valid_times:
            mlflow.log_metric("avg_inference_time", np.mean(valid_times))
            mlflow.log_metric("min_inference_time", min(valid_times))
            mlflow.log_metric("max_inference_time", max(valid_times))
            mlflow.log_metric("p95_inference_time", np.percentile(valid_times, 95))
    
        # Calculate automated metrics using LightEval
        print("\nCalculating evaluation metrics with LightEval...")
        metrics = calculate_metrics(predictions, references)
        
        # Log all calculated metrics to MLflow
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Create a comparison table of predictions vs references
        comparison_data = []
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

        for i, (pred, ref) in enumerate(zip(predictions[:5], references[:5])):
            # Calculate Rouge-1 score for this example
            rouge1_score = scorer.score(ref, pred)['rouge1'].fmeasure
            
            comparison_data.append({
                "example_id": i,
                "prediction": pred[:500] + ("..." if len(pred) > 500 else ""),  # Truncate for readability
                "reference": ref[:500] + ("..." if len(ref) > 500 else ""),     # Truncate for readability
                "rouge1_f": rouge1_score
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        # Save comparison to a temporary CSV and log it as an artifact
        temp_csv = f"/tmp/predictions_comparison_{uuid.uuid4().hex[:8]}.csv"
        comparison_df.to_csv(temp_csv, index=False)
        mlflow.log_artifact(temp_csv, "model_predictions")
            
        # Format results
        results = {
            "model_name": model_name,
            "endpoint_name": endpoint_name,
            "num_samples": len(dataset),
            "metrics": metrics,
            "predictions": predictions[:5],  # First 5 predictions
            "references": references[:5],     # First 5 references
            "inference_times": inference_times  # Include the inference times
        }
    
        # Print key results
        print(f"\nResults for {model_name}:")
        print(f"ROUGE-1 F1: {metrics['rouge1_f']:.4f}")
        print(f"ROUGE-2 F1: {metrics['rouge2_f']:.4f}")
        print(f"ROUGE-L F1: {metrics['rougeL_f']:.4f}")
        print(f"Average Inference Time: {np.mean([t for t in inference_times if t > 0]):.3f} seconds")
    
        return results, metrics['rouge1_f'], metrics['rouge2_f'], metrics['rougeL_f']
    
    mlflow.set_tracking_uri(tracking_server_arn)
    mlflow.set_experiment(experiment_name)

    import boto3
    import os
    
    # Get AWS credentials from the SageMaker execution environment
    session = boto3.Session()
    credentials = session.get_credentials()
    
    # Set as environment variables
    os.environ['AWS_ACCESS_KEY_ID'] = credentials.access_key
    os.environ['AWS_SECRET_ACCESS_KEY'] = credentials.secret_key
    if credentials.token:
        os.environ['AWS_SESSION_TOKEN'] = credentials.token
    
    # Set region - important for Bedrock
    region = boto3.session.Session().region_name
    os.environ['AWS_REGION'] = region
    
    with mlflow.start_run(run_id=run_id):
        with mlflow.start_run(run_name="QuantitativeModelEvaluation", nested=True):
            mlflow.autolog()
            
            # Initialize the SageMaker client
            sm_client = boto3.client('sagemaker-runtime')
            
            FINETUNED_MODEL_ENDPOINT = endpoint_name # Update with Fine-tuned model endpoint name
            
            # Define the model to evaluate
            model_to_evaluate = {
                "name": "Fine-tuned Qwen-Qwen3-4B-Instruct-2507", 
                "endpoint": FINETUNED_MODEL_ENDPOINT
            }
            # Limit the number of samples to evaluate (for faster execution)
            num_samples = 10
            
            # Log evaluation parameters to MLflow
            mlflow.log_param("evaluation_endpoint", FINETUNED_MODEL_ENDPOINT)
            mlflow.log_param("evaluation_num_samples", num_samples)
            mlflow.log_param("evaluation_timestamp", datetime.now().isoformat())
            
            # Load the test split of the medical-o1 dataset
            try:
                dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train")
                
                max_samples = len(dataset)
                
                dataset = dataset.shuffle().select(range(min(num_samples, max_samples)))
                print(f"Loaded medical-o1-reasoning dataset with {len(dataset)} samples out of {max_samples}")
                
                mlflow.log_param("dataset_name", "FreedomIntelligence/medical-o1-reasoning-SFT")
                mlflow.log_param("dataset_actual_samples", len(dataset))
            except Exception as e:
                error_msg = f"Error loading dataset: {str(e)}"
                print(error_msg)
                raise
            
            # Display a sample from the dataset
            sample = dataset[0]
            
            print("\nQuestion:\n", sample["Question"], "\n\n====\n")
            print("Complex_CoT:\n", sample["Complex_CoT"], "\n\n====\n")
            print("Response:\n", sample["Response"], "\n\n====\n")

            try:
                finetuned_model_results, rouge1_f, rouge2_f, rougeL_f = evaluate_model_on_dataset(model_to_evaluate, dataset)
                print("DUMP")
                json.dumps(finetuned_model_results)
                print(f"ROUGE-1 F1: {rouge1_f}")
                print(f"ROUGE-2 F1: {rouge2_f}")
                print(f"ROUGE-L F1: {rougeL_f}")
                
                # Create and log visualizations if MLflow is enabled
                # Log model card with performance summary
                model_card = f"""
                # Model Evaluation Report
                
                ## Model Information
                - **Model Name**: {model_to_evaluate["name"]}
                - **Endpoint**: {model_to_evaluate["endpoint"]}
                - **Evaluation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                - **Dataset**: FreedomIntelligence/medical-o1-reasoning-SFT
                - **Samples Evaluated**: {len(dataset)}
                
                ## Performance Metrics
                - **ROUGE-1 F1**: {rouge1_f:.4f}
                - **ROUGE-2 F1**: {rouge2_f:.4f}
                - **ROUGE-L F1**: {rougeL_f:.4f}
                 - **Average Inference Time**: {np.mean([t for t in finetuned_model_results["inference_times"] if t > 0]):.3f} seconds
                
                ## Detailed Metrics
                {json.dumps(finetuned_model_results["metrics"], indent=2)}
                """

                with open("/tmp/model_card.md", "w") as f:
                    f.write(model_card)
                
                mlflow.log_artifact("/tmp/model_card.md", "evaluation_summary")
                
                # Create a simple bar chart for ROUGE metrics
                plt.figure(figsize=(10, 6))
                rouge_metrics = {
                    'ROUGE-1 F1': rouge1_f, 
                    'ROUGE-2 F1': rouge2_f, 
                    'ROUGE-L F1': rougeL_f
                }
                plt.bar(rouge_metrics.keys(), rouge_metrics.values())
                plt.title('ROUGE Metrics')
                plt.ylabel('Score')
                plt.ylim(0, 1)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.savefig('/tmp/rouge_metrics.png')
                mlflow.log_artifact('/tmp/rouge_metrics.png', "evaluation_plots")
            
            except Exception as e:
                error_msg = f"Error in model evaluation: {str(e)}\n{traceback.format_exc()}"
                print(error_msg)
                
                # Return at least something even if evaluation fails
                return {"error": str(e), "rougeL_f": 0.0}

    return {"rougeL_f": rougeL_f}