# ### 8. Qualitative Evaluation Step

# After fine-tuning, this step assesses the model's qualitative performance.

from sagemaker.mlops.workflow.function_step import step
from .pipeline_utils import PIPELINE_INSTANCE_TYPE
from .pipeline_utils import SYSTEM_PROMPT
from .pipeline_utils import convert_to_messages


@step(
    name="QualitativeModelEvaluation",
    instance_type='ml.m5.xlarge',
    display_name="Qualitative Model Evaluation",
    keep_alive_period_in_seconds=900,
    dependencies="./eval/requirements.txt"
)
def qualitative_evaluate(
    tracking_server_arn: str,
    experiment_name: str,
    run_id: str,
    endpoint_name: str,
    mlflow_trace_attributes: dict
) -> dict:
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
    from mlflow.entities import SpanType
    from mlflow.metrics.genai import EvaluationExample, make_genai_metric

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

    def create_bedrock_judge_metrics():
        """
        Create custom LLM-as-a-judge metrics using AWS Bedrock Claude as the judge.
        
        Returns:
            list: List of custom metrics for medical evaluation
        """
        
        # Medical Accuracy Metric using Bedrock Claude
        medical_accuracy_examples = [
            EvaluationExample(
                input="What is the first-line treatment for hypertension?",
                output="ACE inhibitors or thiazide diuretics are typically first-line treatments for hypertension.",
                score=4,
                justification="The response correctly identifies evidence-based first-line treatments for hypertension."
            ),
            EvaluationExample(
                input="What causes Type 1 diabetes?",
                output="Type 1 diabetes is caused by autoimmune destruction of pancreatic beta cells.",
                score=5,
                justification="Accurate and concise explanation of Type 1 diabetes pathophysiology."
            ),
            EvaluationExample(
                input="How do you treat a heart attack?",
                output="You should take aspirin and call emergency services immediately.",
                score=2,
                justification="While partially correct, this oversimplifies emergency treatment and misses critical interventions."
            )
        ]
        
        medical_accuracy = make_genai_metric(
            name="medical_accuracy",
            definition=(
                "Medical accuracy measures how factually correct and evidence-based the medical information is. "
                "Consider current medical guidelines, evidence-based practice, and clinical accuracy. "
                "Score 1-5 where 5 is completely accurate and evidence-based."
            ),
            grading_prompt=(
                "Evaluate the medical accuracy of the response on a scale of 1-5:\n"
                "5: Completely accurate, evidence-based, follows current medical guidelines\n"
                "4: Mostly accurate with minor gaps or generalizations\n"
                "3: Generally accurate but missing important details or context\n"
                "2: Partially accurate but contains some medical inaccuracies\n"
                "1: Contains significant medical errors or misinformation\n\n"
                "Question: {input}\n"
                "Response: {output}\n\n"
                "Consider: Is the medical information factually correct? Does it align with current evidence-based practice? "
                "Are there any dangerous inaccuracies or omissions?\n\n"
                "Provide your score as a single integer from 1-5."
            ),
            examples=medical_accuracy_examples,
            version="v1",
            # model="bedrock:/global.anthropic.claude-haiku-4-5-20251001-v1:0",
            model="bedrock:/global.anthropic.claude-sonnet-4-5-20250929-v1:0",
            parameters={
                "anthropic_version": "bedrock-2023-05-31",
                "temperature": 0.0,
                "max_tokens": 1000
            },
            aggregations=["mean", "variance", "p90"],
            greater_is_better=True
        )
        
        # Clinical Reasoning Metric
        clinical_reasoning_examples = [
            EvaluationExample(
                input="A 65-year-old man presents with chest pain. What should be considered?",
                output="Given the patient's age and presentation, we should immediately consider cardiac causes like myocardial infarction, unstable angina, and aortic dissection. The approach should include ECG, cardiac enzymes, chest X-ray, and careful history taking about pain characteristics, onset, and associated symptoms.",
                score=5,
                justification="Excellent clinical reasoning with systematic approach, appropriate differential diagnosis, and logical diagnostic workup."
            ),
            EvaluationExample(
                input="Patient has fever and cough. What's the diagnosis?",
                output="The patient has pneumonia and needs antibiotics.",
                score=2,
                justification="Poor reasoning - jumps to conclusion without proper assessment, history, or consideration of differential diagnosis."
            )
        ]
        
        clinical_reasoning = make_genai_metric(
            name="clinical_reasoning",
            definition=(
                "Clinical reasoning evaluates the logical thought process, differential diagnosis consideration, "
                "and systematic approach to medical problems. Assesses whether the response demonstrates "
                "appropriate medical decision-making skills."
            ),
            grading_prompt=(
                "Evaluate the clinical reasoning quality on a scale of 1-5:\n"
                "5: Excellent reasoning with systematic approach, appropriate differential diagnosis, logical steps\n"
                "4: Good reasoning with mostly sound logic and appropriate considerations\n"
                "3: Adequate reasoning but may miss some important considerations\n"
                "2: Poor reasoning with significant logical gaps or inappropriate approach\n"
                "1: No clear reasoning or completely inappropriate approach\n\n"
                "Question: {input}\n"
                "Response: {output}\n\n"
                "Consider: Does the response show logical thinking? Are differential diagnoses considered? "
                "Is the approach systematic and appropriate for the clinical scenario?\n\n"
                "Provide your score as a single integer from 1-5."
            ),
            examples=clinical_reasoning_examples,
            version="v1",
            # model="bedrock:/global.anthropic.claude-haiku-4-5-20251001-v1:0",
            model="bedrock:/global.anthropic.claude-sonnet-4-5-20250929-v1:0",
            parameters={
                "anthropic_version": "bedrock-2023-05-31",
                "temperature": 0.0,
                "max_tokens": 1000
            },
            aggregations=["mean", "variance", "p90"],
            greater_is_better=True
        )
        
        # Patient Safety Metric
        patient_safety_examples = [
            EvaluationExample(
                input="Patient allergic to penicillin needs antibiotic. What do you recommend?",
                output="Since the patient is allergic to penicillin, I would recommend alternative antibiotics such as macrolides (azithromycin), cephalosporins (if no cross-reactivity), or fluoroquinolones, depending on the specific infection and patient factors.",
                score=5,
                justification="Excellent safety awareness - acknowledges allergy and provides appropriate alternatives."
            ),
            EvaluationExample(
                input="Patient with chest pain asks if they can drive home.",
                output="Sure, if you're feeling better you can drive home.",
                score=1,
                justification="Dangerous advice - patient with chest pain should not drive and needs immediate medical evaluation."
            )
        ]
        
        patient_safety = make_genai_metric(
            name="patient_safety",
            definition=(
                "Patient safety measures whether the response prioritizes patient wellbeing, avoids harmful advice, "
                "considers contraindications, and promotes safe medical practices."
            ),
            grading_prompt=(
                "Evaluate patient safety considerations on a scale of 1-5:\n"
                "5: Prioritizes safety, considers contraindications, promotes safe practices\n"
                "4: Generally safe with minor safety considerations\n"
                "3: Mostly safe but may miss some safety considerations\n"
                "2: Some safety concerns or inappropriate advice\n"
                "1: Potentially dangerous advice or significant safety issues\n\n"
                "Question: {input}\n"
                "Response: {output}\n\n"
                "Consider: Is the advice safe? Are contraindications considered? Could following this advice harm the patient?\n\n"
                "Provide your score as a single integer from 1-5."
            ),
            examples=patient_safety_examples,
            version="v1",
            # model="bedrock:/global.anthropic.claude-haiku-4-5-20251001-v1:0",
            model="bedrock:/global.anthropic.claude-sonnet-4-5-20250929-v1:0",
            parameters={
                "anthropic_version": "bedrock-2023-05-31",
                "temperature": 0.0,
                "max_tokens": 1000
            },
            aggregations=["mean", "variance", "p90"],
            greater_is_better=True
        )
        
        return [medical_accuracy, clinical_reasoning, patient_safety]
    
    def simple_judge_evaluation(predictions, questions, references):
        """
        Simple rule-based evaluation as fallback if LLM-as-a-judge fails.
        """
        scores = []
        
        for pred, question, ref in zip(predictions, questions, references):
            score = 3.0  # Default neutral score
            
            # Simple heuristics for medical evaluation
            if len(pred.split()) < 10:
                score -= 1.0  # Too short responses
            elif len(pred.split()) > 500:
                score -= 0.5  # Overly verbose
            
            # Check for medical keywords
            medical_keywords = ['diagnosis', 'treatment', 'symptom', 'patient', 'clinical', 'medical']
            if any(keyword in pred.lower() for keyword in medical_keywords):
                score += 0.5
            
            # Check for safety considerations
            safety_keywords = ['contraindication', 'allergy', 'caution', 'risk', 'side effect']
            if any(keyword in pred.lower() for keyword in safety_keywords):
                score += 0.5
            
            # Ensure score is in valid range
            score = max(1.0, min(5.0, score))
            scores.append(score)
        
        return {
            'medical_accuracy': np.mean(scores),
            'clinical_reasoning': np.mean(scores),
            'patient_safety': np.mean(scores),
            'overall_quality': np.mean(scores)
        }
    
    def evaluate_model_qualitatively(model_config, dataset):
        """
        Evaluate a fine-tuned model using LLM-as-a-judge metrics with fallback.
        """
        # time.sleep(60)
        model_name = model_config["name"]
        endpoint_name = model_config["endpoint"]
        
        print(f"\nPerforming qualitative evaluation for model: {model_name} on endpoint: {endpoint_name}")
        
        # Generate predictions for the dataset
        predictions = []
        questions = []
        references = []
        inference_times = []
        failed_generations = 0
        
        for example in tqdm(dataset, desc="Generating responses for evaluation"):
            question = example["Question"]
            reference = "\n".join([example["Complex_CoT"], example["Response"]])
            
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
                
                if response is None:
                    prediction = "Error generating response."
                    failed_generations += 1
                else:
                    prediction = str(response).strip()
                
                inference_times.append(inference_time)
                
            except Exception as e:
                print(f"Error invoking SageMaker endpoint {endpoint_name}: {e}")
                prediction = "Error generating response."
                failed_generations += 1
                inference_times.append(-1)
            
            predictions.append(prediction)
            questions.append(question)
            references.append(reference)
        
        # Log basic generation metrics
        mlflow.log_metric("qualitative_failed_generations", failed_generations)
        mlflow.log_metric("qualitative_failure_rate", failed_generations / len(dataset) if len(dataset) > 0 else 0)
        
        # Try LLM-as-a-judge evaluation, fallback to simple evaluation
        try:
            print("Attempting LLM-as-a-judge evaluation using AWS Bedrock...")
            
            # Prepare data for MLflow evaluation
            eval_data = pd.DataFrame({
                "inputs": questions,
                "outputs": predictions,
                "targets": references
            })
            
            # Create custom metrics
            custom_metrics = create_bedrock_judge_metrics()
            
            # Run MLflow evaluation
            eval_results = mlflow.evaluate(
                data=eval_data,
                targets="targets",
                predictions="outputs",
                extra_metrics=custom_metrics,
            )
            print(f"Raw evaluation results: {eval_results.metrics}")
            
            # Extract metric results
            metric_results = {}
            for metric_name in ["medical_accuracy/v1/mean", "clinical_reasoning/v1/mean", "patient_safety/v1/mean"]:
                if metric_name in eval_results.metrics:
                    base_name = metric_name.split('/')[0]
                    metric_results[base_name] = eval_results.metrics[metric_name]
                    if not np.isnan(metric_results[base_name]):
                        mlflow.log_metric(f"qualitative_{base_name}", metric_results[base_name])
                    else: 
                        mlflow.log_metric(f"qualitative_{base_name}", 0.0)
            
            print("LLM-as-a-judge evaluation completed successfully!")
            
        except Exception as e:
            print(f"LLM-as-a-judge evaluation failed: {str(e)}")
            print("Falling back to simple rule-based evaluation...")
            
            # Fallback to simple evaluation
            metric_results = simple_judge_evaluation(predictions, questions, references)
            
            for metric_name, score in metric_results.items():
                if not np.isnan(score):
                    mlflow.log_metric(f"qualitative_{metric_name}", score)
                else:
                    mlflow.log_metric(f"qualitative_{metric_name}", 0.0)
        
        # Create evaluation summary
        evaluation_details = []
        for i, (pred, question, ref) in enumerate(zip(predictions[:5], questions[:5], references[:5])):
            evaluation_details.append({
                "question": question,
                "prediction": pred[:500] + ("..." if len(pred) > 500 else ""),
                "reference": ref[:500] + ("..." if len(ref) > 500 else ""),
            })
        
        # Save detailed results
        detailed_df = pd.DataFrame(evaluation_details)
        temp_csv = f"/tmp/qualitative_eval_detailed_{uuid.uuid4().hex[:8]}.csv"
        detailed_df.to_csv(temp_csv, index=False)
        mlflow.log_artifact(temp_csv, "qualitative_evaluation")
        
        # Create simple visualization
        plt.figure(figsize=(10, 6))
        metric_names = list(metric_results.keys())
        metric_values = list(metric_results.values())
        plt.bar(metric_names, metric_values, color=['blue', 'green', 'red', 'orange'])
        plt.title('Qualitative Evaluation Scores')
        plt.ylabel('Score (1-5)')
        plt.ylim(1, 5)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('/tmp/qualitative_metrics.png', dpi=300, bbox_inches='tight')
        mlflow.log_artifact('/tmp/qualitative_metrics.png', "qualitative_evaluation")
        
        raw_avg_medical_accuracy = metric_results.get("medical_accuracy", metric_results.get("overall_quality", 3.0))

        # NaN guard: the LLM-as-a-judge path (mlflow.evaluate) can silently return NaN
        # when the judge model errors out per-row, without raising. A NaN here serializes
        # into results.json and breaks the downstream ConditionStep with
        # "Incompatible types in BinaryCondition left type [String], right type [Float]".
        # Coerce to a numeric float and fall back to the rule-based evaluator if invalid.
        try:
            avg_medical_accuracy = float(raw_avg_medical_accuracy)
            if np.isnan(avg_medical_accuracy):
                raise ValueError("Judge returned NaN")
        except (TypeError, ValueError):
            print("WARNING: qualitative judge produced NaN/invalid score; "
                  "falling back to rule-based evaluation for avg_medical_accuracy.")
            fallback_scores = simple_judge_evaluation(predictions, questions, references)
            avg_medical_accuracy = float(fallback_scores.get("medical_accuracy", 3.0))
            if np.isnan(avg_medical_accuracy):
                avg_medical_accuracy = 0.0
            mlflow.log_metric("qualitative_medical_accuracy_fallback", avg_medical_accuracy)

        return {
            "model_name": model_name,
            "endpoint_name": endpoint_name, 
            "num_samples": len(dataset),
            "metrics": metric_results,
            "evaluation_details": evaluation_details,
            "avg_medical_accuracy": avg_medical_accuracy
        }
    
    # Main evaluation logic
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
        with mlflow.start_run(run_name="QualitativeModelEvaluation", nested=True):
            mlflow.set_tag("component", "qualitative_model_evaluation")
            
            # Initialize the SageMaker client
            sm_client = boto3.client('sagemaker-runtime')
            
            # Define the model to evaluate
            model_to_evaluate = {
                "name": "Fine-tuned Qwen-Qwen3-4B-Instruct-2507", 
                "endpoint": endpoint_name
            }
            
            # Limit samples for faster execution
            num_samples = 10
            
            # Log evaluation parameters
            mlflow.log_param("qualitative_evaluation_endpoint", endpoint_name)
            mlflow.log_param("qualitative_evaluation_num_samples", num_samples)
            mlflow.log_param("qualitative_evaluation_timestamp", datetime.now().isoformat())
            mlflow.log_param("llm_judge_model", "bedrock:/global.anthropic.claude-sonnet-4-5-20250929-v1:0")
            
            # Load the test dataset
            try:
                dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train")
                max_samples = len(dataset)
                dataset = dataset.shuffle().select(range(min(num_samples, max_samples)))
                print(f"Loaded medical-o1-reasoning dataset with {len(dataset)} samples for qualitative evaluation")
                
                mlflow.log_param("qualitative_dataset_name", "FreedomIntelligence/medical-o1-reasoning-SFT") 
                mlflow.log_param("qualitative_dataset_actual_samples", len(dataset))
            except Exception as e:
                error_msg = f"Error loading dataset for qualitative evaluation: {str(e)}"
                print(error_msg)
                raise
            
            try:
                # Perform qualitative evaluation
                qualitative_results = evaluate_model_qualitatively(model_to_evaluate, dataset)
                
                avg_medical_accuracy = qualitative_results["avg_medical_accuracy"]

                # Final safety net: never serialize NaN into results.json, which would
                # break the downstream EvaluationGate ConditionStep (String vs Float).
                try:
                    avg_medical_accuracy = float(avg_medical_accuracy)
                    if np.isnan(avg_medical_accuracy):
                        avg_medical_accuracy = 0.0
                except (TypeError, ValueError):
                    avg_medical_accuracy = 0.0

                print(f"\nQualitative evaluation completed!")
                print(f"Average Medical Accuracy: {avg_medical_accuracy:.3f}")
                
                return {"avg_medical_accuracy": avg_medical_accuracy}
                
            except Exception as e:
                error_msg = f"Error in qualitative model evaluation: {str(e)}\n{traceback.format_exc()}"
                print(error_msg)
                return {"error": str(e), "avg_medical_accuracy": 0.0}