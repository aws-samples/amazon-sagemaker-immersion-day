def selection(*args):
    max_score = -1
    best_model = None
    best_model_evaluation = None

    for arg in args:
        print(arg)
        model_config = arg["model_config"]
        eval_result = arg["eval_output"]
        print(eval_result)

        eval_output = eval_result[0][0]
        print(eval_output)

        eval_score = eval_output.dataset_scores[0].value

        print(eval_score)
        if eval_score > max_score:
            max_score = eval_score
            best_model = model_config
            best_model_evaluation = eval_result

        # get the model name with highest score

        print(best_model)

    return {"model_config": best_model, "eval_output": best_model_evaluation}
