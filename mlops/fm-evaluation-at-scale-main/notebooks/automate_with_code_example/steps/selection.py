def selection(*args):
    
    # Select best model.
    # We are evaluating Factual Knowledge thus we are looking for the maximum score
    
    max_score = -1

    for arg in args:
        print(arg)
        model_name = arg['model_name']

        eval_result = arg['evaluation_output']
        eval_output = eval_result[0][0]
        eval_score = eval_output.dataset_scores[0].value
        print(eval_score)

        if eval_score > max_score:
            max_score = eval_score
            best_model_name = model_name

    return {"best_model_name": best_model_name}
            