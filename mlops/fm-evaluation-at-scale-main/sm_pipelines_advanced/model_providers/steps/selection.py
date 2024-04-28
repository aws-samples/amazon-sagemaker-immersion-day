def selection(*args):

    max_score = -1
    best_model = None
    best_model_evaluation = None

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
            best_model_evaluation = eval_result

    print(best_model_name)
    print(max_score)
    
    return {"best_model_name": best_model_name}
