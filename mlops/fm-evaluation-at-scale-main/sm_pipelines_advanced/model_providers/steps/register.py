from lib.utils import find_model_by_name


def register(model_list, model_registry_config, selection_step_ret=None, *args):

    # Search model:
    if selection_step_ret is None:
        best_model_name = model_list[0].config["name"]
        model = model_list[0]
    else:
        best_model_name = selection_step_ret["best_model_name"]
        model = find_model_by_name(model_list, best_model_name)

    eval_output = None
    for results in args:
        if results["model_name"] == best_model_name:
            eval_output = results["evaluation_output"]
            break

    model_registry_ret = model.register(model, model_registry_config, eval_output)

    return model_registry_ret
