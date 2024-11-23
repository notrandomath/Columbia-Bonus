import os
import json
import argparse

from agentorg.evaluation.simulate_first_pass_convos import simulate_conversations
from agentorg.evaluation.extract_conversation_info import extract_task_completion_metrics
from agentorg.evaluation.simulate_second_pass_convos import get_labeled_convos
from agentorg.utils.model_config import MODEL


def evaluate(config):
    model_api = config['model_api']
    model_params = config['model_params']
    synthetic_data_params = config['synthetic_data_params']
    
    # first pass
    first_pass_data = simulate_conversations(model_api, model_params, synthetic_data_params, config)

    # extract items from first pass data
    bot_goal = config['builder_objective'][0]
    goal_metrics = extract_task_completion_metrics(first_pass_data, bot_goal)

    # second pass
    labeled_convos = get_labeled_convos(first_pass_data, model_api, synthetic_data_params, model_params, config)
    return first_pass_data, labeled_convos, goal_metrics


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_api', type=str)
    parser.add_argument('--model_params', type=dict, default={})
    parser.add_argument('--num_convos', type=int, default=5)
    parser.add_argument('--num_goals', type=int, default=5)
    parser.add_argument('--max_turns', type=int, default=5)
    parser.add_argument('--documents_dir', type=str)
    parser.add_argument('--config', type=str)
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--model', type=str, default=MODEL["model_type_or_path"])
    args = parser.parse_args()

    MODEL["model_type_or_path"] = args.model

    assert args.model_api is not None, "Model api must be provided"
    assert args.config is not None, "Config file must be provided"
    assert args.documents_dir is not None, "Documents directory must be provided"
    if not args.output_dir:
        args.output_dir = args.documents_dir
    
    if not os.path.exists(os.path.join(args.output_dir, 'eval')):
        os.makedirs(os.path.join(args.output_dir, 'eval'), exist_ok=True)

    config = json.load(open(args.config))
    config['model_api'] = args.model_api
    config['documents_dir'] = args.documents_dir
    config['model_params'] = args.model_params
    config['synthetic_data_params'] = {'num_convos': args.num_convos, 'num_goals': args.num_goals, 
                                       'max_turns': args.max_turns}

    first_pass_data, final_convos, goal_metrics = evaluate(config)

    with open(os.path.join(args.output_dir, 'eval', 'simulate_data.json'), 'w') as f:
        json.dump(first_pass_data, f, indent=4)

    with open(os.path.join(args.output_dir, 'eval', 'labeled_data.json'), 'w') as f:
        json.dump(final_convos, f, indent=4)
    
    with open(os.path.join(args.output_dir, 'eval', 'goal_completion.json'), 'w') as f:
        json.dump(goal_metrics, f, indent=4)