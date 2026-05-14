import os
import json
import argparse
import traceback
from autoeval.evaluator import Evaluator
from autoeval.clients import CLIENT_DICT


def load_blocks(path: str) -> list[list[str]]:
    """Load blank-line separated blocks from the log file."""
    blocks, block = [], []
    for line in open(path, 'r'):
        if line.strip() == "":
            blocks.append(block)
            block = []
        else:
            if line.strip():
                block.append(line.strip())

    valid_blocks = []
    for block in blocks:
        is_valid = False
        for line in block:
            if line.startswith('action:'):
                is_valid = True
                break
            if 'browsergym.experiments.loop' in line and 'Running experiment' not in line:
                is_valid = True
                break
        if is_valid:
            valid_blocks.append(block)
    
    assert len(valid_blocks) % 2 == 0
    return valid_blocks

def remove_invalid_steps(actions: list[str]) -> list[str]:
    """Remove invalid steps from the action sequence."""
    valid_actions = []
    for a in actions:
        if "click(" in a:
            arg = a[a.index("(")+1: a.index(")")]
            # if type(eval(arg)) == str:
            if type(arg) == str:
                valid_actions.append(a)
        elif "fill(" in a:
            arg = a[a.index("(")+1: a.index(",")].strip()
            # if type(eval(arg)) == str:
            if type(arg) == str:
                valid_actions.append(a)
        else:
            valid_actions.append(a)
    return valid_actions

def extract_think_and_action(path: str) -> tuple[list[str], list[str]]:
    """Extract the task trajectory from the log file."""
    with open(path, 'r') as exp_log:
        content = exp_log.read()
    if 'browsergym.experiments.loop - INFO - action:' in content:
        content = content.replace('browsergym.experiments.loop - INFO - action:', 'browsergym.experiments.loop - INFO - \n\naction:')
        with open(path, 'w') as exp_log:
            exp_log.write(content)

    blocks = load_blocks(path)
    think_list, action_list = [], []
    for i in range(1, len(blocks), 2):
        # action
        b = blocks[i]
        # print(f"Here is action block:\n{b}")
        actions = remove_invalid_steps(b[1:])
        # print(f"Here is actions:\n{actions}")
        if len(actions) == 0: continue
        action_list.append(actions)
        # think
        b = blocks[i-1]
        # print(f"Here is think block:\n{b}")
        think_line = b[-1]
        for line in b:
            if 'browsergym.experiments.loop' in line:
                think_line = line
        idx = think_line.index("browsergym.experiments.loop - INFO -")
        # print(f"Here is think :\n{think_line[idx+36: ].strip()}")
        think_list.append(think_line[idx+36: ].strip())
    
    assert len(think_list) == len(action_list)
    
    # TODO: merge same actions
    return think_list, action_list

def extract_response(action: str) -> str:
    s, e = action.index("(")+1, action.index(")")
    return action[s: e]


def process_sample(
    idx: str, traj_info: dict, log_save_path,
    model: str, eval_version: str,
) -> list[dict]:
    clients = {model: CLIENT_DICT[model](model_name=model)}
    evaluator = Evaluator(clients, log_save_path=log_save_path + "/trajs")
    try:
        out, _ = evaluator(traj_info, model, eval_version)
        eval_result = None
        if out["status"].lower() == "success": eval_result = True
        else: eval_result = False
        return [{
                "idx": idx,
                "gt": traj_info["eval"],
                "rm": eval_result,
                "thoughts": out["thoughts"], 
                "uid": traj_info["traj_name"],
        }]
    except Exception as e:
        print(f"Error on {idx}, {e}")
        print(traceback.format_exc())
        return {
            "idx": idx,
            "gt": traj_info["eval"],
            "rm": None,
            "thoughts": None, 
            "uid": traj_info["traj_name"],
        }


def main():
    # load task config
    task_id = args.result_dir.split('/')[-1].split(".")[1]
    config_path = os.path.join("config_files", f"{task_id}.json")
    config = json.load(open(config_path))

    # load trajectory log
    log_path = os.path.join(args.result_dir, "experiment.log")
    think_list, action_list = extract_think_and_action(log_path)
    actions = [act for acts in action_list for act in acts]
    if "send_msg_to_user" in action_list[-1][0]:
        response = extract_response(action_list[-1][0])
    else:
        response = ""
    
    # load summary info
    summary_path = os.path.join(args.result_dir, "summary_info.json")
    summary = json.load(open(summary_path, 'r'))
    # collect traj info
    image_paths = [
        os.path.join(args.result_dir, f) for f in os.listdir(args.result_dir) 
        if f.startswith("screenshot_step_") and f.endswith(".png")
    ]
    image_paths = sorted(image_paths, key=lambda x: int(x.split('/')[-1].split("_")[-1].split(".")[0]))
    traj_info = {
        "intent": config["intent"],
        "response": response,
        "captions": think_list,
        "actions": actions,
        "traj_name": config["task_id"],
        "image_paths": image_paths,
        "images": image_paths,
        "eval": summary["cum_reward"]
    }

    # evaluate trajectory
    log_save_path = os.path.join(f"autoeval/{args.log_dir}", args.result_dir.split('/')[-1])
    print("Log Save Path:", log_save_path)
    if not os.path.exists(log_save_path):
        os.makedirs(log_save_path)
        os.makedirs(log_save_path + "/trajs")
    eval_info = process_sample(
        idx=config["task_id"], traj_info=traj_info,
        log_save_path=log_save_path, 
        model=args.model, eval_version=args.prompt,
    )
    output_eval_path = os.path.join(args.result_dir, f"{args.model}_autoeval.json")
    json.dump(eval_info, open(output_eval_path, 'w'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True,
                        help="Path to the result directory, e.g., 'webarena.0'.")
    # autoeval
    parser.add_argument("--model", type=str, default="gpt-4o",
                        choices=["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini"])
    parser.add_argument("--prompt", type=str, default="text",
                        choices=["text", "vision"])
    parser.add_argument("--log_dir", type=str, default="log")

    args = parser.parse_args()

    if args.model == "gpt-4o" and args.prompt != "vision":
        print(f"Waring: use vision prompt by default for {args.model}.")
        args.prompt = "vision"

    main()
