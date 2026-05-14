import os
import json
import random
import argparse

import openai
openai.api_key = os.environ["OPENAI_API_KEY"]
from openai import OpenAI
client = OpenAI()

# %% load examples
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

def format_trajectory(think_list: list[str], action_list: list[list[str]]) -> str:
    trajectory = []
    for t, a in zip(think_list, action_list):
        acts = '\n'.join(a)
        trajectory.append(f"<think>\n{t}\n</think>\n<action>\n{acts}\n</action>")
    return '\n\n'.join(trajectory)

def random_group_sample(d: dict, n) -> list:
    """Randomly sample n groups from the dictionary."""
    return [ex for v in d.values() for ex in random.sample(v, min(n, len(v)))]

def llm_generate(examples: dict, args, verbose: bool = False, is_success: bool = True):
    trajectory = format_trajectory(examples["think_list"], examples["action_list"])
    input_prompt = f"Query: {examples['query']}\n\nTrajectory:\n{trajectory}"

    """Call gpt model to generate memory_item."""
    if is_success:
        prompt = '\n\n'.join([args.SUCCEED_PROMPT, input_prompt])
    else:
        prompt = '\n\n'.join([args.FAILED_PROMPT, input_prompt])
    if verbose: print("Prompt:\n", prompt, '\n\n')

    response = client.chat.completions.create(
        model=args.model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
        max_tokens=2048,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    response = response.choices[0].message.content

    if verbose: print(response)
    return response, trajectory


def induce_prompt(args, task_id, result_dir, bank_path):
    # result_dir = result_dir.split() # e.g., ["results/webarena.0", ...]
    record_path = os.path.join(result_dir, f"webarena.{task_id}", "gpt-4o_autoeval.json")
    log_path = os.path.join(result_dir, f"webarena.{task_id}", "experiment.log")
    if not os.path.exists(record_path):
        print("Current task autoeval doesn't exist!")
    else:
        record = json.load(open(record_path))
        if isinstance(record, list):
            is_success = record[0]["rm"]        # 获取当前任务是否成功

            config_path = os.path.join("config_files", f"{task_id}.json")
            config = json.load(open(config_path))
            query = config["intent"]            # 获得qurey
            try:
                think_list, action_list = extract_think_and_action(log_path)    # 提取轨迹
            except Exception as e:
                print(e)

            wdict = {"query": query, "think_list": think_list, "action_list": action_list}
            memory_item, trajectory = llm_generate(wdict, args, False, is_success)
            
            bank_dict = {"task_id": task_id, "query": query, "success": is_success,
                "trajectory": trajectory, "memory_item": memory_item}

            if bank_path is None:
                website = config["sites"][0]  # assumes all results are about the same website
                args.output_path = f"bank/{website}_neural.txt"
                print(f"[Warning] no output path specified, using '{args.output_path}' by default")
                
            if os.path.exists(bank_path):
                # 如果文件存在，先读取现有数据
                with open(bank_path, 'r', encoding='utf-8') as f:
                    try:
                        existing_data = json.load(f)
                    except json.JSONDecodeError:
                        existing_data = []  # 处理空文件的情况
            else:
                # 如果文件不存在，初始化数据为一个空列表
                existing_data = []

            # 将新的数据追加到现有数据
            existing_data.append(bank_dict)
            # 写回更新后的数据
            with open(bank_path, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=4)


