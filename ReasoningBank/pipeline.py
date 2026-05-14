import os
import json
import argparse
from subprocess import Popen
import subprocess
from threading import Timer
import shutil
import time
import argparse
from pathlib import Path
from browsergym.experiments import ExpArgs, EnvArgs
from agents.legacy.agent import GenericAgentArgs
from agents.legacy.dynamic_prompting import Flags
from agents.legacy.utils.chat_api import ChatModelArgs
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from func_timeout import func_set_timeout
from threading import Thread
from induce_prompt import induce_prompt

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def parse_args():
    parser = argparse.ArgumentParser(description="Run experiment with hyperparameters.")

    parser.add_argument(
        "--website", 
        type=str, 
        required=True,
        choices=["shopping", "shopping_admin", "gitlab", "reddit", "map"]
    )

    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)

    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-3.5-turbo",
        help="Model name for the chat model.",
    )
    parser.add_argument(
        "--start_url",
        type=str,
        default="https://www.google.com",
        help="Starting URL (only for the openended task).",
    )
    parser.add_argument(
        "--slow_mo", type=int, default=30, help="Slow motion delay for the playwright actions."
    )
    parser.add_argument(
        "--headless",
        type=str2bool,
        default=True,
        help="Run the experiment in headless mode (hides the browser windows).",
    )
    parser.add_argument(
        "--demo_mode",
        type=str2bool,
        default=True,
        help="Add visual effects when the agents performs actions.",
    )
    parser.add_argument(
        "--use_html", type=str2bool, default=False, help="Use HTML in the agent's observation space."
    )
    parser.add_argument(
        "--use_ax_tree",
        type=str2bool,
        default=True,
        help="Use AX tree in the agent's observation space.",
    )
    parser.add_argument(
        "--use_screenshot",
        type=str2bool,
        default=True,
        help="Use screenshot in the agent's observation space.",
    )
    parser.add_argument(
        "--multi_actions", type=str2bool, default=True, help="Allow multi-actions in the agent."
    )
    parser.add_argument(
        "--action_space",
        type=str,
        default="bid",
        choices=["python", "bid", "coord", "bid+coord", "bid+nav", "coord+nav", "bid+coord+nav"],
        help="",
    )
    parser.add_argument(
        "--use_history",
        type=str2bool,
        default=True,
        help="Use history in the agent's observation space.",
    )
    parser.add_argument(
        "--use_thinking",
        type=str2bool,
        default=True,
        help="Use thinking in the agent (chain-of-thought prompting).",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=10,
        help="Maximum number of steps to take for each task.",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default='results',
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default='logs/log.txt',
    )

    parser.add_argument(
        "--faithfulness_experiment",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--fewshot_modification_type",
        type=str,
        default='None',
    )
    parser.add_argument(
        "--insights_modification_type",
        type=str,
        default='None',
    )
    parser.add_argument(
        "--use_bank",
        type=str2bool,
        default=True,
    )

    return parser.parse_args()

def find_most_similar_task(args, tid, bank_path) -> str:
    """
    返回最相似的一个task的memory_item
    """
    # 1. 读取bank文件获取task_id列表
    with open(bank_path, 'r', encoding='utf-8') as f:
        bank_data = json.load(f)
    
    # task_ids = [int(item['task_id']) for item in bank_data if 'task_id' in item]
    task_ids = list(set(int(item['task_id']) for item in bank_data if 'task_id' in item and int(item['task_id']) != tid))

    if not task_ids:
        print("No current memory.")
        return '', '', ''
    
    # 2. 读取嵌入向量
    embeddings_df = pd.read_csv("embeddings/intent_embeddings.csv", header=None)
    
    # 检查tid有效性
    if tid >= len(embeddings_df) or tid < 0:
        print("Invalid current task id.")
        return '', '', ''
    
    # 获取目标向量和bank向量
    target_embedding = embeddings_df.iloc[tid].values.reshape(1, -1)
    
    # 过滤有效task_id
    valid_task_ids = [task_id for task_id in task_ids if task_id < len(embeddings_df) and task_id >= 0]
    
    if not valid_task_ids:
        return '', '', ''
    
    bank_embeddings = embeddings_df.iloc[valid_task_ids].values
    
    # 3. 计算相似度
    similarities = cosine_similarity(target_embedding, bank_embeddings)
    most_similar_idx = np.argmax(similarities[0])

    similar_tid = valid_task_ids[most_similar_idx]
    query = ''
    trajectory = ''
    memory_item = ''
    for item in bank_data:
        if int(item['task_id']) == similar_tid:
            memory_item = item['memory_item']
            query = item['query']
            trajectory = item['trajectory']

    return query, trajectory, memory_item

# @func_set_timeout(600)
def run(args, task_name, bank_path, result_dir, log_path):
    dir_path = '/'.join(bank_path.split('/')[:-1])
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if (bank_path is not None) and (not os.path.exists(bank_path)):
        with open(bank_path, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=4)

    env_args = EnvArgs(
        task_name=task_name,
        task_seed=None,
        max_steps=args.max_steps,
        headless=args.headless,
        viewport={"width": 1500, "height": 1280},
        slow_mo=args.slow_mo,
    )

    if task_name == "openended":
        env_args.wait_for_user_message = True
        env_args.task_kwargs = {"start_url": args.start_url}
    
    tid = int(task_name.split('.')[1])
    if args.use_bank:
        instruction = 'Below are some memory items that I accumulated from past interaction from the environment that may be helpful to solve the task. You can use it when you feel it’s relevant. In each step, please first explicitly discuss if you want to use each memory item or not, and then take action.'
        query, trajectory, memory_item = find_most_similar_task(args, tid, bank_path)
        if memory_item:
            memory_item = instruction + '\n\n' + memory_item
        else:
            memory_item = ''
    else:
        memory_item = ''
    
    # print('-' * 50)
    # print(f"Here is memory_item:\n{memory_item}\n")
    # print('-' * 50)

    exp_args = ExpArgs(
        env_args=env_args,
        agent_args=GenericAgentArgs(
            chat_model_args=ChatModelArgs(
                model_name= args.model_name if "Qwen/Qwen3" in args.model_name else 'openai/' + args.model_name,
                max_total_tokens=40960,  # "Maximum total tokens for the chat model."
                max_input_tokens=38960,  # "Maximum tokens for the input to the chat model."
                max_new_tokens=2_000,  # "Maximum total tokens for the chat model."
                temperature=0.7,
            ),
            flags=Flags(
                use_html=args.use_html,
                use_ax_tree=args.use_ax_tree,
                use_thinking=args.use_thinking,  # "Enable the agent with a memory (scratchpad)."
                use_error_logs=True,  # "Prompt the agent with the error logs."
                use_memory=False,  # "Enables the agent with a memory (scratchpad)."
                use_history=args.use_history,
                use_diff=False,  # "Prompt the agent with the difference between the current and past observation."
                use_past_error_logs=True,  # "Prompt the agent with the past error logs."
                use_action_history=True,  # "Prompt the agent with the action history."
                multi_actions=args.multi_actions,
                use_abstract_example=True,  # "Prompt the agent with an abstract example."
                use_concrete_example=True,  # "Prompt the agent with a concrete example."
                use_screenshot=args.use_screenshot,
                enable_chat=True,
                demo_mode="default" if args.demo_mode else "off",
                bank_path=bank_path,
                memory_item=memory_item
            ),
            faithfulness_experiment=args.faithfulness_experiment,
            fewshot_modification_type=args.fewshot_modification_type,
            insights_modification_type=args.insights_modification_type,
            log_path = log_path,
            tid = tid
        ),
    )

    exp_args.prepare(Path(f"./{result_dir}"))
    exp_args.run()

    os.rename(exp_args.exp_dir, f"{result_dir}/{task_name}")

def main(args):
    # collect examples
    config_files = [
        os.path.join("config_files", f) for f in os.listdir("config_files")
        if f.endswith(".json") and f.split(".")[0].isdigit()
    ]
    config_files = sorted(config_files, key=lambda x: int(x.split("/")[-1].split(".")[0]))
    config_list = [json.load(open(f)) for f in config_files]
    config_flags = [config["sites"][0] == args.website and len(config["sites"]) == 1 for config in config_list]
    task_ids = [config["task_id"] for config, flag in zip(config_list, config_flags) if flag]
    
    model_flag = args.model_name.split("/")[-1]
    result_dir = f'results/{model_flag}'
    log_dir = f'log/{model_flag}'
    workflow_dir = f'{model_flag}/'
    log_path = f'logs/{model_flag}/'
    if args.faithfulness_experiment:
        if args.fewshot_modification_type == 'shuffle':
            result_dir += '/fewshot/shuffle'
            log_dir += '/fewshot/shuffle'
            workflow_dir += 'fewshot/shuffle'
            log_path += 'fewshot_shuffle_'
        elif args.fewshot_modification_type == 'irrelevant':
            result_dir += '/fewshot/irrelevant'
            log_dir += '/fewshot/irrelevant'
            workflow_dir += 'fewshot/irrelevant'
            log_path += 'fewshot_irrelevant_'
        elif args.insights_modification_type == 'wo':
            result_dir += '/insights/wo'
            log_dir += '/insights/wo'
            workflow_dir += 'insights/wo'
            log_path += 'insights_wo_'
        elif args.insights_modification_type == 'corrupt':
            result_dir += '/insights/corrupt'
            log_dir += '/insights/corrupt'
            workflow_dir += 'insights/corrupt'
            log_path += 'insights_corrupt_'
        elif args.insights_modification_type == 'irrelevant':
            result_dir += '/insights/irrelevant'
            log_dir += '/insights/irrelevant'
            workflow_dir += 'insights/irrelevant'
            log_path += 'insights_irrelevant_'
        elif args.insights_modification_type == 'filler_tokens':
            result_dir += '/insights/filler_tokens'
            log_dir += '/insights/filler_tokens'
            workflow_dir += 'insights/filler_tokens'
            log_path += 'insights_filler_tokens_'
    
    if not args.use_bank:
        result_dir += '/nobank'
        log_dir += '/nobank'
        workflow_dir += 'nobank'
        log_path += 'nobank'
    log_path += args.website + '.txt'
    bank_path = f"bank/{workflow_dir}/{args.website}.json"

    index = 0
    for i in range(len(task_ids)):
        if task_ids[i] >= args.start_index:
            index = i
            break
    if args.end_index == None: args.end_index = len(task_ids)
    
    for _ in range(3):
        for tid in task_ids[index: args.end_index]:
            record_path = os.path.join(result_dir, f"webarena.{tid}", "gpt-4o_autoeval.json")
            if os.path.exists(record_path):
                continue
            # step 1: run inference
            if os.path.exists(f'{result_dir}/webarena.{tid}'):
                for attempt in range(3):
                    try:
                        shutil.rmtree(f'{result_dir}/webarena.{tid}')
                    except Exception as e:
                        time.sleep(1)
            run(args, f"webarena.{tid}", bank_path, result_dir, log_path)
            # t1 = Thread(target=run, args=(args, f"webarena.{tid}", bank_path, result_dir, log_path))
            # t1.start()
            # t1.join()
            
            # step 2: run evaluation
            process = Popen([
                "python", "-m", "autoeval.evaluate_trajectory",
                "--result_dir", f"{result_dir}/webarena.{tid}",
                "--log_dir", f"{log_dir}",
                "--model", "gpt-4o"  # f"{args.model_name}"
            ])
            process.wait()

            # step 3: update memory
            induce_prompt(args, tid, result_dir, bank_path)


if __name__ == "__main__":
    args = parse_args()
    args.SUCCEED_PROMPT = open("prompt/succeed_prompt.txt", 'r').read()
    args.FAILED_PROMPT = open("prompt/failed_prompt.txt", 'r').read()

    main(args)