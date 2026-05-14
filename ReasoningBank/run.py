"""
WARNING DEPRECATED WILL BE REMOVED SOON
"""

import os
import argparse
from pathlib import Path

from browsergym.experiments import ExpArgs, EnvArgs

from agents.legacy.agent import GenericAgentArgs
from agents.legacy.dynamic_prompting import Flags
from agents.legacy.utils.chat_api import ChatModelArgs
import pandas as pd
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity


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
        "--model_name",
        type=str,
        default="openai/gpt-4o",
        help="Model name for the chat model.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="openended",
        help="Name of the Browsergym task to run. If 'openended', you need to specify a 'start_url'",
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
        "--bank_path",
        type=str,
        default=None,
        help="Path to the memory file to load for the agent.",
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


def find_most_similar_task(args, tid) -> str:
    """
    返回最相似的一个task的memory_item
    """
    # 1. 读取bank文件获取task_id列表
    with open(args.bank_path, 'r', encoding='utf-8') as f:
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


def main():
    args = parse_args()
    dir_path = '/'.join(args.bank_path.split('/')[:-1])
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if (args.bank_path is not None) and (not os.path.exists(args.bank_path)):
        with open(args.bank_path, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=4)
        # open(args.bank_path, "w").close()

    env_args = EnvArgs(
        task_name=args.task_name,
        task_seed=None,
        max_steps=args.max_steps,
        headless=args.headless,
        viewport={"width": 1500, "height": 1280},
        slow_mo=args.slow_mo,
    )

    if args.task_name == "openended":
        env_args.wait_for_user_message = True
        env_args.task_kwargs = {"start_url": args.start_url}
    
    tid = int(args.task_name.split('.')[1])
    if args.use_bank:
        instruction = 'Below are some memory items that I accumulated from past interaction from the environment that may be helpful to solve the task. You can use it when you feel it’s relevant. In each step, please first explicitly discuss if you want to use each memory item or not, and then take action.'
        query, trajectory, memory_item = find_most_similar_task(args, tid)
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
                model_name=args.model_name,
                max_total_tokens=128_000,  # "Maximum total tokens for the chat model."
                max_input_tokens=126_000,  # "Maximum tokens for the input to the chat model."
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
                bank_path=args.bank_path,
                memory_item=memory_item
            ),
            faithfulness_experiment=args.faithfulness_experiment,
            fewshot_modification_type=args.fewshot_modification_type,
            insights_modification_type=args.insights_modification_type,
            log_path = args.log_path
        ),
    )

    exp_args.prepare(Path(f"./{args.result_dir}"))
    exp_args.run()

    os.rename(exp_args.exp_dir, f"{args.result_dir}/{args.task_name}")


if __name__ == "__main__":
    main()
