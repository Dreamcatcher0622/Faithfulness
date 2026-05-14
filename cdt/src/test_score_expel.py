import sys, os, argparse
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import AutoTokenizer, AutoModelForCausalLM
from loguru import logger
import itertools
from peft import peft_model, PeftModelForCausalLM
import numpy as np
import datasets
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logger.info(sys.path)
from retrieval_head_detection import SentenceSampler
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
from utils import *
import time

def split_context(context: str) -> Dict:
    # 定义分隔符
    system_instruction_end = "Here are two examples:" if "Here are two examples:" in context else "Here are three examples:"
    raw_experience_end = "The following are some experience"
    condensed_experience_end = "Now it's your turn!"

    # 找到各个分隔符的位置
    try:
        system_instruction = context.split(system_instruction_end)[0].strip() + '\n'
        raw_experience = system_instruction_end + '\n' + context.split(system_instruction_end)[1].split(raw_experience_end)[0].strip() + '\n\n\n\n'
        condensed_experience = raw_experience_end + ' ' + context.split(raw_experience_end)[1].split(condensed_experience_end)[0].strip() + '\n\n'
        current_trajectory = condensed_experience_end + '\n' + context.split(condensed_experience_end)[1].strip()
    except IndexError:
        raise ValueError("The long string does not contain all required parts.")

    return {
        "System Instruction": system_instruction,
        "Raw Experience": raw_experience,
        "Condensed Experience": condensed_experience,
        "Current Trajectory": current_trajectory
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, default=0, help='selected index')
    parser.add_argument('--start_layer', type=int, default=0, help='start layer')
    parser.add_argument('--full_data_path', type=str, default="../data/input_data/Qwen3-1.7B/first_turn/expel_first.json",help='path to multi-hop file')
    parser.add_argument('--model_path', type=str, default="Qwen/Qwen3-1.7B", help='path to model')
    parser.add_argument("--adapter_path", type=str, default="", help='path to adapter')
    parser.add_argument('--dataset_path', type=str, default=None, help='path to `pg19-test` dataset')
    parser.add_argument("--loss-type",type=str, default = "label")
    parser.add_argument('--save_dir', type=str, default="../results", help='path to dataset')
    parser.add_argument("--tag", type = str, default = "expel")
    parser.add_argument("--only_ig", type = str, default = "False")
    args = parser.parse_args()

    turn = args.full_data_path.split('/')[-2]
    file = args.full_data_path.split('/')[-1].split('.')[0]
    args.save_dir = f"{args.save_dir}/{args.tag}/{turn}/{file}"

    print("Pid:",os.getpid())

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    full_data= auto_read_data(args.full_data_path)

    # for cnt, data in enumerate(full_data):
    data = full_data[args.task_id]
    input_context = data['context']
    context = split_context(input_context)

    task_id = data['task_id']
    answer = data['agent_output']
    model_name = args.model_path.split("/")[-1]
    save_file_name = f"{model_name}/layer_{args.start_layer}/{task_id}"
    if args.only_ig == 'True':
        save_file_name += '_ig'
    
    with_adapter= True if args.adapter_path else False
    start_layer = args.start_layer

    print("Loading Model.")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, attn_implementation = "flash_attention_2").half().to('cuda')
    if torch.cuda.device_count() > 1:
        if '8B' in args.model_path:
            device_map = {
                "model.embed_tokens": 0, "model.rotary_emb" :0, 
                "model.layers.0" :0, "model.layers.1" :0, "model.layers.2" :0,
                "model.layers.3" :1, "model.layers.4" :1, "model.layers.5" :1,
                "model.layers.6" :0, "model.layers.7" :0, "model.layers.8" :0,
                "model.layers.9" :1, "model.layers.10" :1, "model.layers.11" :1,
                "model.layers.12" :0, "model.layers.13" :0, "model.layers.14" :0,
                "model.layers.15" :1, "model.layers.16" :1, "model.layers.17" :1,
                "model.layers.18" :0, "model.layers.19" :0, "model.layers.20" :0,
                "model.layers.21" :1, "model.layers.22" :1, "model.layers.23" :1,
                "model.layers.24" :0, "model.layers.25" :1, "model.layers.26" :0,
                "model.layers.27" :1, "model.layers.28" :0, "model.layers.29" :1,
                "model.layers.30" :0, "model.layers.31" :1, "model.layers.32" :0,
                "model.layers.33" :1, "model.layers.34" :0, "model.layers.35" :1,
                "model.norm" :0, "lm_head" :1   
            }
        elif '14B' in args.model_path:
            device_map = {
                "model.embed_tokens": 0, "model.rotary_emb" :0, 
                "model.layers.0" :0, "model.layers.1" :0, "model.layers.2" :0,
                "model.layers.3" :1, "model.layers.4" :1, "model.layers.5" :1,
                "model.layers.6" :0, "model.layers.7" :0, "model.layers.8" :0,
                "model.layers.9" :1, "model.layers.10" :1, "model.layers.11" :1,
                "model.layers.12" :0, "model.layers.13" :0, "model.layers.14" :0,
                "model.layers.15" :1, "model.layers.16" :1, "model.layers.17" :1,
                "model.layers.18" :0, "model.layers.19" :0, "model.layers.20" :0,
                "model.layers.21" :1, "model.layers.22" :1, "model.layers.23" :1,
                "model.layers.24" :0, "model.layers.25" :1, "model.layers.26" :0,
                "model.layers.27" :1, "model.layers.28" :0, "model.layers.29" :1,
                "model.layers.30" :0, "model.layers.31" :1, "model.layers.32" :0,
                "model.layers.33" :1, "model.layers.34" :0, "model.layers.35" :1,
                "model.layers.36" :0, "model.layers.37" :1, "model.layers.38" :0,
                "model.layers.39" :1, "model.norm" :0, "lm_head" :1   
            }
        elif '32B' in args.model_path:
            device_map = {
                "model.embed_tokens": 0, "model.rotary_emb" :0, 
                "model.layers.0" :0, "model.layers.1" :0, "model.layers.2" :0,
                "model.layers.3" :1, "model.layers.4" :1, "model.layers.5" :1,
                "model.layers.6" :2, "model.layers.7" :2, "model.layers.8" :2,
                "model.layers.9" :3, "model.layers.10" :3, "model.layers.11" :3,
                "model.layers.12" :0, "model.layers.13" :0, "model.layers.14" :0,
                "model.layers.15" :1, "model.layers.16" :1, "model.layers.17" :1,
                "model.layers.18" :2, "model.layers.19" :2, "model.layers.20" :2,
                "model.layers.21" :3, "model.layers.22" :3, "model.layers.23" :3,
                "model.layers.24" :0, "model.layers.25" :1, "model.layers.26" :0,
                "model.layers.27" :1, "model.layers.28" :0, "model.layers.29" :1,
                "model.layers.30" :2, "model.layers.31" :2, "model.layers.32" :2,
                "model.layers.33" :3, "model.layers.34" :3, "model.layers.35" :3,
                "model.layers.36" :0, "model.layers.37" :1, "model.layers.38" :0,
                "model.layers.39" :1, "model.layers.39" :0, "model.layers.40" :1,
                "model.layers.41" :2, "model.layers.42" :2, "model.layers.43" :2,
                "model.layers.44" :3, "model.layers.45" :3, "model.layers.46" :3,
                "model.layers.47" :0, "model.layers.48" :0, "model.layers.49" :0, 
                "model.layers.50" :1, "model.layers.51" :1, "model.layers.52" :1, 
                "model.layers.53" :2, "model.layers.54" :2, "model.layers.55" :2, 
                "model.layers.56" :3, "model.layers.57" :3, "model.layers.58" :3, 
                "model.layers.59" :0, "model.layers.60" :1, "model.layers.61" :2, 
                "model.layers.62" :3, "model.layers.63" :0,
                "model.norm" :0, "lm_head" :1   
            }
        model = dispatch_model(model, device_map=device_map)
    print("Process inputs.")

    start = 3
    search_inps = tokenizer(context['System Instruction'], return_offsets_mapping=True, return_tensors='pt').input_ids.to(model.device)
    search_pos = [(start, start + search_inps.size(-1)-1)]
    start += search_inps.size(-1)
    inp = search_inps

    attack_inps = tokenizer(context['Raw Experience'], return_offsets_mapping=True, return_tensors='pt').input_ids.to(model.device)
    attack_pos = [(start, start + attack_inps.size(-1)-1)]
    start += attack_inps.size(-1)
    inp = torch.cat((inp, attack_inps), dim=1)

    other_inps = tokenizer(context['Condensed Experience'], return_offsets_mapping=True, return_tensors='pt').input_ids.to(model.device)
    other_pos = [(start, start + other_inps.size(-1)-1)]
    start += other_inps.size(-1)
    inp = torch.cat((inp, other_inps), dim=1)

    # temp = attack_pos
    # attack_pos = other_pos
    # other_pos = temp

    emoji_inps = tokenizer(context['Current Trajectory'] + '\n', return_offsets_mapping=True, return_tensors='pt').input_ids.to(model.device)
    emoji_spans = [(start, start + emoji_inps.size(-1)-1)]
    start += emoji_inps.size(-1)
    inp = torch.cat((inp, emoji_inps), dim=1)

    messages = [
        {"role": "user", "content": ''}
    ]
    template_ids = tokenizer.apply_chat_template(
        conversation=messages, 
        tokenize=True, 
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors='pt'
    ).to(model.device)
    inp = torch.cat((template_ids[:, :3], inp, template_ids[:, 3:]), dim=1)

    context_length = len(inp)
    is_0k = (context_length == 0)

    # answer_ids = tokenizer(answer, add_special_tokens=False, return_tensors='pt')["input_ids"].to(model.device)
    # inp_with_answer = torch.cat((inp, answer_ids[:, 0:1]), dim=1)

    # toks_length = answer_ids.size(-1)
    # for j in range(inp_with_answer.size(-1), toks_length, -1):
    #     if (inp_with_answer[0, j-toks_length : j] == answer_ids).sum().item() == toks_length:
    #         target_pos = (j-toks_length, j) 
    #         break
    # else:
    #     raise ValueError("Not find target in input tokens!")
    
    print("Testing scores.")
    if args.loss_type == "label":
        # label = torch.full(inp_with_answer.shape, -100).to(model.device)
        # for sub_pos in range(*target_pos):
        #     label[0, sub_pos] = inp_with_answer[0, sub_pos]
        if args.only_ig == 'True':
            print("Testing only IG.")
            flow_res = test_model_ig_update(model, inp, search_pos, attack_pos, other_pos,
                                                    emoji_spans,
                                                    is_0k,
                                                    model_name, tokenizer)
        else:
            flow_res = test_model_with_adapter_update(model, inp, search_pos, attack_pos, other_pos,
                                                    emoji_spans,
                                                    is_0k,
                                                    model_name, tokenizer, with_adapter=with_adapter,
                                                    start_layer=start_layer)
    
    elif args.loss_type == "ce":
        if args.only_ig == 'True':
            flow_res = test_model_ig_update(model, inp, search_pos, attack_pos, other_pos,
                                                    emoji_spans,
                                                    is_0k,
                                                    model_name, tokenizer)
        else:
            flow_res = test_model_with_adapter_update(model, inp, search_pos, attack_pos, other_pos,
                                                    emoji_spans,
                                                    is_0k,
                                                    model_name, tokenizer, with_adapter=with_adapter,
                                                    start_layer = start_layer)

    flow_res["pred_res"] = answer
    flow_res["score"] = 100 if data['is_success'] else 0

    logger.info(flow_res)
    auto_save_data(flow_res, f"{args.save_dir}/{save_file_name}.pkl")

    logger.info("Now deleting cache.")
    del model
    torch.cuda.empty_cache()
    time.sleep(5)

    print("TESTING OVER!")