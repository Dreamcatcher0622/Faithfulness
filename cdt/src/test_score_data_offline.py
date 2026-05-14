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

def begin_test(args, input_context, question, answer, selected_idx, model, tokenizer, depth_percent, evidence, evidence_tok_needles,disturb_tok_needles, emoji_text_spans, save_file_name, model_name, is_0k, use_emoji, with_adapter=False, start_layer = 0):


    inps = tokenizer(input_context, return_offsets_mapping=True, return_tensors = 'pt')

    inp = inps.input_ids.to(model.device)
    search_pos = find_multi_needle_idx(inp[0], tokenizer, evidence_tok_needles)
    attack_pos = find_multi_needle_idx(inp[0], tokenizer, disturb_tok_needles)

    offset_mapping = inps.offset_mapping[0].tolist()
    emoji_spans = []

    for emj_txt_span in emoji_text_spans:
        emj_l,emj_r  = emj_txt_span
        overlap = 0
        # sec = None
        L ,R = None, None
        for ix, x in enumerate(offset_mapping):
            l,r = x
            if l<=  emj_l <= r:
                L = ix
            if l<= emj_r <= r or ( L is not None and l>emj_r):
                R = ix

            if (L is not None ) and (R is not None):
                break

        emoji_spans += [(L,R)]


    print("emoji:")
    for emoji_span in emoji_spans:
        print("Detected:",tokenizer.decode(inp[0,emoji_span[0]:emoji_span[1]].tolist()),inp[0,emoji_span[0]:emoji_span[1]].tolist())
        print()
    
    with torch.no_grad():
        pred_res = tokenizer.decode(model.generate(inp, max_new_tokens=32, do_sample=False)[0, inp.size(-1):])
        logger.info(pred_res)

    logger.info(inp.shape)

    inp = tokenizer(input_context + "\n" + answer, return_tensors='pt').input_ids.to(model.device)
    

    answer_ids = tokenizer(answer, add_special_tokens=False, return_tensors='pt')["input_ids"].to(model.device)
    toks_length = answer_ids.size(-1)
    for j in range(inp.size(-1), toks_length, -1):
        if (inp[0, j-toks_length : j] == answer_ids).sum().item() == toks_length:
            target_pos = (j-toks_length, j) 
            break
    else:
        raise ValueError("Not find target in input tokens!")
    
    if args.loss_type == "label":
        label = torch.full(inp.shape, -100).to(model.device)
        for sub_pos in range(*target_pos):
            label[0, sub_pos] = inp[0, sub_pos]

        flow_res = test_model_with_adapter(model, inp, label, search_pos, attack_pos, 
                                                     emoji_spans,
                                                     (target_pos[0] - 1,
                                                      target_pos[1] - 1), 
                                                      is_0k,
                                                      model_name, tokenizer, with_adapter=with_adapter,
                                                      start_layer = start_layer)
    
    elif args.loss_type == "ce":
        flow_res = test_model_with_adapter(model, inp, inp, search_pos, attack_pos, 
                                                     emoji_spans,
                                                     (target_pos[0] - 1,
                                                      target_pos[1] - 1), 
                                                      is_0k,
                                                      model_name, tokenizer, with_adapter=with_adapter,
                                                      start_layer = start_layer)

    flow_res["pred_res"] = pred_res
    flow_res["score"] = 100 if answer.lower() in pred_res.lower() else 0

    logger.info(flow_res)
    auto_save_data(flow_res, f"{args.save_dir}/{save_file_name}.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--selected_idx', type=int, default=0, help='selected index')
    parser.add_argument('--full_data_path', type=str, default="../data/full20.jsonl",help='path to multi-hop file')
    parser.add_argument('--model_path', type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help='path to model')
    parser.add_argument("--adapter_path", type=str, default="", help='path to adapter')
    parser.add_argument('--dataset_path', type=str, default=None, help='path to `pg19-test` dataset')
    parser.add_argument("--loss-type",type=str, default = "label")
    parser.add_argument("--use_emoji", type = bool, default = True)
    parser.add_argument('--save_dir', type=str, default="../results", help='path to dataset')
    parser.add_argument("--tag", type = str, default = "information_flow_data_offline")
    args = parser.parse_args()

    
    args.save_dir = f"{args.save_dir}/{args.tag}"

    print("Pid:",os.getpid())


    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    full_data= auto_read_data(args.full_data_path)

    needles_and_stacks = [k['source_needles'] for k in full_data]


    for cnt, data in enumerate(full_data):
        context_length = int(data['context_length'][:-1])*1000 - 100
        needle_and_stack = data['source_needles']

        evidence_tok_needles = needle_and_stack['real_needle']
        disturb_tok_needles = [k for k in needle_and_stack['needle'] if k not in evidence_tok_needles]
        
        needle = [tokenizer(i, add_special_tokens=False)['input_ids'] for i in needle_and_stack['needle']]
        evidence = [tokenizer(i, add_special_tokens=False)['input_ids'] for i in needle_and_stack['real_needle']]
        question = needle_and_stack['question']
        answer = needle_and_stack['golden_answer']
        tag = needle_and_stack['tag']
        s_id = data['s_id']
        depth_percent = data['depth_percent']

        input_context = data['input_context']
        emoji_text_spans = data['emoji_text_spans']


        model = AutoModelForCausalLM.from_pretrained(args.model_path,
        attn_implementation = "flash_attention_2").half().to('cuda')

        # device_map = {
        #     "model.embed_tokens": 0, "model.rotary_emb" :0, 
        #     "model.layers.0" :0, "model.layers.1" :0, "model.layers.2" :0,
        #     "model.layers.3" :1, "model.layers.4" :1, "model.layers.5" :1,
        #     "model.layers.6" :2, "model.layers.7" :2, "model.layers.8" :2,
        #     "model.layers.9" :3, "model.layers.10" :3, "model.layers.11" :3,
        #     "model.layers.12" :4, "model.layers.13" :4, "model.layers.14" :4,
        #     "model.layers.15" :5, "model.layers.16" :5, "model.layers.17" :5,
        #     "model.layers.18" :6, "model.layers.19" :6, "model.layers.20" :6,
        #     "model.layers.21" :7, "model.layers.22" :7, "model.layers.23" :7,
        #     "model.layers.24" :0, "model.layers.25" :1, "model.layers.26" :2,
        #     "model.layers.27" :3, "model.layers.28" :4, "model.layers.29" :5,
        #     "model.layers.30" :6, "model.layers.31" :7, "model.norm" :6,
        #     "lm_head"  : 7
            
        # }

        # model = dispatch_model(model, device_map=device_map)

        depth_tag = "-".join([str(i) for i in depth_percent])
        model_name = args.model_path.split("/")[-1]
        
        save_file_name = f"{model_name}/{context_length}/{args.loss_type}/{tag}_sid-{s_id}_pid-{cnt}_{depth_tag}"
        
        begin_test(args, input_context, question, answer, s_id, model, tokenizer, depth_percent, evidence, evidence_tok_needles, disturb_tok_needles, emoji_text_spans, save_file_name, model_name, is_0k = (context_length == 0), use_emoji = args.use_emoji, with_adapter= True if args.adapter_path else False,                                   start_layer = 24)
        
        del model
        torch.cuda.empty_cache()

    print("TESTING OVER!")