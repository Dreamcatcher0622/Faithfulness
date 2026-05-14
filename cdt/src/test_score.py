import sys, os, argparse
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import AutoTokenizer, AutoModelForCausalLM
from loguru import logger
import itertools
from peft import peft_model, PeftModelForCausalLM
import numpy as np
import datasets
logger.info(sys.path)
from retrieval_head_detection import SentenceSampler
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
from utils import *

def begin_test(args, question, answer, selected_idx, model, tokenizer, depth_percent, background_text, disturb_pos, disturb_tok_needles, evidence, evidence_list, save_file_name, model_name, use_emoji, is_0k, with_adapter=False, start_layer = 0):
    """
        question: 输入的问题 str
        answer: 问题的答案 str
        selected_inx: 当前数据的index int
        model: GPU分布后的模型
        tokenizer: 分词器
        depth_percent: len(evidence) 长度的 [0-9] 随机组合 List[int]
        background_text: tokenize后的无关句子列表，总token数为args.context_lengths List[List[int]]
        disturb_tok_needles: tokenize后的干扰句子列表 List[List[int]]
        disturb_pos: np.random.choice(len(background_text) + 1, len(disturb_tok_needles))
        evidence:tokenize后的相关句子列表 List[List[int]]
        evidence_list: 未tonkize的相关句子列表 List[List[str]]
        save_file_name: 保存文件名 str
        model_name: 模型名 str
        use_emoji: bool
        is_0k: 当前的context_lengths == 0 bool
        with_adapter: bool
        start_layer: int
    """
    if background_text is not None:
        if use_emoji:
            # emoji 相当于low-frequency words低频字
            emojis10 = get_random_emoji(tokenizer, 10, return_idx=True, seed=42)    # 获取随机emoji的index list
            background_text, emoji_pos = random_combine(background_text, emojis10,  # 将emoji插入到无关句子列表中
                                                        return_snd_pos=True, seed=42)
            emoji_pos = set(emoji_pos)
            cumsum_num = 0
            emoji_spans = []

        depth_percent = [i / 10 for i in depth_percent]
        updated_sample = [[] for _ in range(len(background_text) + 1)]
        real_pos = [int(len(background_text) * i) for i in depth_percent]
        for fact, pos in zip(evidence, real_pos):  # insert real needle len(evidence)==len(real_pos)
            updated_sample[pos].append(fact)    # 相当于在随机pos插入相关句子
        for fact, pos in zip(disturb_tok_needles, disturb_pos):  # insert disturb needle
            updated_sample[pos].append(fact)    # 相当于在随机pos插入干扰句子
        # [[real1, disturb1], [real2], [], [disturb2], ...] 共len(background_text)+1长度

        for i, s in enumerate(background_text):  # insert irrevelent needle
            if use_emoji and (i in emoji_pos):
                cur_pos = sum((len(l) for l in updated_sample[i]), 0)   # updated_sample当前位置所有token的长度
                emoji_spans += [(cumsum_num + cur_pos, cumsum_num + cur_pos + len(s))] # ？？？
            updated_sample[i].append(s) # 把无关样本插入

            if use_emoji:
                cumsum_num += sum((len(l) for l in updated_sample[i]), 0)
    else:
        updated_sample = random_combine(evidence[:-1], disturb_tok_needles + [evidence[-1]], seed=42)
        updated_sample = [[k] for k in updated_sample]
    
    if not use_emoji or is_0k:
        emoji_spans = []

    # 把所有token全部展平 List[List[List[int]]] -> List[int]
    flat = [i for s in updated_sample for i in s]
    tokens = [i for s in flat for i in s]

    # 终于把输入的数据处理完了！转换为字符串
    new_context = tokenizer.decode(tokens)
    input_context = new_context + f"\n{question}\nAnswer:"
    if tokenizer.chat_template is not None:
        shift = 30
        # tokenize 为 torch.tensor
        inp = tokenizer.apply_chat_template([{ "role": "user", "content": input_context}], tokenize=True, add_generation_prompt=True, return_tensors='pt')
    else:
        shift = 0
        inp = tokenizer(input_context, return_tensors='pt').input_ids
    emoji_spans = [(k[0] + shift, k[1] + shift) for k in emoji_spans]
    
    if use_emoji:
        print("emoji:")
        for emoji_span, emj in zip(emoji_spans, emojis10):
            print("Original:",tokenizer.decode(emj),emj)
            print("Detected:",tokenizer.decode(inp[0,emoji_span[0]:emoji_span[1]].tolist()),inp[0,emoji_span[0]:emoji_span[1]].tolist())
            print()

    search_pos = find_multi_needle_idx(inp[0], tokenizer, evidence_list[selected_idx])  # 查找当前evidence在inp（输入token里）的位置
    attack_pos = find_multi_needle_idx(inp[0], tokenizer, disturb_tok_needles)          # 查找当前无关句子在inp（输入token里）的位置
    inp = inp.to(model.device)      # 输入model的tokenize后
    
    with torch.no_grad():
        # 想模型问问题，获取模型输出pred_res
        pred_res = tokenizer.decode(model.generate(inp, max_new_tokens=32, do_sample=False)[0, inp.size(-1):])
        logger.info(pred_res)

    logger.info(inp.shape)

    if tokenizer.chat_template is not None:
        # 问题和答案同时tokenize
        inp = tokenizer.apply_chat_template(
            [{"role": "user", "content": input_context}, {"role": "assistant", "content": answer}], 
            tokenize=True, add_generation_prompt=False, return_tensors='pt'
        ).to(model.device)
    else:
        inp = tokenizer(input_context + "\n" + answer, return_tensors='pt').input_ids.to(model.device)
    
    # 在输入的材料中获取答案所在的位置target_pos
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
        # 获取IG分数
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
    parser.add_argument('--needle_path', type=str, default="preliminary/data/reasoning_needle.jsonl",help='path to multi-hop file')
    parser.add_argument('--model_path', type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help='path to model')
    parser.add_argument("--adapter_path", type=str, default="", help='path to adapter')
    parser.add_argument('--dataset_path', type=str, default="preliminary/data/pg19-test", help='path to `pg19-test` dataset')
    parser.add_argument('--save_dir', type=str, default="preliminary/results", help='path to dataset')
    parser.add_argument("--tag", type = str, default = "information_flow")
    parser.add_argument("--select-range",type = str, default = "0,200", help="Selected range of samples")
    parser.add_argument("--use-emoji", type = bool, default = True)
    parser.add_argument("--context_lengths", type = str, default = "11900,7900,3900,1900,900", help = 'contexts of lengths that will be tested')

    args = parser.parse_args()
    args.save_dir = f"{args.save_dir}/{args.tag}"

    print("Pid:",os.getpid())

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)      # 获取tokenizer

    needles_and_stacks = auto_read_data(args.needle_path)           # 读取文件

    l,r = tuple(map(int,args.select_range.split(",")))
    step = 1
    selected_idx = list(range(l, r, step))

    # 获取 needle question real_needle golden_answer tag 的 list
    needle_list = [l["needle"] for l in needles_and_stacks]
    retrieval_question_list = [l["question"] for l in needles_and_stacks]
    evidence_list = [l["real_needle"] for l in needles_and_stacks]
    golden_answer_list = [l["golden_answer"] for l in needles_and_stacks]
    tags = [l["tag"] for l in needles_and_stacks]

    # 搞不懂这是干嘛
    for pe,pn in zip(evidence_list, needle_list):
        last_idx = pn.index(pe[-1])
        assert last_idx > -1
        pe += [pn[last_idx + 1]]
    
    random.seed(42)
    for context_length in list(map(int, args.context_lengths.split(","))):  # 获得长度 11190 7900 ...
        for loss_type in [ "label" ]:
            args.context_length = context_length
            args.loss_type = loss_type 
            for s_id in selected_idx:       # 手动选择的 index，对应的question answer tag needle real_needle
                logger.info(f"Selected idx: {s_id}")
                logger.info(f"Question: {retrieval_question_list[s_id]}")
                logger.info(f"Answer: {golden_answer_list[s_id]}")
                logger.info(f"Tag: {tags[s_id]}")
                logger.info(f"Needle: {needle_list[s_id]}")
                logger.info(f"Real Needle: {evidence_list[s_id]}")
                logger.info("=============================================")

                # 获取当前 needle evidence 各句在词汇表中tokenize后的
                needle = [tokenizer(i, add_special_tokens=False)['input_ids'] for i in needle_list[s_id]]
                evidence = [tokenizer(i, add_special_tokens=False)['input_ids'] for i in evidence_list[s_id]]
                question = retrieval_question_list[s_id]
                answer = golden_answer_list[s_id]
                tag = tags[s_id]

                # 初始化采样器
                haystack = datasets.load_dataset(args.dataset_path, split="test")   # pg19-test:书名-发表日期-url-描述
                if args.context_length > 0:
                    noise_sampler_test = SentenceSampler(haystack, tokenizer=tokenizer, shuffle=False, random_seed=42)  # 在数据集中抽样
                    # 这是irrelevant documents，不相关样本
                    background_text = noise_sampler_test.get_sample(args.context_length)  # 在数据集的描述'text'中获取句子，tokenize后足够11900，List[List[int]]
                    # 这是interference facts，干扰样本
                    disturb_tok_needles = [i for i in needle if i not in evidence]
                    np.random.seed(42)
                    disturb_pos = np.random.choice(len(background_text) + 1, len(disturb_tok_needles))  # 在不相关样本长度中随机选择和干扰样本一样多的index
                    print("disturb:", disturb_pos)
                else:
                    background_text = None
                    disturb_tok_needles = [i for i in needle if i not in evidence]
                    disturb_pos = None

                combinations_number = 100
                # 在[0-9]元素中获取所有长度为evidence长度的组合，如(0, 1, 2) (0, 1, 3)等等所有
                all_combinations = list(itertools.combinations(list(range(10)), len(evidence)))
                # 从所有组合中随机选100个
                all_combinations = random.sample(all_combinations, combinations_number)
                cnt = 0
                with tqdm(total=len(all_combinations)) as pbar:
                    for _, depth_percent in enumerate(all_combinations):
                        if cnt == 3: break      # 成功三次就停止
                        try:
                            # 获取半精度模型
                            model = AutoModelForCausalLM.from_pretrained(args.model_path, 
                                                    attn_implementation = "flash_attention_2").half()
                            # ！！！GPU分配表，把模型不同层分配到不同GPU上
                            device_map = {
                                "model.embed_tokens": 0, "model.rotary_emb" :0, 
                                "model.layers.0" :0, "model.layers.1" :0, "model.layers.2" :0,
                                "model.layers.3" :1, "model.layers.4" :1, "model.layers.5" :1,
                                "model.layers.6" :2, "model.layers.7" :2, "model.layers.8" :2,
                                "model.layers.9" :3, "model.layers.10" :3, "model.layers.11" :3,
                                "model.layers.12" :4, "model.layers.13" :4, "model.layers.14" :4,
                                "model.layers.15" :5, "model.layers.16" :5, "model.layers.17" :5,
                                "model.layers.18" :6, "model.layers.19" :6, "model.layers.20" :6,
                                "model.layers.21" :7, "model.layers.22" :7, "model.layers.23" :7,
                                "model.layers.24" :0, "model.layers.25" :1, "model.layers.26" :2,
                                "model.layers.27" :3, "model.layers.28" :4, "model.layers.29" :5,
                                "model.layers.30" :6, "model.layers.31" :7, "model.norm" :6,
                                "lm_head"  : 7
                            }
                            model = dispatch_model(model, device_map=device_map)

                            pbar.set_description(f"Processing depth {depth_percent}")   # log
                            depth_tag = "-".join([str(i) for i in depth_percent])
                            model_name = args.model_path.split("/")[-1]
                            # 结果保存路径
                            save_file_name = f"{model_name}/{args.context_length}/{args.loss_type}/{tag}_sid-{s_id}_pid-{cnt}_{depth_tag}"
                            
                            begin_test(args, question, answer, s_id, model, tokenizer, depth_percent, background_text, disturb_pos, disturb_tok_needles, evidence, evidence_list, save_file_name, model_name, args.use_emoji, is_0k = (context_length == 0), with_adapter= True if args.adapter_path else False,                                   start_layer = 24)
                            pbar.update(1)
                            cnt += 1
                            print("dep_p:", depth_percent)

                        except ZeroDivisionError as ze:
                            continue
                        except ValueError as e:
                            if str(e) == "evidence_list and disturb_tok_needles length not match!":
                                continue
                        finally:
                            del model
                            torch.cuda.empty_cache()

                    if cnt != 3:
                        print(f"args.context_length: {args.context_length}")
                        print(f"args.loss_type: {args.loss_type}")
                        print(f"cnt: {cnt}")
                        print(f"s_id: {s_id}")

            file_dir =f"{args.save_dir}/{model_name}/{args.context_length}/{args.loss_type}/"

        print("TESTING OVER:", context_length, loss_type)