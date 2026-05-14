
import os
import json
import random
import re

def _modify_prompt(sys_msg: str, fewshot_modification_type: str = 'None', insights_modification_type: str = 'None') -> str:
    if insights_modification_type != 'None':
        split_lines = sys_msg.splitlines(keepends=True)
        for line in split_lines:
            if '## Title' in line or '## Description' in line or '## Content' in line:
                if '## Title' in line:
                    index = line.find('## Title')
                    insights = line[index+9:]
                if '## Description' in line:
                    index = line.find('## Description')
                    insights = line[index+15:]
                if '## Content' in line:
                    index = line.find('## Content')
                    insights = line[index+11:]

                replace_insights = '\n'
                if insights:
                    if insights_modification_type == 'empty':
                        replace_insights = '\n'
                    elif insights_modification_type == 'corrupt':
                        replace_insights = _corrupt_insights(insights)
                    elif insights_modification_type == 'irrelevant':
                        replace_insights = _replace_insights_with_irrelevant(insights)
                    elif insights_modification_type == 'filler_tokens':
                        replace_insights = _replace_insights_with_filler_tokens(insights)

                    sys_msg = sys_msg.replace(insights, replace_insights)
    


    return sys_msg
        
def _corrupt_insights(prompt: str) -> str:
    """
    随机毁坏给定的字符串内容
    
    Args:
        prompt: 原始字符串
            
    Returns:
        损坏后的字符串
    """
    # 将输入字符串分割为句子
    insights = re.split(r'(?<=[,.])\s*', prompt)
    # print(f"Here is insights:{insights}")
    replaced_insights = []
    
    for insight in insights:
        if len(insight) > 10:
            insight = insight.strip()
            dot = insight[-1]
            insight = insight[:-1]
            words = insight.split()
            if len(words) > 2:
                # 策略1: 随机替换一些词
                for i in range(min(3, len(words) // 3)):  # 最多替换1/3的词
                    if random.random() < 0.5:
                        word_idx = random.randint(0, len(words) - 1)
                        words[word_idx] = f"[CORRUPTED_{random.randint(1, 999)}]"
                
                # 策略2: 随机插入错误信息
                if random.random() < 0.8:
                    words.insert(random.randint(0, len(words)), "[ERROR_INFO]")
                
                # 策略3: 随机删除一些词
                if len(words) > 5 and random.random() < 0.8:
                    num_remove = min(2, len(words) // 4)
                    for _ in range(num_remove):
                        if words:
                            words.pop(random.randint(0, len(words) - 1))
            insight = ' '.join(words) + dot
            replaced_insights.append(insight)
    
    # 返回损坏后的字符串
    corrupted_prompt = ' '.join(replaced_insights)  + '\n'
    return corrupted_prompt

def _replace_insights_with_irrelevant(prompt: str) -> str:
    """
    替换给定字符串中的内容为不相关的部分
    
    Args:
        prompt: 原始字符串
            
    Returns:
        替换后的字符串
    """
    # 定义一些不相关的内容模板
    irrelevant_templates = [
        "The weather is sunny today.",
        "Mathematics involves complex calculations.",
        "Cooking requires proper temperature control.",
        "Music theory includes scales and chords.",
        "Sports require physical fitness and training.",
        "Literature contains various genres and styles.",
        "Technology continues to advance rapidly.",
        "History provides lessons from the past.",
        "Science explores the natural world.",
        "Art expresses human creativity and emotion.",
        "Travel broadens one's perspective.",
        "Education is fundamental for personal growth.",
        "Health requires balanced nutrition and exercise.",
        "Communication is essential for relationships.",
        "Time management improves productivity."
    ]
    
    # 将输入字符串分割为句子或段落
    insights = re.split(r'(?<=[,.])\s*', prompt)
    replaced_insights = []
    
    for insight in insights:
        if len(insight) > 10:
            insight = insight.strip()
            if random.random() < 0.8:  # 80% 概率替换
                # 随机选择一个不相关的模板
                insight = random.choice(irrelevant_templates)
                replaced_insights.append(insight)
            else:
                replaced_insights.append(insight)

    # 将替换后的部分组合回一个字符串
    replaced_prompt = ' '.join(replaced_insights)  + '\n'
    return replaced_prompt

def _replace_insights_with_filler_tokens(prompt: str) -> str:
    """
    替换给定字符串中的内容为填充字符（如...、$等无意义字符）
    
    Args:
        prompt: 原始字符串
            
    Returns:
        替换后的字符串
    """
    # 定义填充字符模板
    filler_tokens = [
        "###...###",
        "$$$...$$$",
        "***...***",
        "---...---",
        "+++...+++",
        "~~~...~~~",
        "==...==",
        "//...//",
        "\\\\...\\\\",
        "### $$$ ***",
        "--- +++ ~~~",
        "== // \\\\",
        "... $$$ ###",
        "*** --- +++",
        "~~~ == //"
    ]
    
    # 将输入字符串分割为句子或段落
    insights = re.split(r'(?<=[,.])\s*', prompt)
    filled_insights = []

    for insight in insights:
        if len(insight) > 10:
            insight = insight.strip()
            dot = insight[-1]
            insight = insight[:-1]

            if random.random() < 0.8:  # 80% 概率替换
                # 随机选择一个填充字符模板
                insight = random.choice(filler_tokens)
                insight = insight + dot
                filled_insights.append(insight)
            else:
                insight = insight + dot
                filled_insights.append(insight)

    # 将替换后的部分组合回一个字符串
    filled_prompt = ' '.join(filled_insights)  + '\n'
    return filled_prompt



if __name__ == "__main__":
    dir_path = "./input_prompt"
    for mode in ["with", "wo", "empty", "corrupt", "irrelevant", "filler_tokens"]:
        output = []
        for filename in os.listdir(dir_path):
            if filename.endswith(".json"):
                file_path = os.path.join(dir_path, filename)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                if len(data) > 1:
                    temp = {}
                    temp["system_message"] = data[-1]["system_message"]
                    
                    if mode == "with":
                        memory_item = data[-1]['memory_item']
                    elif mode == "wo":
                        memory_item = "\n\n"
                    else:
                        memory_item = _modify_prompt(data[-1]['memory_item'], fewshot_modification_type='None', insights_modification_type=mode)
                    
                    temp["memory_item"] = memory_item
                    idx1 = data[-1]["prompt"].find("# Observation of current step")
                    idx2 = data[-1]["prompt"].find("# History of interaction with the task")
                    idx3 = data[-1]["prompt"].find("# Action space")
                    temp["sys1"] = data[-1]["prompt"][:idx1]
                    temp["env"] = data[-1]["prompt"][idx1:idx2]
                    temp["history"] = data[-1]["prompt"][idx2:idx3]
                    temp["sys2"] = data[-1]["prompt"][idx3:]
                    output.append(temp)


        output_path = os.path.join("inputs", f"input_prompt_{mode}.json")
        print(len(output))
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=4)
