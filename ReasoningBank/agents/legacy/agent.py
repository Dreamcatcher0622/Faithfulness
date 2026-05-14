"""
WARNING DEPRECATED WILL BE REMOVED SOON
"""

from dataclasses import asdict, dataclass, field
import traceback
from warnings import warn
from langchain.schema import HumanMessage, SystemMessage

from browsergym.core.action.base import AbstractActionSet
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html
from browsergym.experiments import Agent, AbstractAgentArgs

from ..legacy import dynamic_prompting
from .utils.llm_utils import ParseError, retry
from .utils.chat_api import ChatModelArgs

import random
import re
import json
import os

def save_logs(log:str, log_path:str):
    if log:
        with open(log_path, 'a') as f:
            f.write(log)

@dataclass
class GenericAgentArgs(AbstractAgentArgs):
    chat_model_args: ChatModelArgs = None
    flags: dynamic_prompting.Flags = field(default_factory=lambda: dynamic_prompting.Flags())
    max_retry: int = 4
    faithfulness_experiment: bool = False
    fewshot_modification_type: str = 'None'
    insights_modification_type: str = 'None'
    log_path: str = 'logs/log.txt'
    tid: int = 0

    def make_agent(self):
        return GenericAgent(
            chat_model_args=self.chat_model_args, flags=self.flags, max_retry=self.max_retry,
            faithfulness_experiment=self.faithfulness_experiment,
            fewshot_modification_type=self.fewshot_modification_type,
            insights_modification_type=self.insights_modification_type,
            log_path = self.log_path,
            tid = self.tid
        )


class GenericAgent(Agent):

    def obs_preprocessor(self, obs: dict) -> dict:
        """
        Augment observations with text HTML and AXTree representations, which will be stored in
        the experiment traces.
        """

        obs = obs.copy()
        obs["dom_txt"] = flatten_dom_to_str(
            obs["dom_object"],
            with_visible=self.flags.extract_visible_tag,
            with_center_coords=self.flags.extract_coords == "center",
            with_bounding_box_coords=self.flags.extract_coords == "box",
            filter_visible_only=self.flags.extract_visible_elements_only,
        )
        obs["axtree_txt"] = flatten_axtree_to_str(
            obs["axtree_object"],
            with_visible=self.flags.extract_visible_tag,
            with_center_coords=self.flags.extract_coords == "center",
            with_bounding_box_coords=self.flags.extract_coords == "box",
            filter_visible_only=self.flags.extract_visible_elements_only,
        )
        obs["pruned_html"] = prune_html(obs["dom_txt"])

        return obs

    def __init__(
        self,
        chat_model_args: ChatModelArgs = None,
        flags: dynamic_prompting.Flags = None,
        max_retry: int = 4,
        faithfulness_experiment: bool = False,
        fewshot_modification_type: str = 'None',
        insights_modification_type: str = 'None',
        log_path: str = 'logs/log.txt',
        tid: int = 0,
    ):
        self.chat_model_args = chat_model_args if chat_model_args is not None else ChatModelArgs()
        self.flags = flags if flags is not None else dynamic_prompting.Flags()
        self.max_retry = max_retry

        self.chat_llm = chat_model_args.make_chat_model()
        self.action_set = dynamic_prompting._get_action_space(self.flags)

        self.faithfulness_experiment = faithfulness_experiment
        self.fewshot_modification_type = fewshot_modification_type
        self.insights_modification_type = insights_modification_type
        self.log_path = log_path
        self.tid = tid

        # consistency check
        if self.flags.use_screenshot:
            if not self.chat_model_args.has_vision():
                warn(
                    """\

Warning: use_screenshot is set to True, but the chat model \
does not support vision. Disabling use_screenshot."""
                )
                self.flags.use_screenshot = False

        # reset episode memory
        self.obs_history = []
        self.actions = []
        self.memories = []
        self.thoughts = []

    def get_action(self, obs):

        self.obs_history.append(obs)

        main_prompt = dynamic_prompting.MainPrompt(
            obs_history=self.obs_history,
            actions=self.actions,
            memories=self.memories,
            thoughts=self.thoughts,
            flags=self.flags,
        )

        # Determine the minimum non-None token limit from prompt, total, and input tokens, or set to None if all are None.
        maxes = (
            self.flags.max_prompt_tokens,
            self.chat_model_args.max_total_tokens,
            self.chat_model_args.max_input_tokens,
        )
        maxes = [m for m in maxes if m is not None]
        max_prompt_tokens = min(maxes) if maxes else None

        prompt = dynamic_prompting.fit_tokens(
            main_prompt,
            max_prompt_tokens=max_prompt_tokens,
            model_name=self.chat_model_args.model_name,
        )

        input_prompt = {}

        sys_msg = dynamic_prompting.SystemPrompt().prompt

        input_prompt["system_message"] = sys_msg
        input_prompt["prompt"] = prompt

        if self.flags.memory_item is not None:
            sys_msg += '\n\n' + self.flags.memory_item
            input_prompt["memory_item"] = self.flags.memory_item


        # 将input_prompt以json形式追加到文件中，如果不存在，先创建json文件
        path = f"./input_prompt/{self.tid}.json"
        if os.path.exists(path):
            with open(path, 'r') as f:
                existing_data = json.load(f)
            existing_data.append(input_prompt)
            print(f"Path exists. Now length: {len(existing_data)}")
            json.dump(existing_data, open(path, 'w'), indent=4)
        else:
            with open(path, 'w') as f:
                json.dump([input_prompt], f, indent=4)


        # if self.flags.workflow_path is not None:
        #     sys_msg += '\n\n' + open(self.flags.workflow_path).read()

        # print('-' * 50)
        # print(f"Here is sys_msg:\n{sys_msg}\n")
        # print('-' * 50)

        # faithfullness experiments --------------------------
        if self.faithfulness_experiment:
            sys_msg = self._modify_prompt(sys_msg, self.fewshot_modification_type, self.insights_modification_type)
        save_logs('\n######################################################################\n', self.log_path)
        save_logs(sys_msg, self.log_path)
        # ----------------------------------------------------

        chat_messages = [
            SystemMessage(content=sys_msg),
            HumanMessage(content=prompt),
        ]

        def parser(text):
            try:
                ans_dict = main_prompt._parse_answer(text)
            except ParseError as e:
                # these parse errors will be caught by the retry function and
                # the chat_llm will have a chance to recover
                return None, False, str(e)

            return ans_dict, True, ""
        
        for i in range(3):
            try:
                ans_dict = retry(self.chat_llm, chat_messages, n_retry=self.max_retry, parser=parser)
                # inferring the number of retries, TODO: make this less hacky
                ans_dict["n_retry"] = (len(chat_messages) - 3) / 2
                break
            except ValueError as e:
                # Likely due to maximum retry. We catch it here to be able to return
                # the list of messages for further analysis
                ans_dict = {"action": None}
                ans_dict["err_msg"] = str(e)
                ans_dict["stack_trace"] = traceback.format_exc()
                ans_dict["n_retry"] = self.max_retry

        self.actions.append(ans_dict["action"])
        self.memories.append(ans_dict.get("memory", None))
        self.thoughts.append(ans_dict.get("think", None))

        ans_dict["chat_messages"] = [m.content for m in chat_messages]
        ans_dict["chat_model_args"] = asdict(self.chat_model_args)

        return ans_dict["action"], ans_dict

    def _modify_prompt(self, sys_msg: str, fewshot_modification_type: str = 'None', insights_modification_type: str = 'None') -> str:
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
                        if insights_modification_type == 'wo':
                            replace_insights = '\n'
                        elif insights_modification_type == 'corrupt':
                            replace_insights = self._corrupt_insights(insights)
                        elif insights_modification_type == 'irrelevant':
                            replace_insights = self._replace_insights_with_irrelevant(insights)
                        elif insights_modification_type == 'filler_tokens':
                            replace_insights = self._replace_insights_with_filler_tokens(insights)

                        sys_msg = sys_msg.replace(insights, replace_insights)
        
        # if fewshot_modification_type != 'None':
        #     if fewshot_modification_type == 'shuffle':
        #         sys_msg = self._shuffle_fewshot(sys_msg)
        #     if fewshot_modification_type == 'irrelevant':
        #         sys_msg = self._irrelevant_fewshot(sys_msg)

        return sys_msg
            
    def _corrupt_insights(self, prompt: str) -> str:
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

    def _replace_insights_with_irrelevant(self, prompt: str) -> str:
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

    def _replace_insights_with_filler_tokens(self, prompt: str) -> str:
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

    def extract_workflows(self, text):
        # 找到所有匹配"Workflow 数字"的行号
        matches = re.finditer(r'^.*Workflow \d+:.*$', text, re.IGNORECASE | re.MULTILINE)
        indices = [match.start() for match in matches]
        
        if not indices:
            return []
        
        # 获取每行的起始和结束位置
        lines = text.splitlines(keepends=True)
        line_indices = []
        current_pos = 0
        for line in lines:
            line_indices.append((current_pos, current_pos + len(line)))
            current_pos += len(line)

        first_start = 0
        # 构建结果
        result = []
        for i in range(len(indices)):
            start = indices[i]
            if i < len(indices) - 1:
                end = indices[i+1]
            else:
                end = len(text)
            
            # 找到包含start和end的行
            start_line = None
            end_line = None
            for j, (line_start, line_end) in enumerate(line_indices):
                if line_start <= start < line_end:
                    start_line = j
                    if i == 0:
                        first_start = line_start
                if line_start <= end <= line_end:
                    end_line = j
                    break
            
            if start_line is not None and end_line is not None:
                extracted = ''.join(lines[start_line:end_line])
                result.append(extracted)

        before = text[:first_start]
        after = ''
        if result:
            parts = result[-1].split("\n\n")
            if len(parts) >= 3:
                result[-1] = "\n\n".join(parts[:-3]) + "\n\n"
                after = "\n\n".join(parts[-3:]) + lines[-1]
        
        return result, before, after

    def _shuffle_fewshot(self, prompt: str) -> str:
        workflows, before, after = self.extract_workflows(prompt)
        shuffled_workflows = workflows.copy()
        random.shuffle(shuffled_workflows)

        for i, workflows in enumerate(shuffled_workflows):
            lines = workflows.split('\n')
            if len(lines) > 3:
                pre = lines[:3]
                lines = lines[3:]
                random.shuffle(lines)
                shuffled_workflows[i] = '\n'.join(pre) + '\n'.join(lines)
        
        workflows = '\n\n'.join(shuffled_workflows)
        prompt = before + workflows + after
        return prompt
