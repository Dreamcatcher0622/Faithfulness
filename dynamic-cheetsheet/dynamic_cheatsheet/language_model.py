import numpy as np
import tiktoken
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from .utils.execute_code import extract_and_run_python_code
from .utils.extractor import extract_answer, extract_cheatsheet
from litellm import completion
from functools import partial
import time

class LanguageModel:
    def __init__(self,
        model_name: str,
    ) -> None:
        """
        LanguageModel class to interact with different language models.

        Arguments:
            model_name : str : The name of the language model to use.
            api_key : str : The API key for the model.

        Raises:
            ValueError : If the model name is not found.
        """

        self.model_name = model_name

        # Load the client for the model based on the model name
        if self.model_name in [
            "openai/gpt-4o-mini", "openai/gpt-4o-mini-2024-07-18",
            "openai/gpt-4o", "openai/gpt-4o-2024-08-06", "openai/gpt-4o-2024-11-20",
            "openai/gpt-3.5-turbo",
            "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "openai/o3-mini", "openai/o3-mini-2025-01-31",
            "openai/o1", "openai/o1-2024-12-17", "anthropic/claude-sonnet-4-5-20250929",
            "anthropic/claude-3-5-sonnet-20240620",
            "anthropic/claude-3-5-sonnet-latest", "anthropic/claude-3-5-sonnet-20241022",
            "anthropic/claude-3-5-haiku-latest", "anthropic/claude-3-5-haiku-20241022",
            "anthropic/claude-3-7-sonnet-latest", "anthropic/claude-3-7-sonnet-20250219",
            "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "together_ai/deepseek-ai/DeepSeek-R1",
            "together_ai/deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
            "together_ai/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            "together_ai/Qwen/Qwen2.5-Coder-32B-Instruct",
            "together_ai/Qwen/QwQ-32B",
            "together_ai/Qwen/Qwen2-72B-Instruct",
            "together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo",
            "together_ai/Qwen/Qwen2.5-72B-Instruct-Turbo",
            "gemini/gemini-2.0-flash",
            "ollama/llama3:70b",
        ]:
            self.client = partial(completion, model=self.model_name)
        else:
            raise ValueError(f"Model '{model_name}' not found.")
        
        self.gpt4Tokenizer = tiktoken.encoding_for_model('gpt-4o')
        

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the text.
        """
        tokens = self.gpt4Tokenizer.encode(text)
        return len(tokens)

    def generate(self,
        history: List[str],
        temperature: float = 0.1,
        max_tokens: int = 2048,
        current_depth: int = 1,
        max_depth_num_rounds: int = 3,
        allow_code_execution: bool = True,
        code_execution_flag: str = "EXECUTE CODE!",
        final_output: str = ""
    ) -> str:
        """
        Generate a response from the language model.

        Arguments:
            history : List[str] : The conversation history.
            temperature : float : The sampling temperature for the model.
            max_tokens : int : The maximum number of tokens to generate.
            current_depth : int : The current depth of the conversation.
            max_depth_num_rounds : int : The maximum number of rounds allowed.
            allow_code_execution : bool : Whether to allow code execution.
            code_execution_flag : str : The flag to trigger code execution.
            final_output : str : The final output to return.

        Returns:
            str : The final output of the conversation.

        Raises:
            ValueError : If the history is empty.
        """
        if len(history) == 0:
            raise ValueError("History must contain at least one message.")
        

        # Generate the response from the language model
        for i in range(3):
            try:
                if 'claude' in self.model_name:
                    output = self.client(
                        messages=history,
                        model=self.model_name,
                        temperature=temperature,
                        max_completion_tokens=max_tokens,
                    )['choices'][0]['message']["content"]
                else:
                    output = self.client(
                        messages=history,
                        model=self.model_name,
                        temperature=temperature,
                        max_completion_tokens=max_tokens,
                    ).choices[0].message["content"]
                break
            except Exception as e:
                if i == 2:  # 最后一次失败
                    raise RuntimeError("API request failed after 3 retries") from e
                time.sleep(5)

        # If Python code execution is allowed, execute the code
        pre_code_execution_flag = output.split(code_execution_flag)[0].strip()
        if allow_code_execution and code_execution_flag in output and '```' == pre_code_execution_flag[-3:]:
            if code_execution_flag in output:
                output_prefix = output.split(code_execution_flag)[0].strip()
            else:
                # TODO (msuzgun): This is a temporary solution. We may want to find a better way to handle this.
                output_prefix = output
            executed_code = extract_and_run_python_code(output_prefix)
            executed_code = executed_code.strip()
            current_output = f"{output_prefix}\n{code_execution_flag}\n\n{executed_code}"
            final_output = f"{final_output}\n\n{current_output}".strip()
            # import pdb; pdb.set_trace()
            # print(f"*** This code has been executed:\n{executed_code}\n\n")
            # print(f"***And the output is:\n{current_output}")
            # If the current depth is less than or equal to the maximum depth, add a new message to the history
            if current_depth <= max_depth_num_rounds:
                warning_txt = ""
                if current_depth == max_depth_num_rounds:
                    warning_txt = f" (This is the last round. No more code execution will be allowed. Please present your final solution now.)"
                new_messages = [
                    {"role": "assistant", "content": current_output},
                    {"role": "user", "content": f"Proceed with any additional steps required and provide the completed solution. If everything is already complete, type FINAL ANSWER and submit it in the expected format. If you are stucked, please try alternative methods to solve the problem and provide the final solution.{warning_txt}"}
                ]
                history += new_messages
                return self.generate(
                    history=history,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    current_depth=current_depth+1,
                    max_depth_num_rounds=max_depth_num_rounds,
                    allow_code_execution=allow_code_execution,
                    code_execution_flag=code_execution_flag,
                    final_output=final_output,
                )
            else:
                final_output = f"{final_output}\n\n{current_output}".strip()
                return final_output
        else:
            # If code execution is not allowed or no code block is found, return the final output
            # Add the output to the final output
            final_output = f"{final_output}\n\n{output}".strip()
            return final_output

    def advanced_generate(self,
        approach_name: str,
        input_txt: str,
        cheatsheet: str = None,
        generator_template: str = None,
        cheatsheet_template: str = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        max_num_rounds: int = 1,
        allow_code_execution: bool = True,
        code_execution_flag: str = "EXECUTE CODE!",
        add_previous_answers_to_cheatsheet: bool = True,
        original_input_corpus: List[str] = None,
        original_input_embeddings: np.ndarray = None,
        generator_outputs_so_far: List[str] = None,
        retrieve_top_k: int = 3,
        faithfulness_experiment: bool = False,
        insights_modification_type: str = "None",
        fewshot_modification_type: str = "None",
        generator_cheatsheet_so_far: List[str] = None,
    ) -> Tuple[str, str, str, str]:
        """
        Generate a response from the language model.

        Arguments:
            approach_name : str : The name of the approach to use.
            input_txt : str : The input text for the model.
            cheatsheet : str : The cheatsheet for the model.
            generator_template : str : The template for the generator model.
            cheatsheet_template : str : The template for the cheatsheet extraction model.
            temperature : float : The sampling temperature for the model.
            max_tokens : int : The maximum number of tokens to generate.
            max_num_rounds : int : The maximum number of rounds allowed.
            allow_code_execution : bool : Whether to allow code execution.
            code_execution_flag : str : The flag to trigger code execution.
            add_previous_answers_to_cheatsheet : bool : Whether to add the previous answers to the cheatsheet.
            original_input_corpus : List[str] : The original input corpus.
            original_input_embeddings : np.ndarray : The original input embeddings.
            generator_outputs_so_far : List[str] : The generator outputs so far.
            retrieve_top_k : int : The number of top k inputs to retrieve.

        Returns:
            Tuple[str, str, str, str] : The generator answer, evaluator solution, answer check, and new cheatsheet.

        Raises:
            ValueError : If the proper templates are not provided.
        """

        # If the approach name is "default", run the generator model with the input text and the current cheatsheet
        if approach_name == "default":
            generator_prompt = generator_template.replace("[[QUESTION]]", input_txt).replace("[[CHEATSHEET]]", "(empty)")
            generator_history = [
                {"role": "user", "content": generator_prompt},
            ]
            generator_output = self.generate(
                history=generator_history,
                temperature=temperature,
                max_tokens=max_tokens,
                allow_code_execution=allow_code_execution,
                code_execution_flag=code_execution_flag,
            )

            generator_answer = extract_answer(
                generator_output,
            )

            return {
                "input_txt": input_txt,
                "steps": [
                    {
                        "round": 0,
                        "generator_prompt": generator_prompt,
                        "generator_output": generator_output,
                        "generator_answer": generator_answer,
                        "current_cheatsheet": None,
                        "new_cheatsheet": None,
                    }
                ],
                "previous_answers": None,
                "final_answer": generator_answer,
                "final_output": generator_output,
                "final_cheatsheet": None,
                "generator_output": generator_output,
            }
        
        elif approach_name == "DynamicCheatsheet_Cumulative":
            if cheatsheet is None:
                raise ValueError("Cheatsheet must be provided for dynamic_cheatsheet approach.")
            if cheatsheet_template is None:
                raise ValueError("Cheatsheet template must be provided for dynamic_cheatsheet approach.")
            
            steps = []
            previous_answers = []

            generator_output = ''

            for round in range(max(1, max_num_rounds)):
                ## STEP 1: Run the generator model with the input text and the cheatsheet
                generator_cheatsheet_content = cheatsheet

                # If there are previous answers, add them to the cheatsheet content for the generator
                if round > 0 and add_previous_answers_to_cheatsheet:
                    previous_answers_txt = f"PREVIOUS ANSWERS:\n{'; '.join(previous_answers)}"
                    generator_cheatsheet_content = f"{generator_cheatsheet_content}\n\n{previous_answers_txt}"

                generator_prompt = generator_template.replace("[[QUESTION]]", input_txt).replace("[[CHEATSHEET]]", generator_cheatsheet_content)
                current_cheatsheet = cheatsheet

                # Prepare the message history for the generator model
                generator_history = [{"role": "user", "content": generator_prompt}]
                # Run the generator model
                generator_output = self.generate(
                    history=generator_history,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    allow_code_execution=allow_code_execution,
                    code_execution_flag=code_execution_flag,
                )
                # Extract the output from the generator model
                generator_answer = extract_answer(generator_output)

                ## STEP 2: Run the cheatsheet extraction model with the generator output and the current cheatsheet
                cheatsheet_prompt = cheatsheet_template.replace("[[QUESTION]]", input_txt).replace("[[MODEL_ANSWER]]", generator_output).replace("[[PREVIOUS_CHEATSHEET]]", current_cheatsheet)

                cheatsheet_history = [{"role": "user", "content": cheatsheet_prompt}]
                cheatsheet_output = self.generate(
                    history=cheatsheet_history,
                    temperature=temperature,
                    max_tokens=2*max_tokens,
                    allow_code_execution=False,
                )

                # Extract the new cheatsheet from the output (if present); otherwise, return the old cheatsheet
                new_cheatsheet = extract_cheatsheet(response=cheatsheet_output, old_cheatsheet=current_cheatsheet)
                cheatsheet = new_cheatsheet

                previous_answers.append(f"Round {round+1}: {generator_answer}")
            
                steps.append({
                    "round": round,
                    "generator_prompt": generator_prompt,
                    "generator_output": generator_output,
                    "generator_answer": generator_answer,
                    "current_cheatsheet": current_cheatsheet,
                    "new_cheatsheet": new_cheatsheet,
                })

            return {
                "input_txt": input_txt,
                "steps": steps,
                "previous_answers": previous_answers,
                "final_answer": generator_answer,
                "final_cheatsheet": new_cheatsheet,
                "final_output": generator_output,
            }
        elif approach_name == "FullHistoryAppending":
            length_of_history = len(generator_outputs_so_far)
            if length_of_history > 0:
                top_k_original_inputs = original_input_corpus[:length_of_history]
                top_k_original_outputs = generator_outputs_so_far

                curated_cheatsheet = "### PREVIOUS SOLUTIONS (START)\n\n"
                for i, (previous_input_txt, previous_output_txt) in enumerate(zip(original_input_corpus, generator_outputs_so_far)):
                    curated_cheatsheet += f"#### Previous Input #{i+1}:\n\n{previous_input_txt}\n\n#### Model Solution to Previous Input #{i+1}:\n\n{previous_output_txt}\n---\n---\n\n"
                curated_cheatsheet += "#### PREVIOUS SOLUTIONS (END)"
            else:
                top_k_original_inputs = []
                top_k_original_outputs = []
                curated_cheatsheet = "(empty)"
            
            # Replace the relevant placeholders in the generator template with the input text and the curated cheatsheet and then run the generator model
            generator_prompt = generator_template.replace("[[QUESTION]]", input_txt).replace("[[CHEATSHEET]]", curated_cheatsheet)
            generator_history = [{"role": "user", "content": generator_prompt}]
            generator_output = self.generate(
                    history=generator_history,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    allow_code_execution=allow_code_execution,
                    code_execution_flag=code_execution_flag,
                )
            # Extract the answer from the generator model
            generator_answer = extract_answer(generator_output)

            return {
                "input_txt": input_txt,
                "steps": [
                    {
                        "round": 0,
                        "generator_prompt": generator_prompt,
                        "generator_output": generator_output,
                        "generator_answer": generator_answer,
                        "current_cheatsheet": curated_cheatsheet,
                        "new_cheatsheet": None,
                    }
                ],
                "top_k_original_inputs": top_k_original_inputs,
                "top_k_original_outputs": top_k_original_outputs,
                "final_answer": generator_answer,
                "final_output": generator_output,
                "final_cheatsheet": curated_cheatsheet,
            }
        elif approach_name in ["Dynamic_Retrieval", "DynamicCheatsheet_RetrievalSynthesis"]:
            # Get the current original input embedding
            current_original_input_embedding = original_input_embeddings[-1] # Current original input embedding
            prev_original_input_embeddings = original_input_embeddings[:-1] # Note that this can be empty
            
            # Retrieve the most similar k input-output pairs from the previous inputs and outputs
            if len(prev_original_input_embeddings) > 0:
                similarities = cosine_similarity([current_original_input_embedding], prev_original_input_embeddings)
                top_k_indices = np.argsort(similarities[0])[::-1][:retrieve_top_k]
                top_k_original_inputs = [original_input_corpus[i] for i in top_k_indices]
                top_k_original_outputs = [generator_outputs_so_far[i] for i in top_k_indices]
                top_k_similar_values = similarities[0][top_k_indices]
                # Use the retrieved pairs to curate the cheatsheet for the generator model
                curated_cheatsheet = "### PREVIOUS SOLUTIONS (START)\n\nNote: The input-output pairs listed below are taken from previous test cases and are meant to assist you in understanding potential solution strategies or tool usages. While they can offer insight and inspiration, they should not be blindly copied, as they may contain errors or may not fit your specific use case. Approach them with a critical mindset—analyze their logic, verify their correctness, and adapt them as needed. Your goal should be to develop a well-reasoned solution that best addresses the problem at hand.\n\n"
            else:
                top_k_original_inputs = []
                top_k_original_outputs = []
                top_k_similar_values = []
                curated_cheatsheet = '(empty)'
            
            # The following only adds the previous input-output pairs to the cheatsheet
            for i, (previous_input_txt, previous_output_txt, similarity) in enumerate(zip(top_k_original_inputs[::-1], top_k_original_outputs[::-1], top_k_similar_values[::-1])):
                curated_cheatsheet += f"#### Previous Input #{i+1} (Similarity: {similarity:.2f}):\n\n{previous_input_txt}\n\n#### Model Solution to Previous Input  #{i+1}:\n\n{previous_output_txt}\n---\n---\n\n"
            curated_cheatsheet = curated_cheatsheet.strip()
            
            # If it is empty, we should not add the "PREVIOUS SOLUTIONS (END)" to the cheatsheet
            if curated_cheatsheet != '(empty)':
                curated_cheatsheet += "\n\n#### PREVIOUS SOLUTIONS (END)"

            # Run the Generator model with the input text and the curated cheatsheet (input-output pairs) to generate a better (more tailored) cheatsheet   
            previous_cheatsheet = cheatsheet
            if approach_name == "DynamicCheatsheet_RetrievalSynthesis":
                # First, we need to make the necessary replacements in the cheatsheet template
                cheatsheet_prompt = cheatsheet_template.replace("[[PREVIOUS_INPUT_OUTPUT_PAIRS]]", curated_cheatsheet)
                cheatsheet_prompt = cheatsheet_prompt.replace("[[NEXT_INPUT]]", input_txt)
                cheatsheet_prompt = cheatsheet_prompt.replace("[[PREVIOUS_CHEATSHEET]]", previous_cheatsheet)
                # Now, we are ready to run the cheatsheet curator model
                cheatsheet_history = [{"role": "user", "content": cheatsheet_prompt}]
                cheatsheet_output = self.generate(
                    history=cheatsheet_history,
                    temperature=temperature,
                    max_tokens=2*max_tokens,
                    allow_code_execution=False,
                )
                # Finally, extract the new cheatsheet from the output (if present); otherwise, return the old cheatsheet
                new_cheatsheet = extract_cheatsheet(response=cheatsheet_output, old_cheatsheet=curated_cheatsheet)
                curated_cheatsheet = new_cheatsheet

            # Replace the relevant placeholders in the generator template with the input text and the curated cheatsheet and then run the generator model
            generator_prompt = generator_template.replace("[[QUESTION]]", input_txt).replace("[[CHEATSHEET]]", curated_cheatsheet)
            
            # 对insight和fewshot进行更改 --------------------------
            if faithfulness_experiment:
                generator_prompt = _modify_prompt(generator_prompt, insights_modification_type, fewshot_modification_type,
                            current_original_input_embedding, prev_original_input_embeddings, generator_cheatsheet_so_far)
            # --------------------------

            generator_history = [{"role": "user", "content": generator_prompt}]
            generator_output = self.generate(
                    history=generator_history,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    allow_code_execution=allow_code_execution,
                    code_execution_flag=code_execution_flag,
                )
            # Extract the answer from the generator model
            generator_answer = extract_answer(generator_output)

            return {
                "input_txt": input_txt,
                "steps": [
                    {
                        "round": 0,
                        "generator_prompt": generator_prompt,
                        "generator_output": generator_output,
                        "generator_answer": generator_answer,
                        "current_cheatsheet": curated_cheatsheet,
                        "new_cheatsheet": None,
                    }
                ],
                "top_k_original_inputs": top_k_original_inputs,
                "top_k_original_outputs": top_k_original_outputs,
                "final_answer": generator_answer,
                "final_output": generator_output,
                "final_cheatsheet": curated_cheatsheet,
            }
        else:
            raise ValueError(f"Approach '{approach_name}' not found.")

import re
import random
def _modify_prompt(generator_prompt, insights_modification_type, fewshot_modification_type, current_original_input_embedding, prev_original_input_embeddings, generator_cheatsheet_so_far):
    """
    修改 generator_prompt 样例以测试 Faithfulness
    
    Args:
        generator_prompt: 原始的 prompt 字符串
        insights_modification_type: insights 修改类型 ("none", "corrupted", "irrelevant", "filler_tokens")
        fewshot_modification_type: fewshot 修改类型 ("none", "shuffle", "irrelevant")
    
    Returns:
        修改后的 prompt 字符串
    """
    # print(f"Here is insights_modification_type:{insights_modification_type}")
    if insights_modification_type != 'None':
        pattern = r"<description>(.*?)</description>"
        matches = re.findall(pattern, generator_prompt, re.DOTALL)
        modified_prompt = generator_prompt
        # print(f"Here is matches:{matches}")
        
        for match in matches:
            if insights_modification_type == "corrupted":
                modified_content = _corrupt_insights(match)
            elif insights_modification_type == "irrelevant":
                modified_content = _replace_insights_with_irrelevant(match)
            elif insights_modification_type == "filler_tokens":
                modified_content = _replace_insights_with_filler_tokens(match)
            elif insights_modification_type == "wo" or insights_modification_type == 'empty':
                modified_content = ''
            
            modified_prompt = modified_prompt.replace(match, modified_content, 1)
        
        if insights_modification_type == 'wo':
            modified_prompt = modified_prompt.replace('<description>', '')
            modified_prompt = modified_prompt.replace('</description>', '')
        
        generator_prompt = modified_prompt

    if fewshot_modification_type != 'None':
        if fewshot_modification_type == "shuffle":
            pattern = r"<example>(.*?)</example>"
            matches = re.findall(pattern, generator_prompt, re.DOTALL)
            modified_prompt = generator_prompt
            for match in matches:
                modified_content = _shuffle_fewshot(match)
                modified_prompt = modified_prompt.replace(match, modified_content, 1)
        elif fewshot_modification_type == "irrelevant":
            modified_prompt = _swap_fewshot(generator_prompt, current_original_input_embedding, prev_original_input_embeddings, generator_cheatsheet_so_far)
        elif fewshot_modification_type == "wo" or fewshot_modification_type == 'empty':
            pattern = r"<example>(.*?)</example>"
            matches = re.findall(pattern, generator_prompt, re.DOTALL)
            modified_prompt = generator_prompt
            for match in matches:
                modified_prompt = modified_prompt.replace(match, "", 1)
            
            if fewshot_modification_type == "wo":
                modified_prompt = modified_prompt.replace('<example>', "")
                modified_prompt = modified_prompt.replace('</example>', "")

        generator_prompt = modified_prompt

    return generator_prompt

def _corrupt_insights(prompt: str) -> str:
    """
    随机毁坏给定的字符串内容
    
    Args:
        prompt: 原始字符串
            
    Returns:
        损坏后的字符串
    """
    # 将输入字符串分割为句子或段落
    insights = re.findall(r'[^.,\n:]+[.,\n:]?', prompt, re.DOTALL)
    # insights = re.split(r'(?<=[.,])\s+', prompt, re.DOTALL) 
    # print(f"Here is insights:{insights}")
    replaced_insights = []

    for insight in insights:
        if len(insight) > 10:
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
            replaced_insights.append(' '.join(words))
    
    # 返回损坏后的字符串
    corrupted_prompt = ''.join(replaced_insights)
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
    insights = re.findall(r'[^.,\n:]+[.,\n:]?', prompt, re.DOTALL)
    replaced_insights = []

    for insight in insights:
        if len(insight) > 10:
            if random.random() < 0.8:  # 80% 概率替换
                # 随机选择一个不相关的模板
                replacement = random.choice(irrelevant_templates)
                replaced_insights.append(replacement)
            else:
                replaced_insights.append(insight)

    # 将替换后的部分组合回一个字符串
    replaced_prompt = ''.join(replaced_insights)
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
    insights = re.findall(r'[^.,\n:]+[.,\n:]?', prompt, re.DOTALL)
    filled_insights = []

    for insight in insights:
        if len(insight) > 10:
            if random.random() < 0.8:  # 80% 概率替换
                # 随机选择一个填充字符模板
                filler = random.choice(filler_tokens)
                filled_insights.append(filler)
            else:
                filled_insights.append(insight)

    # 将替换后的部分组合回一个字符串
    filled_prompt = ''.join(filled_insights)
    return filled_prompt

def _shuffle_fewshot(prompt: str) -> str:
    """
    根据输入的字符串内容进行轨迹行分组和打乱
    
    Args:
        prompt: 原始字符串
            
    Returns:
        打乱后的字符串
    """
    # 将输入字符串按行分割
    trajectory_lines = prompt.strip().split('\n')
    # 忽略空行
    trajectory_lines = [line for line in trajectory_lines if line]
    # 随机打乱行
    random.shuffle(trajectory_lines)
    # 将打乱后的行组合回一个字符串
    shuffled_prompt = '\n'.join(trajectory_lines)
    
    return shuffled_prompt

def _swap_fewshot(prompt: str, current_original_input_embedding=[], prev_original_input_embeddings=[], generator_cheatsheet_so_far=[]) -> str:
    """
    选择历史cheatsheet中与当前问题最无关的，用其<example>替换当前<example>
    
    Args:
        prompt: 原始字符串
            
    Returns:
        替换后的字符串
    """
    if len(prev_original_input_embeddings) > 1 and '<example>' in prompt:
        similarities = cosine_similarity([current_original_input_embedding], prev_original_input_embeddings[1:])
        index = np.argsort(similarities[0])[::-1][-1]
        print(f"Here is index:{index+1}")
        not_relevant_cheatsheet = generator_cheatsheet_so_far[index+1]

        if '<example>' in not_relevant_cheatsheet:
            # 匹配 <example></example> 内容
            prompt_examples = re.findall(r'<example>(.*?)</example>', prompt, re.DOTALL)
            cheatsheet_examples = re.findall(r'<example>(.*?)</example>', not_relevant_cheatsheet, re.DOTALL)
            
            print(f"Here is len1:{len(prompt_examples)}")
            print(f"Here is len2:{len(cheatsheet_examples)}")
            # 替换 prompt 中的 <example> 部分
            for i in range(len(prompt_examples)):
                if i < len(cheatsheet_examples):
                    if prompt_examples[i] != cheatsheet_examples[i]:
                        print("Yes!")
                    prompt = prompt.replace(prompt_examples[i], cheatsheet_examples[i])
                else:
                    if prompt_examples[i] != cheatsheet_examples[i % len(cheatsheet_examples)]:
                        print("Yes!")
                    prompt = prompt.replace(prompt_examples[i], cheatsheet_examples[i % len(cheatsheet_examples)])
            
    return prompt