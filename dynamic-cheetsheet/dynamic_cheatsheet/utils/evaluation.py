import sys
import re
import os
from typing import List
# from .sonnet_eval import sonnet_errors
from .execute_code import execute_code_with_timeout


def clean_output_for_arithmetic(output: str) -> str:
    """
    Clean the output for arithmetic problems.

    Args:
        output (str): The output to clean.

    Returns:
        str: The cleaned output.
    """
    if "=" in output:
        output = output.split("=")[1].strip()
    if " is" in output:
        output = output.split(" is")[1].strip()
    if " equals" in output:
        output = output.split(" equals")[1].strip()
    if " evaluates to" in output:
        output = output.split(" evaluates to")[1].strip()
    if " is equal to" in output:
        output = output.split(" is equal to")[1].strip()
    return output


def clean_output_for_GameOf24(output: str) -> str:
    """
    Clean the output for GameOf24 problems.
    """
    if "=" in output:
        output = output.split("=")[0].strip()
    if "is" in output:
        output = output.split("is")[1].strip()
    if "equals" in output:
        output = output.split("equals")[0].strip()
    if "evaluates to" in output:
        output = output.split("evaluates to")[0].strip()
    return output


def eval_for_GameOf24(input: str, output: str) -> bool:
    """
    Given an input and output, check if the output is correct and follows the rules of the game.
    """
    clean_output = output

    clean_output = clean_output_for_GameOf24(output)
    clean_output = clean_output.replace("x", "*").strip()
    clean_output = clean_output.replace("×", "*").strip()
    clean_output = clean_output.replace("÷", "/").strip()
    
    try:
        # Get the value of the expression using eval
        value = eval(clean_output)
        if not (abs(value - 24) < 1e-3):
            return False
        # Split the input and output digits by space
        input_digits = input.split(" ")
        # Replace the following symbols with space
        replacements = ["+", "-", "*", "/", "÷", "(", ")"]
        for symbol in replacements:
            clean_output = clean_output.replace(symbol, " ")
        # Replace multiple spaces with single space
        clean_output = re.sub(" +", " ", clean_output)
        clean_output = clean_output.strip()
        output_digits = clean_output.split(" ")
        # Sort the digits
        input_digits.sort()
        output_digits.sort()
        # Check if the digits are the same
        if input_digits != output_digits:
            return False
        return True
    except Exception as e:
        return False


def remove_punctuation(output: str) -> str:
    """
    Remove punctuation from the output.
    """
    markers = [",", ";", ":", ".", '"']
    for marker in markers:
        output = output.replace(marker, "")
    return output


def convert_newline_to_space(output: str) -> str:
    """
    Convert newline to space.
    """
    output = output.replace("\n", " ")
    return output


def eval_for_exact_matching_with_no_punctuation(
    output: str, target: str
) -> bool:
    """
    Evaluate if the output is exactly the same as the target.
    """
    output = remove_punctuation(output)
    output = convert_newline_to_space(output)
    if target == output:
        return True
    return False


def eval_for_softmatch(input: str, output: str, target: str) -> bool:
    """
    Evaluate if the output is a soft match of the target.
    """
    output = remove_punctuation(output)
    if target in output:
        return True
    return False


def eval_for_CheckmateInOne(input: str, output: str, target: str) -> bool:
    """
    Evaluate if the output is a checkmate in one.
    """
    output = output.strip()
    if output[-1] == "#":
        output = output.split(" ")[-1].strip()
    # Based on the input, determine the number of the last move
    last_move = input.split(".")[-1].strip()
    move_idx = input.split(".")[-2].split(" ")[-1].strip()
    # If the last move is an empty string, then the last move is white; otherwise, it is black
    if last_move == "":
        last_move = "White"
    else:
        last_move = "Black"
    next_move_idx = str(int(move_idx) + 1)
    if not (next_move_idx in output):
        if target in output or (target[1] == 'x' and (target[0] + target[2:]) in output):
            return True
    else:
        output = output.split(next_move_idx)[0].strip()
        if target in output or (target[1] == 'x' and (target[0] + target[2:]) in output):
            return True
    return False


def eval_equation_balancer(input: str, output: str, target: str) -> bool:
    """
    Evaluate if the output is a valid equation balancer.
    """
    output = output.split("=")[0].strip()
    target_val = target.split("=")[1].strip()
    target = target.split("=")[0].strip()
    # First make sure that the output has the same format as the target (when operators (e.g., +, -, *, /) are removed)
    output_nums = output.replace("+", "").replace("-", "").replace("*", "").replace("/", "").replace(" ", "").strip()
    target_nums = target.replace("+", "").replace("-", "").replace("*", "").replace("/", "").replace(" ", "").strip()
    if output_nums != target_nums:
        return False
    # Now, evaluate the output and target
    try:
        output_value = eval(output)
        if abs(output_value - eval(target_val)) < 1e-6:
            return True
    except Exception as e:
        return False
    return False


def eval_for_multiple_choice(input_text: str, final_answer: str, target: str) -> bool:
    """
    Evaluates if the final answer matches the target using pattern matching.
    
    Args:
        input_text (str): The original question text including options
        final_answer (str): The model's answer
        target (str): The correct answer
    
    Returns:
        bool: True if answer is correct, False otherwise
    """
    # Handle empty or None inputs
    if not final_answer or not target:
        return False
    
    def clean_text(text: str) -> str:
        if not text:
            return ""
        return text.lower().strip().replace('`', '').replace('(', '').replace(')', '')
    
    def extract_option_text(input_text: str, option_letter: str) -> str:
        try:
            # Try different formats of options sections
            options_section = ""
            if 'options:' in input_text.lower():
                options_section = input_text.lower().split('options:')[1].strip()
            elif 'choices:' in input_text.lower():
                options_section = input_text.lower().split('choices:')[1].strip()
            
            if not options_section:
                # Try to find options in the format (A) text, (B) text
                lines = input_text.lower().split('\n')
                for i, line in enumerate(lines):
                    if line.strip().startswith(f'({option_letter})') or line.strip().startswith(f'{option_letter})'):
                        return line.split(')', 1)[1].strip()
                
            # Process the options section if found
            for line in options_section.split('\n'):
                line = line.strip()
                if line.startswith(f'({option_letter})') or line.startswith(f'{option_letter})'):
                    return line.split(')', 1)[1].strip()
                # Handle options like "A. text" format
                if line.startswith(f'{option_letter}.'):
                    return line.split('.', 1)[1].strip()
        except:
            return ''
        return ''

    # Full option match (A), (B), etc. (e.g., (A) == (A))
    if final_answer == target:
        return True

    # Clean and normalize inputs
    clean_answer = clean_text(final_answer)
    clean_target = clean_text(target)
    
    # Handle target formats: (A), A), A, etc.
    target_letter = ""
    if len(clean_target) == 1:
        target_letter = clean_target
    elif clean_target.endswith(')'):
        target_letter = clean_target[-2]
    else:
        # Extract the last character if it's a letter a-d or A-D
        last_char = clean_target[-1]
        if last_char in 'abcd':
            target_letter = last_char
    
    # Direct letter match (a, b, c, d)
    if len(clean_answer) == 1 and clean_answer in 'abcd' and clean_answer == target_letter:
        return True
    
    # Handle answer formats like "A" or "A."
    if clean_answer.startswith(target_letter) and (len(clean_answer) == 1 or 
                                                  (len(clean_answer) == 2 and clean_answer[1] == '.')):
        return True
    
    # Handle answer formats like "Option A" or "Answer is A"
    if clean_answer.endswith(target_letter) and (clean_answer[-2:] == f" {target_letter}" or 
                                               clean_answer[-3:] == f" {target_letter}."):
        return True
    
    # Text content match - check if the target option text is in the answer
    target_text = extract_option_text(input_text, target_letter)
    
    if target_text and target_text in clean_answer:
        return True
    
    # Handle numerical answers (if target is a number and answer contains that number)
    if target_letter.isdigit() and target_letter in clean_answer:
        return True
        
    return False


def eval_for_pyton_programming_puzzles(input: str, output: str) -> bool:
    """
    Evaluate if the output is a valid Python programming puzzle solution.
    """
    if "```python" in output:
        output = output.split("```python")[-1].strip()
        output = output.split("```")[0].strip()

    if "def sat" in output:
        if "from typing" not in output:
            output = f"from typing import *\n{output}"
        code = f"{output}\nanswer = solution()\nprint(sat(answer))"
    else:
        code = f"from typing import *\n{input}\n{output}\nanswer = solution()\nprint(sat(answer))"
    
    code = code.replace("List[", "list[")
    eval_bool = execute_code_with_timeout(code, timeout=3)

    if "NameError: name 'answer' is not defined" in eval_bool:
        print(f"Eval bool: {eval_bool}")
        print(f"Code:\n{code}")
        print("*" * 100)
    if "True" in eval_bool:
        return True
    return False
