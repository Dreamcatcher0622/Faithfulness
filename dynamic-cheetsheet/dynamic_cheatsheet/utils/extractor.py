"""
This file contains the functions to extract the final answer, cheatsheet and solution evaluation from model responses.

The functions are:
* extract_answer(response: str) -> str: Extracts the final answer from the model response.
* extract_cheatsheet(response: str, old_cheatsheet: str) -> str: Extracts the cheatsheet from the model response.
* extract_solution(response: str, header: str = "SOLUTION EVALUATION:", error_message : str = "No solution evaluation found") -> str: Extracts the solution evaluation from the model response.

Additional functions can be added as needed.
"""

def extract_answer(
    response: str,
) -> str:
    """
    Extracts the final answer from the model response.

    Arguments:
        response : str : The response from the model.

    Returns:
        str : The extracted final answer (if not found, returns "No final answer found").
    """
    if "<answer>" in response:
        # <answer> (content) </answer>
        try:
            txt = response.split("<answer>")[-1].strip()
            txt = txt.split("</answer>")[0].strip()
            return txt
        except:
            return "No final answer found"
    else:
        if not("FINAL ANSWER" in response):
            return "No final answer found"
        try:
            response = response.split("FINAL ANSWER")[-1].strip()
            if response[0] == ":":
                response = response[1:].strip()

            # First decide whether to split by "```" or "'''" based on the presence of "```" or "'''"
            idx_1 = response.find("'''")
            idx_2 = response.find("```")
            if min(idx_1, idx_2) != -1: 
                if idx_1 < idx_2:
                    response = response.split("'''")[1].strip()
                else:
                    response = response.split("```")[1].strip()
            else:
                if idx_1 == -1:
                    response = response.split("```")[1].strip()
                else:
                    response = response.split("'''")[1].strip()

            # Special case for P3-Test task: If the first line contains "python" then remove it
            if response.split("\n")[0].strip().lower() == "python":
                response = "\n".join(response.split("\n")[1:]).strip()
            return response
        except:
            return "No final answer found"


def extract_cheatsheet(
    response: str,
    old_cheatsheet: str,
) -> str:
    """
    Extracts the cheatsheet from the model response.
    
    Arguments:
        response : str : The response from the model.
        old_cheatsheet : str : The old cheatsheet to return if the new one is not found.

    Returns:
        str : The extracted cheatsheet (if not found, returns the old cheatsheet).
    """
    response = response.strip()
    # <cheatsheet> (content) </cheatsheet>
    if "<cheatsheet>" in response:
        try:
            txt = response.split("<cheatsheet>")[1].strip()
            txt = txt.split("</cheatsheet>")[0].strip()
            return txt
        except:
            return old_cheatsheet
    else:
        return old_cheatsheet


def extract_solution(
    response: str,
    header: str = "SOLUTION EVALUATION:",
    error_message : str = "No solution evaluation found",
) -> str:
    """
    Extracts the solution evaluation from the model response.

    Arguments:
        response : str : The response from the model.
        header : str : The header to search for the solution evaluation.
        error_message : str : The error message to return if the solution evaluation is not found.

    Returns:
        str : The extracted solution evaluation (if not found, returns the error message).
    """
    response = response.strip()
    try:
        txt = response.split(header)[1]
        try:
            txt = txt.split("'''")[1].strip()
        except:
            return txt.strip()
    except:
        return response
        # return error_message
    return txt
