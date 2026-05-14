"""
This module provides functions to extract and execute Python code from a string.

The functions are:
* extract_and_run_python_code(txt: str) -> str: Extracts and executes Python code from a string.
* execute_code_with_timeout(code: str, timeout: int = 3) -> str: Executes Python code with a timeout and returns the output.

Additional functions can be added as needed.
"""

import os
import tempfile
from subprocess import Popen, PIPE, TimeoutExpired

def extract_and_run_python_code(txt: str) -> str:
    """
    Extract and execute Python code from a provided string.

    Handles missing print statements for non-comment last lines,
    executes the code, and captures output or errors.

    Parameters:
        txt (str): Input string containing a possible Python code block.

    Returns:
        str: Execution result or error message wrapped in output formatting.
    """
    def extract_code(input_str: str) -> str:
        """Extract Python code block delimited by ```python and ```."""
        try:
            return input_str.split("```python", 1)[1].split("```", 1)[0].strip()
        except IndexError:
            raise ValueError("No valid Python code block found.")

    def ensure_print_statement(code: str) -> str:
        """
        Append a print statement if the last line isn't a comment or a print statement.
        """
        lines = code.splitlines()
        last_line = lines[-1].rstrip()
        if not last_line.startswith(("print(", "#", " ", "\t")) and (not ("return" in last_line)):# and len((last_line.split(" "))) == 1:
            lines[-1] = f"print({last_line})"
        return "\n".join(lines)

    if "```python" not in txt:
        return None  # Return early if no Python code block is present

    try:
        # Extract and sanitize the code
        code_block = extract_code(txt)
        code_with_print = ensure_print_statement(code_block)

        # Execute the code and return output
        python_output = execute_code_with_timeout(code_with_print)
        # return f"PYTHON CODE OUTPUT:\n'''\n{python_output}\n'''"
        return f"Output of the Python code above:\n```\n{python_output}\n```"

    except Exception as error:
        return f"PYTHON CODE OUTPUT:\n```\nError: {str(error)}\n```"


# Python code execution function with timeout
# TODO (msuzgun): Improve the security of this function by using a sandboxed environment
def execute_code_with_timeout(code: str, timeout: int = 3) -> str:
    """
    Execute Python code with a timeout and return the output.
    
    Parameters:
        code (str): Python code to execute.
        timeout (int): Timeout duration in seconds.

    Returns:
        str: Captured output or error message from the code execution.
    """
    with tempfile.NamedTemporaryFile(
        mode="w+t", suffix=".py", delete=False
    ) as temp_file:
        temp_file.write(code)
        temp_file.flush()

    try:
        # In case alias python=python3 is not set, use python3 instead of python
        process = Popen(["python3", temp_file.name], stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate(timeout=timeout)
        captured_output = stdout.decode().strip()
        error_output = stderr.decode().strip()

        if captured_output == "":
            if error_output != "":
                captured_output = f"Error in execution: {error_output}"
            else:
                captured_output = "(No output was generated. It is possible that you did not include a print statement in your code. If you want to see the output, please include a print statement.)"

    except TimeoutExpired:
        process.kill()
        captured_output = "Execution took too long, aborting..."

    finally:
        os.remove(temp_file.name)

    return captured_output