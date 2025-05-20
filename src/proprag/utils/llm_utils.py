import json
import re
from string import Template
from typing import (
    Optional,
    Union,
    List,
    TypedDict
)

from openai import (
    APIConnectionError,
    RateLimitError,
    Timeout
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from .config_utils import BaseConfig


class TextChatMessage(TypedDict):
    """Representation of a single text-based chat message in the chat history."""
    role: str  # Either "system", "user", or "assistant"
    content: Union[str, Template]  # The text content of the message (could also be a string.Template instance)


def convert_format_to_template(original_string: str, placeholder_mapping: Optional[dict] = None, static_values: Optional[dict] = None) -> str:
    """
    Converts a .format() style string to a Template-style string.

    Args:
        original_string (str): The original string using .format() placeholders.
        placeholder_mapping (dict, optional): Mapping from original placeholder names to new placeholder names.
        static_values (dict, optional): Mapping from original placeholders to static values to be replaced in the new template.

    Returns:
        str: The converted string in Template-style format.
    """
    # Initialize mappings
    placeholder_mapping = placeholder_mapping or {}
    static_values = static_values or {}

    # Regular expression to find .format() style placeholders
    placeholder_pattern = re.compile(r'\{(\w+)\}')

    # Substitute placeholders in the string
    def replace_placeholder(match):
        original_placeholder = match.group(1)

        # If the placeholder is in static_values, substitute its value directly
        if original_placeholder in static_values:
            return str(static_values[original_placeholder])

        # Otherwise, rename the placeholder if needed, or keep it as is
        new_placeholder = placeholder_mapping.get(original_placeholder, original_placeholder)
        return f'${{{new_placeholder}}}'

    # Replace all placeholders
    template_string = placeholder_pattern.sub(replace_placeholder, original_string)

    return template_string


def safe_unicode_decode(content: Union[bytes, str]) -> str:
    """
    Safely decodes a string or byte sequence containing Unicode escape sequences into a properly decoded Unicode string.

    Args:
        content (Union[bytes, str]): The input data, either as a byte sequence or a string, which may contain Unicode escape sequences.

    Returns:
        str: The decoded Unicode string with escape sequences replaced by their corresponding characters.
    
    Raises:
        AttributeError: If the input is neither bytes nor a string.
    """
    # Regular expression to find all Unicode escape sequences of the form \uXXXX
    unicode_escape_pattern = re.compile(r"\\u([0-9a-fA-F]{4})")

    # Function to replace the Unicode escape with the actual character
    def replace_unicode_escape(match: re.Match) -> str:
        # Convert the matched hexadecimal value into the actual Unicode character
        return chr(int(match.group(1), 16))

    # If content is bytes, decode it to a UTF-8 string
    if isinstance(content, bytes):
        content = content.decode("utf-8")
    elif not isinstance(content, str):
        raise AttributeError("Input must be of type bytes or str.")

    # Perform the substitution
    decoded_content = unicode_escape_pattern.sub(replace_unicode_escape, content)

    return decoded_content


def dynamic_retry(experiment_config: BaseConfig):
    """
    Factory function to create a retry decorator with dynamic parameters.

    Args:
        experiment_config (BaseConfig): Configuration containing stop and wait parameters. Expected to use the global config for running all experiments. 

    Returns:
        Callable: A retry decorator with dynamically set parameters.
    """
    stop = stop_after_attempt(experiment_config.async_max_retry_attempts)
    wait = wait_exponential(
        multiplier=experiment_config.async_retry_wait_exp_multiplier,
        min=experiment_config.async_retry_min_wait_exp_time,
        max=experiment_config.async_retry_max_wait_exp_time,
    )

    return retry(
        stop=stop,
        wait=wait,
        retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
    )


# def fix_broken_generated_json(json_str: str) -> str:
#     """
#     Fixes a malformed JSON string by:
#     - Removing the last comma and any trailing content.
#     - Appending a closing bracket `]` and brace `}` to properly terminate the JSON.

#     Args:
#         json_str (str): The malformed JSON string to be fixed.

#     Returns:
#         str: The corrected JSON string.
#     """
#     last_comma_index = json_str.rfind(',')
#     if last_comma_index != -1:
#         json_str = json_str[:last_comma_index]

#     processed_string = json_str + ']\n}'
#     return processed_string


# def fix_broken_generated_json(json_str: str) -> str:
#     """
#     Fixes a malformed JSON string by:
#     - Removing the last comma and any trailing content.
#     - Iterating over the JSON string once to determine and fix unclosed braces or brackets.
#     - Ensuring braces and brackets inside string literals are not considered.

#     If the original json_str string can be successfully loaded by json.loads(), will directly return it without any modification.
    
#     Args:
#         json_str (str): The malformed JSON string to be fixed.

#     Returns:
#         str: The corrected JSON string.
#     """

#     def find_unclosed(json_str):
#         """
#         Identifies the unclosed braces and brackets in the JSON string.

#         Args:
#             json_str (str): The JSON string to analyze.

#         Returns:
#             list: A list of unclosed elements in the order they were opened.
#         """
#         unclosed = []
#         inside_string = False
#         escape_next = False

#         for char in json_str:
#             if inside_string:
#                 if escape_next:
#                     escape_next = False
#                 elif char == '\\':
#                     escape_next = True
#                 elif char == '"':
#                     inside_string = False
#             else:
#                 if char == '"':
#                     inside_string = True
#                 elif char in '{[':
#                     unclosed.append(char)
#                 elif char in '}]':
#                     if unclosed and ((char == '}' and unclosed[-1] == '{') or (char == ']' and unclosed[-1] == '[')):
#                         unclosed.pop()

#         return unclosed

#     try:
#         # Try to load the JSON to see if it is valid
#         json.loads(json_str)
#         return json_str  # Return as-is if valid
#     except json.JSONDecodeError as e:
#         pass

#     # Step 1: Remove trailing content after the last comma.
#     last_comma_index = json_str.rfind(',')
#     if last_comma_index != -1:
#         json_str = json_str[:last_comma_index]

#     # Step 2: Identify unclosed braces and brackets.
#     unclosed_elements = find_unclosed(json_str)

#     # Step 3: Append the necessary closing elements in reverse order of opening.
#     closing_map = {'{': '}', '[': ']'}
#     for open_char in reversed(unclosed_elements):
#         json_str += closing_map[open_char]

#     return json_str

def fix_broken_generated_json(partial_json):
    """
    Attempts to parse a potentially broken JSON string by fixing common issues.
    """
    # TODO: Does not currently support boolean values
    # TODO: Does not currently support json_string that represents a list, but internal list in a dictionary is supported
    # TODO: Currently only support auto-fix quotes in string value field of a dictionary
    stack = []
    in_string = False
    escaped = False
    last_key_index = None
    last_string_value_index = None
    last_comma_index = None
    colon_index = None
    real_last_comma_index = None

    start = partial_json.find("{")
    if start == -1:
        start = partial_json.find("[")
    if start == -1:
        return ""
    partial_json = partial_json[start:]

    replace_targets = []
    for i, char in enumerate(partial_json):
        if not in_string:
            if char == "{":
                stack.append("}")
                last_key_index = None
                last_string_value_index = None
                last_comma_index = None
                colon_index = None
            elif char == "[":
                stack.append("]")
            elif char == '"':
                in_string = True
                if (
                    last_key_index is not None
                    and colon_index is not None
                    and colon_index > last_key_index
                    and (
                        (
                            last_comma_index is not None
                            and last_key_index > last_comma_index
                        )
                        or last_comma_index is None
                    )
                ):
                    # Avoid the case of {"key1": [1, 2], "key2}
                    last_string_value_index = i
                else:
                    last_key_index = i
            elif char == ":" and stack and stack[-1] == "}":
                colon_index = i
            elif char == ",":
                real_last_comma_index = i
                if stack and stack[-1] == "}":
                    last_comma_index = i
            elif char in "}]":
                if not stack or char != stack[-1]:
                    return None  # Invalid Format
                stack.pop()
                if not stack:
                    break
            elif (
                char not in " \t\r\n0123456789."
            ):  # Invalid case, try resolve cases of missing opening quote for key or value.
                replace_char = f'"{char}'
                replace_targets.append((i, replace_char))
                in_string = True
                if (
                    last_key_index is not None
                    and colon_index is not None
                    and colon_index > last_key_index
                    and (
                        (
                            last_comma_index is not None
                            and last_key_index > last_comma_index
                        )
                        or last_comma_index is None
                    )
                ):
                    # Avoid the case of {"key1": [1, 2], "key2}
                    last_string_value_index = i
                else:
                    last_key_index = i
        else:
            if char == '"' and not escaped:
                if stack[-1] == "]" or ( # in a string value field of a dictionary
                    last_string_value_index is not None
                    and last_string_value_index > last_key_index
                    and stack[-1] == "}"
                ):
                    # Check if the quote is incorrectly placed by looking at the first non-whitespace character after it
                    match = re.search(r"\S", partial_json[i + 1 :])
                    # No more non-space character ensues it, then we should assume end of string
                    # Assumes stack is populated, which should be the case
                    if match is None or match.group() in [",", stack[-1]]:
                        in_string = False
                    else:
                        replace_targets.append((i, '\\"'))
                else:
                    in_string = False
            elif char == "\\":
                escaped = not escaped
            elif char in "\t\n\r":
                if char == "\t":
                    replace_char = "\\t"
                elif char == "\n":
                    replace_char = "\\n"
                else:
                    replace_char = "\\r"
                replace_targets.append((i, replace_char))
            else:
                escaped = False
    partial_json = partial_json[: i + 1]  # truncate whatever is after
    if not in_string and partial_json[-1] == ".":  # partial floating point number
        partial_json = partial_json[:-1]

    if in_string:
        if (
            stack[-1] == "}"
            and last_string_value_index
            and last_string_value_index > last_key_index
        ) or stack[-1] == "]":  # in value field
            if escaped:
                partial_json += '\\"'
            else:
                partial_json += '"'
        else:  # in key field, then remove the key
            partial_json = partial_json[:last_key_index]

    if (
        real_last_comma_index
        and partial_json[real_last_comma_index + 1 :].isspace()
        or real_last_comma_index == len(partial_json) - 1
    ):
        partial_json = partial_json[:real_last_comma_index]

    if stack and stack[-1] == "}":
        if colon_index is not None and last_key_index is not None:
            if (
                (
                    colon_index > last_key_index
                    and partial_json[colon_index + 1 :].isspace()
                )
                or colon_index + 1 == len(partial_json)
                or colon_index < last_key_index
            ):
                # The case of '{..., "key":' or '{..., "key": ' or '{..., "key"'
                partial_json = partial_json[: last_comma_index or last_key_index]
        # Now, consider the case of '{"key"'
        if colon_index is None and last_key_index is not None:
            partial_json = partial_json[: last_comma_index or last_key_index]

    while stack:
        if stack[-1] == "}":
            partial_json += "}"
        elif stack[-1] == "]":
            partial_json += "]"
        stack.pop()

    fixed_json_string = ""
    last_pos = 0
    for pos, replace_char in replace_targets:
        if pos <= len(partial_json):
            fixed_json_string += partial_json[last_pos:pos] + replace_char
            last_pos = pos + 1
    if last_pos <= len(partial_json):
        fixed_json_string += partial_json[last_pos:]

    return fixed_json_string


def filter_invalid_triples(triples: List[List[str]]) -> List[List[str]]:
    """
    Filters out invalid and duplicate triples from a list of triples.

    A valid triple meets the following criteria:
    1. It contains exactly three elements.
    2. It is unique within the list (no duplicates in the output).

    The function ensures:
    - Each valid triple is converted to a list of strings.
    - The order of unique, valid triples is preserved.
    - Do not apply any text preprocessing techniques or rules within this function.
    
    Args:
        triples (List[List[str]]): 
            A list of triples (each a list of strings or elements that can be converted to strings).

    Returns:
        List[List[str]]: 
            A list of unique, valid triples, each represented as a list of strings.
    """
    unique_triples = set()
    valid_triples = []

    for triple in triples:
        if len(triple) != 3: continue  # Skip triples that do not have exactly 3 elements

        valid_triple = [str(item) for item in triple]
        if tuple(valid_triple) not in unique_triples:
            unique_triples.add(tuple(valid_triple))
            valid_triples.append(valid_triple)

    return valid_triples


PROMPT_JSON_TEMPLATE = {
    "ner": {
        "type": "object",
        "properties": {
            "named_entities": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "minItems": 0
            }
        },
        "required": ["named_entities"]
    },
    "triples": {
        "type": "object",
        "properties": {
            "triples": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "maxItems": 3,
                    "minItems": 3,
                },
                "minItems": 0
            }
        },
        "required": ["triples"]
    },
    "fact": {
        "type": "object",
        "properties": {
            "fact": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "maxItems": 3,
                    "minItems": 3,
                },
                "minItems": 0
            }
        },
        "required": ["fact"]
    },
    "json": {
        "type": "object",
    },
    "qa_cot": {
        "type": "object",
        "required": ["Thought", "Answer"],
        "properties": {
            "Thought": {
                "type": "string",
                "minLength": 1,
                "maxLength": 2000
            },
            "Answer": {
                "type": "string",
                "minLength": 1,
                "maxLength": 200
            },
        },
    }
}


def num_tokens_by_tiktoken(text: str):
    import tiktoken
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

    return len(enc.encode(text))
