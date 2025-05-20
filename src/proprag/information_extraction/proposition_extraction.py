import json
import re
from dataclasses import dataclass
from typing import Dict, Any, List, TypedDict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from ..prompts import PromptTemplateManager
from ..utils.logging_utils import get_logger
from ..utils.llm_utils import fix_broken_generated_json
from ..utils.misc_utils import PropositionRawOutput, RetryExecutor

import re
import ast  # For safe evaluation of entities list structure
import json # For loading JSON and dumping fixed structures
import logging

# Configure logging for better visibility (optional)
# logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
# logging.disable(logging.CRITICAL) # Uncomment to disable logging completely

# Regex to find the structure, capturing text and entities content
# Using re.DOTALL flag later to handle multi-line text fields
REGEX_PATTERN = r'{\s*"text":\s*"(?P<text>.*?)"\s*,\s*"entities":\s*\[(?P<entities_content>.*?)]\s*}'

def fix_and_format_json_segment(original_segment: str, text_content: str, entities_content: str) -> str:
    """
    Attempts to fix unescaped quotes in text_content based on entities_content.
    Returns a valid, formatted JSON string for the segment if successful,
    otherwise returns the original_segment.
    """
    logger.info(f"Attempting fix for segment: {original_segment}")

    # --- 1. Safely evaluate entities structure ---
    try:
        # Construct a valid list string for ast.literal_eval
        entities_str_to_eval = f"[{entities_content.strip()}]" if entities_content.strip() else "[]"
        entities_list = ast.literal_eval(entities_str_to_eval)
        if not isinstance(entities_list, list) or not all(isinstance(e, str) for e in entities_list):
             raise ValueError("Evaluated entities is not a list of strings")
        # logging.info(f"  Successfully evaluated entities structure: {entities_list}")
    except (SyntaxError, ValueError, TypeError) as e:
        logger.warning(f"  Could not evaluate entities structure: {e}. Cannot perform fix.")
        return original_segment # Return original if entities structure is bad

    # --- 2. Store entity texts (no parsing needed now) ---
    entity_texts = set(entities_list) # Directly use the strings

    # --- 3. Find indices of *unescaped* double quotes in text_content ---
    unescaped_quote_indices = []
    i = 0
    while i < len(text_content):
        if text_content[i] == '"':
            # Check if it's escaped (look behind)
            if i > 0 and text_content[i-1] == '\\':
                # It's escaped, do nothing special
                pass
            else:
                # It's unescaped
                unescaped_quote_indices.append(i)
                # logging.debug(f"  Found unescaped quote at index {i}") # Use debug level
        i += 1 # Always advance index

    if not unescaped_quote_indices:
        # logging.info("  No unescaped quotes found in text. Re-formatting as JSON.")
        # Even if no quotes to fix, re-format to ensure valid JSON output
        try:
            output_dict = {"text": text_content, "entities": entities_list}
            return json.dumps(output_dict, ensure_ascii=False)
        except Exception as json_e:
             logger.error(f"  Failed to format segment as JSON even without quote fixing: {json_e}. Returning original.")
             return original_segment
    # logging.info(f"  Found {unescaped_quote_indices} to fix.")

    # --- 4. Apply fixing logic ---
    replacements_made = {} # Store index -> replacement char
    entity_lens = [len(entity) for entity in entity_texts]

    for uq_index in unescaped_quote_indices:
        is_entity_boundary = False
        for entity_len in entity_lens:
            # print(f"Checking quote at index {uq_index}, entity_end_index {uq_index+1+entity_len}: {text_content[uq_index+1:min(uq_index+1+entity_len, len(text_content))]}, entity_start_index {uq_index-entity_len-1}: {text_content[max(0, uq_index-entity_len-1):uq_index]}")
            # Check if the unescaped quote is EXACTLY at the start or end boundary
            if uq_index+1+entity_len <= len(text_content) and text_content[uq_index+1:uq_index+1+entity_len] in entity_texts:
                entity_end_index = uq_index+1+entity_len
                entity = text_content[uq_index+1:uq_index+1+entity_len]
                # logging.info(f"  Unescaped quote at index {uq_index} matches START of entity '{entity}' (ends at {entity_end_index})")
                # Check the char at the OTHER end (the entity's end)
                if 0 <= entity_end_index < len(text_content):
                    other_end_char = text_content[entity_end_index]
                    if other_end_char == "'":
                        # logging.info(f"    Entity '{entity}': Other end is single quote. Replacing quote at {uq_index} with '")
                        replacements_made[uq_index] = "'"
                        is_entity_boundary = True
                    elif other_end_char == '"': # Escaped or unescaped double quote
                        break

            elif uq_index - entity_len - 1 >= 0 and text_content[uq_index - entity_len:uq_index] in entity_texts:
                entity_start_index = uq_index - entity_len - 1
                entity = text_content[uq_index - entity_len:uq_index]
                # logging.info(f"  Unescaped quote at index {uq_index} matches END of entity '{entity}' (starts at {entity_start_index})")
                # Check the char at the OTHER end (the entity's start)
                if 0 <= entity_start_index < len(text_content):
                    other_end_char = text_content[entity_start_index]
                    if other_end_char == "'":
                        # logging.info(f"    Entity '{entity}': Other end is single quote. Replacing quote at {uq_index} with '")
                        replacements_made[uq_index] = "'"
                        is_entity_boundary = True
                    elif other_end_char == '"': # Escaped or unescaped double quote
                        break

    # Rebuild the string using the replacements map
    new_text_parts = []
    last_index = 0
    for index in sorted(replacements_made.keys()):
        new_text_parts.append(text_content[last_index:index])
        new_text_parts.append(replacements_made[index])
        last_index = index + 1 # Move past the original quote position
    new_text_parts.append(text_content[last_index:]) # Add the rest of the string

    fixed_text_content = "".join(new_text_parts)
    # logging.info(f"  Fixed text content: {fixed_text_content}")

    # --- 6. Reconstruct the segment as valid JSON ---
    try:
        output_dict = {
            "text": fixed_text_content,
            "entities": entities_list # Use the already validated list
        }
        # Produce compact JSON output, ensure_ascii=False handles unicode correctly
        reconstructed_segment = json.dumps(output_dict, ensure_ascii=False)
        # logging.info(f"  Successfully fixed and reconstructed JSON segment.")
        return reconstructed_segment
    except Exception as json_e:
        logger.error(f"  Failed to reconstruct segment as valid JSON after fixing: {json_e}. Returning original.")
        logger.error(f"  Original Text Content: {text_content}")
        logger.error(f"  Fixed Text Content Attempted: {fixed_text_content}")
        logger.error(f"  Entities List: {entities_list}")
        return original_segment # Fallback


def fix_large_json_text(large_text: str) -> str:
    """
    Finds all {"text":..., "entities":...} structures in large_text.
    Attempts to fix segments that are not valid JSON.
    Returns the full text with fixed segments replaced.
    """
    processed_parts = []
    last_end = 0
    regex = re.compile(REGEX_PATTERN, flags=re.DOTALL)

    for match in regex.finditer(large_text):
        match_start, match_end = match.span()
        original_segment = match.group(0)
        text_content = match.group('text')
        entities_content = match.group('entities_content')

        # Add the text *before* this match
        processed_parts.append(large_text[last_end:match_start])

        # --- Check if the original segment is valid JSON ---
        is_valid_json = False
        try:
            # Attempt to load the original segment directly
            json.loads(original_segment)
            is_valid_json = True
            # logger.debug(f"Segment at {match_start} is already valid JSON.")
            processed_parts.append(original_segment) # Append original if valid
        except json.JSONDecodeError as e:
            logger.warning(f"Segment at {match_start} is NOT valid JSON: {e}. Attempting fix...")
            # --- If not valid, attempt to fix it ---
            fixed_segment = fix_and_format_json_segment(
                original_segment, text_content, entities_content
            )
            processed_parts.append(fixed_segment) # Append fixed or original (if fix failed)
        except Exception as e:
             logger.error(f"Unexpected error checking/fixing segment at {match_start}: {e}. Keeping original.")
             processed_parts.append(original_segment) # Append original on unexpected error


        last_end = match_end # Update the end position for the next iteration

    # Add any remaining text after the last match
    processed_parts.append(large_text[last_end:])

    # logging.info("Text processing finished.")
    return "".join(processed_parts)

def maximal_parsable_json(partial_json):
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

logger = get_logger(__name__)


@dataclass
class Proposition:
    """Class for representing a proposition extracted from text."""
    text: str
    entities: List[str]


class PropositionExtractor:
    """
    Class to extract propositions from passages before entity-relation extraction.
    Each proposition represents a fully contextualized unit of meaning.
    """
    
    def __init__(self, llm_model):
        # Init prompt template manager
        self.prompt_template_manager = PromptTemplateManager(
            role_mapping={"system": "system", "user": "user", "assistant": "assistant"}
        )
        self.llm_model = llm_model
    
    def extract_propositions(self, chunk_key: str, passage: str, named_entities: Optional[List[str]] = None, temperature=0.0, use_cache=True) -> PropositionRawOutput:
        """
        Extract propositions from a passage.
        
        Args:
            chunk_key: Identifier for the chunk
            passage: The text passage to extract propositions from
            named_entities: Optional list of pre-extracted named entities to use
            
        Returns:
            PropositionRawOutput object containing the propositions and metadata
        """
        # Create the prompt for proposition extraction
        if named_entities:
            # Use the new prompt template with named entities
            proposition_input_message = self.prompt_template_manager.render(
                name='proposition_extraction', 
                passage=passage,
                named_entities=json.dumps(named_entities)
            )
        else:
            # Use the original prompt template without named entities
            proposition_input_message = self.prompt_template_manager.render(
                name='proposition_extraction', 
                passage=passage,
                named_entities="[]"  # Empty list as fallback
            )
        
        raw_response = ""
        metadata = {}
        
        try:
            # LLM INFERENCE
            raw_response, metadata, cache_hit = self.llm_model.infer(
                messages=proposition_input_message,
                temperature=temperature,
                use_cache=use_cache
            )
            metadata['cache_hit'] = cache_hit
            
            if metadata['finish_reason'] == 'length':
                logger.warning("="*80)
                logger.warning(f"LENGTH LIMIT REACHED! Raw response: {raw_response}")
                logger.warning("="*80)
            real_response = raw_response
                
            # Extract propositions from the response
            extracted_data = self._extract_propositions_from_response(real_response)
            propositions = extracted_data["propositions"]
        except ValueError as e:
            logger.warning(e)
            logger.warning(f"JSON parsing error! Try to fix JSON: {raw_response}")
            fix_json = True
            use_cache = True
            while fix_json:
                json_fix_message = self.prompt_template_manager.render(name='fix_json', json=raw_response)
                raw_response, _, _ = self.llm_model.infer(
                    messages=json_fix_message,
                    temperature=temperature,
                    use_cache=use_cache
                )
                try:
                    extracted_data = self._extract_propositions_from_response(raw_response)
                    propositions = extracted_data["propositions"]
                    fix_json = False
                    logger.info(f"JSON fix successful! {raw_response}")
                except Exception as e:
                    logger.warning(f"JSON fix error for chunk {chunk_key}: {e}")
                    logger.warning(f"Raw response: {raw_response}")
                    logger.warning(f"Try again with use_cache = False!")
                    use_cache = False
                    # metadata.update({'error': str(e)})
                    # return PropositionRawOutput(
                    #     chunk_id=chunk_key,
                    #     response=raw_response,
                    #     propositions=[],
                    #     metadata=metadata
                    # )
        except AttributeError as e:
            logger.warning(e)
            return self.extract_propositions(chunk_key, passage, named_entities, temperature=temperature, use_cache=False)
        except AssertionError as e:
            logger.warning(f"Entities and text fields do not match, try to regenerate it: {raw_response}")
            return self.extract_propositions(chunk_key, passage, named_entities, temperature=temperature, use_cache=False)
        except Exception as e:
            logger.warning(f"Unknown error for chunk {chunk_key}: {e}")
            logger.warning(f"Raw response: {raw_response}")
            metadata.update({'error': str(e)})
            return PropositionRawOutput(
                chunk_id=chunk_key,
                response=raw_response,
                propositions=[],
                metadata=metadata
            )

        # s1 = set(named_entities)
        # s2 = set().union(*[set(prop['entities']) for prop in propositions])
        # missing_entities_count = len(s1 - s2)
        # import threading
        # # Thread-safe logging of statistics to file
        # with threading.Lock():
        #     try:
        #         with open('proposition_stats.txt', 'a') as f:
        #             f.write(f"{missing_entities_count}\n")
        #     except Exception as e:
        #         logger.warning(f"Failed to write proposition stats: {e}")
            
        return PropositionRawOutput(
            chunk_id=chunk_key,
            response=raw_response,
            propositions=propositions,
            metadata=metadata
        )
    
    def _extract_propositions_from_response(self, response: str) -> Dict:
        """
        Extract propositions from the LLM response.
        
        Args:
            response: The raw response from the LLM
            
        Returns:
            Dictionary containing the extracted propositions
        """
        # # First, try to use the advanced JSON parsing function
        # try:
        #     fixed_json = maximal_parsable_json(response)
        #     if fixed_json:
        #         result = json.loads(fixed_json)
        #         if "propositions" in result:
        #             return result
        # except Exception as e:
        #     logger.debug(f"Advanced JSON parsing failed: {e}")
            
        # Fallback to regex-based extraction
        pattern = r'\{[^{}]*"propositions"\s*:'

        proposition_pattern = r'\{\s*"text":\s*"(?:\\.|[^\\"])*"\s*,\s*"entities":\s*\[(?:\s*"(?:\\.|[^\\"])*"\s*(?:,\s*"(?:\\.|[^\\"])*"\s*)*)?\s*\]\s*\}'

        match = re.search(pattern, response, re.DOTALL)
        if not match:
            raise AttributeError(f"JSON response is invalid: {response}")

        start_idx = match.span()[0]
        curly_braces_count = 0
        
        in_quote = False
        escape = False
        idx = start_idx
        response_len = len(response)
        while idx < response_len:
            char = response[idx]
            if escape:
                if char == 'u': # unicode
                    idx += 5
                else:
                    idx += 1
                escape = False
                continue
            if char == '{' and not in_quote:
                curly_braces_count += 1
            elif char == '}' and not in_quote:
                curly_braces_count -= 1
            elif char == '"':
                in_quote = not in_quote
            elif char == '\\':
                escape = True
            if curly_braces_count == 0:
                break
            idx += 1
        if curly_braces_count > 0: # Incomplete Suspect
            if response.count('{') != response.count('}'): # Likely a mismatched quote issue
                raise AttributeError(f"JSON response is incomplete: {response}")
            else: # revert idx to the last curly brace
                idx = response.rfind('}', 0, idx)
        json_str = response[start_idx:idx+1]
        
        # if match:
        #     try:
        #         return json.loads(match.group())
        #     except json.JSONDecodeError:
        #         # Attempt to fix the JSON
        #         try:
        #             fixed_match = match.group().replace("'", '"')
        #             return json.loads(fixed_match)
        #         except json.JSONDecodeError:
        #             # If JSON parsing still fails, try a manual approach
        #             result = self._extract_propositions_manually(response)
        #             if result["propositions"]:
        #                 return result

        try:
            loaded_json = json.loads(json_str)
        except json.JSONDecodeError:
            json_str = fix_large_json_text(json_str)
            loaded_json = json.loads(json_str)
            logger.info(f"JSON fix successful!")
        # Fix the JSON as it is not complete
        for prop in loaded_json["propositions"]:
            assert "text" in prop and "entities" in prop
            # Regenerate using higher temperature
        return loaded_json
    
    def _extract_propositions_manually(self, response: str) -> Dict:
        """
        Manually extract propositions from text when JSON parsing fails.
        
        Args:
            response: The raw response text
            
        Returns:
            A dictionary with a "propositions" key containing extracted propositions
        """
        propositions = []
        
        # Look for text/entities patterns in the response
        text_pattern = r'"text":\s*"([^"]+)"'
        entities_pattern = r'"entities":\s*\[(.*?)\]'
        
        text_matches = re.findall(text_pattern, response.replace("'", '"'), re.DOTALL)
        entities_matches = re.findall(entities_pattern, response.replace("'", '"'), re.DOTALL)
        assert len(text_matches) == len(entities_matches)
        # If we have both text and entities, pair them up
        if text_matches and entities_matches and len(text_matches) == len(entities_matches):
            for text, entities_str in zip(text_matches, entities_matches):
                # Extract entity strings from the entities array
                entity_list = []
                for entity_match in re.finditer(r'"([^"]+)"', entities_str):
                    entity_list.append(entity_match.group(1))
                
                propositions.append({
                    "text": text, 
                    "entities": entity_list
                })
        
        return {"propositions": propositions}
    
    def batch_extract_propositions(self, chunks: Dict[str, Dict], named_entities_dict: Optional[Dict[str, List[str]]] = None) -> Dict[str, PropositionRawOutput]:
        """
        Extract propositions from multiple chunks in parallel.
        
        Args:
            chunks: Dictionary of chunk IDs to chunk info
            named_entities_dict: Optional dictionary mapping chunk IDs to pre-extracted named entities
            
        Returns:
            Dictionary of chunk IDs to proposition extraction results
        """
        # Extract passages from the provided chunks
        chunk_passages = {chunk_key: chunk["content"] for chunk_key, chunk in chunks.items()}
        
        proposition_results_list = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        num_cache_hit = 0
        
        with ThreadPoolExecutor(max_workers=300) as executor:
            # Create proposition extraction futures for each chunk
            proposition_futures = {}
            for chunk_key, passage in chunk_passages.items():
                # Get named entities for this chunk if available
                named_entities = named_entities_dict.get(chunk_key, None) if named_entities_dict else None
                # print(f"Named entities for chunk {chunk_key}: {named_entities}")
                # Submit the task
                future = executor.submit(self.extract_propositions, chunk_key, passage, named_entities)
                proposition_futures[future] = chunk_key

            with RetryExecutor(executor, proposition_futures,
                             lambda chunk_key: ((self.extract_propositions, chunk_key, chunk_passages[chunk_key], named_entities_dict.get(chunk_key, None)), {}),
                             desc="Extracting propositions") as retry_exec:
                def process_proposition(future, chunk_key, pbar):
                    result = future.result()
                    proposition_results_list.append(result)
                    metadata = result.metadata
                    nonlocal total_prompt_tokens, total_completion_tokens, num_cache_hit
                    total_prompt_tokens += metadata.get('prompt_tokens', 0)
                    total_completion_tokens += metadata.get('completion_tokens', 0)
                    if metadata.get('cache_hit'):
                        num_cache_hit += 1
                    pbar.set_postfix({
                        'total_prompt_tokens': total_prompt_tokens,
                        'total_completion_tokens': total_completion_tokens,
                        'num_cache_hit': num_cache_hit
                    })
                retry_exec.process(process_proposition)
            
        
        # Convert list of results to dictionary keyed by chunk ID
        proposition_results_dict = {res.chunk_id: res for res in proposition_results_list}
        
        return proposition_results_dict