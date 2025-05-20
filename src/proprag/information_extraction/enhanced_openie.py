import json
import re
from dataclasses import dataclass
from typing import Dict, Any, List, TypedDict, Tuple, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from tqdm import tqdm

from ..prompts import PromptTemplateManager
from ..utils.logging_utils import get_logger
from ..utils.llm_utils import fix_broken_generated_json, filter_invalid_triples
from ..utils.misc_utils import TripleRawOutput, NerRawOutput, PropositionRawOutput, text_processing, RetryExecutor
from ..llm.openai_gpt import CacheOpenAI
from .proposition_extraction import PropositionExtractor

logger = get_logger(__name__)


class ChunkInfo(TypedDict):
    num_tokens: int
    content: str
    chunk_order: List[Tuple]
    full_doc_ids: List[str]


@dataclass
class LLMInput:
    chunk_id: str
    input_message: List[Dict]


def _extract_ner_from_response(real_response):
    pattern = r'\{[^{}]*"entities"\s*:\s*\[[^\]]*\][^{}]*\}'
    match = re.search(pattern, real_response, re.DOTALL)
    # eval_string = "{" + match.group() + "}"
    eval_string = match.group()
    # try:
    return eval(eval_string)["entities"]
    # except Exception as e:
    #     logger.warning(f"Trying to fix Error: {e}, real_response: {real_response}")
    #     lines = eval_string.split('\n')
    #     for i, line in enumerate(lines):
    #         if "entities" not in line and line.count('"') == 2:
    #             start_idx = line.find('"')
    #             end_idx = line[start_idx + 1:].find('"') + start_idx + 2
    #             lines[i] = line[:end_idx] + ','
    #     eval_string = '\n'.join(lines)
    #     try:
    #         return eval(eval_string)["entities"]
    #     except Exception as e:
    #         logger.warning(f"Fixing Error failed: {e} \n raw_response: {real_response} \n last eval_string: {eval_string}")
    #         raise e


class EnhancedOpenIE:
    """
    Enhanced version of OpenIE that uses proposition extraction before entity-relation extraction.
    This creates more contextually aware triples by first breaking passages into atomic propositions.
    """
    
    def __init__(self, llm_model: CacheOpenAI):
        # Init prompt template manager
        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})
        self.llm_model = llm_model
        # Initialize proposition extractor
        self.proposition_extractor = PropositionExtractor(llm_model)

    def ner(self, chunk_key: str, passage: str, temperature=0.0, fix_attempt=False, use_cache=True) -> NerRawOutput:
        """
        Extract named entities from a passage.
        
        Args:
            chunk_key: Identifier for the chunk
            passage: The text passage to extract entities from
            
        Returns:
            NerRawOutput object containing the entities and metadata
        """
        # PREPROCESSING
        # ner_input_message = self.prompt_template_manager.render(name='ner', passage=passage)
        ner_input_message = self.prompt_template_manager.render(name='ner_expanded', passage=passage)
        raw_response = ""
        # metadata = {}
        try:
            # LLM INFERENCE
            while len(raw_response) == 0:
                raw_response, metadata, cache_hit = self.llm_model.infer(
                    messages=ner_input_message,
                    temperature=temperature,
                    use_cache=use_cache
                )
                if len(raw_response) == 0:
                    logger.warning(f"Empty response, try again.")
                    use_cache = False
            metadata['cache_hit'] = cache_hit
            if metadata['finish_reason'] == 'length':
                logger.warning("="*80)
                logger.warning(f"LENGTH LIMIT REACHED! Raw response: {raw_response}")
                logger.warning("="*80)
            real_response = raw_response
            if fix_attempt:
                logger.warning(f"Fix attempt with following response: {raw_response}")
            extracted_entities = _extract_ner_from_response(real_response)
            unique_entities = list(dict.fromkeys(extracted_entities))

            if len(unique_entities) == 0:
                logger.warning(f"Number of entities is 0, raw_response: {raw_response}, passage: {passage}")

            if temperature > 0.0:
                logger.info(f"Fixed with following response: {raw_response} at temperature {temperature}")

        except Exception as e:
            logger.warning(f"Error Occurred: {e}\nraw_response: {raw_response}\npassage: {passage}")
            logger.warning(f"Response length: {len(raw_response)}")

            logger.warning(f"Error occurred, initiating JSON fix!")

            json_fix_message = self.prompt_template_manager.render(name='fix_json', json=raw_response)
            raw_response, _, _ = self.llm_model.infer(
                messages=json_fix_message,
                temperature=temperature
            )
            try:
                extracted_entities = _extract_ner_from_response(raw_response)
                unique_entities = list(dict.fromkeys(extracted_entities))

                if len(unique_entities) == 0:
                    logger.warning(f"Number of entities is 0, raw_response: {raw_response}, passage: {passage}")

                logger.info(f"JSON fix successful! {raw_response}")
            except Exception as e:
                logger.warning(f"JSON fix failed!")
                # if temperature < 1:
                #     return self.ner(chunk_key, passage, temperature=temperature + 0.1)
                # # For any other unexpected exceptions, log them and return with the error message
                # logger.info(raw_response)
                metadata.update({'error': str(e)})
                return NerRawOutput(
                    chunk_id=chunk_key,
                    response=raw_response,  # Store the error message in metadata
                    unique_entities=[],
                    metadata=metadata  # Store the error message in metadata
                )

        return NerRawOutput(
            chunk_id=chunk_key,
            response=raw_response,
            unique_entities=unique_entities,
            metadata=metadata
        )


    def check_new_entities(self, triple: List[str], original_entities: List[str]) -> List[str]:
        """
        Check if there are new entities in the triple that are not in the original input entities.
        
        Args:
            triple: A triple (subject, predicate, object)
            original_entities: List of original entities from input
            
        Returns:
            List of new entities found in the triple
        """
        # We only check subject and object, as predicates are usually relationships
        subject, _, obj = triple
        
        # Process original entities using the same text_processing function used in HippoRAG
        normalized_original_entities = [text_processing(e) for e in original_entities]
        
        # Check if subject or object are new entities
        new_entities = []
        
        # Process and check subject
        processed_subject = text_processing(subject)
        if processed_subject not in normalized_original_entities:
            # Check if this is a partial match (could be an entity with additional context)
            if not any(processed_subject == e for e in normalized_original_entities):
                new_entities.append(subject)
        
        # Process and check object
        processed_obj = text_processing(obj)
        if processed_obj not in normalized_original_entities:
            # Check if this is a partial match (could be an entity with additional context)
            if not any(processed_obj == e for e in normalized_original_entities):
                new_entities.append(obj)
                
        return new_entities

    def triple_extraction(self, chunk_key: str, passage: str, propositions: List[Dict[str, Any]]) -> TripleRawOutput:
        """
        Extract triples from all propositions in a passage by combining them into a single text.
        This approach treats all propositions together as a cohesive chunk, similar to how 
        fact embeddings are processed in the original HippoRAG.
        
        Args:
            chunk_key: Identifier for the chunk
            passage: The original passage (for fallback)
            propositions: List of propositions extracted from the passage
            
        Returns:
            TripleRawOutput object containing all triples and metadata
        """
        metadata = {}
        
        # Instead of processing each proposition separately, combine them into a single text
        if propositions:
            # Combine all propositions into a single string
            combined_propositions_text = " ".join([prop["text"] for prop in propositions])
            
            # Collect all unique entities from all propositions
            all_entities = []
            for prop in propositions:
                all_entities.extend(prop["entities"])
            unique_entities = list(dict.fromkeys(all_entities))
            
            # Create message for triple extraction on the combined text
            messages = self.prompt_template_manager.render(
                name='triple_extraction',
                passage=combined_propositions_text,
                named_entity_json=json.dumps({"named_entities": unique_entities})
            )
            
            raw_response = ""
            try:
                # LLM INFERENCE
                raw_response, metadata, cache_hit = self.llm_model.infer(
                    messages=messages,
                    temperature=0.0
                )
                metadata['cache_hit'] = cache_hit
                if metadata['finish_reason'] == 'length':
                    real_response = fix_broken_generated_json(raw_response)
                else:
                    real_response = raw_response
                    
                # Extract triples from the response
                pattern = r'\{[^{}]*"triples"\s*:\s*\[[^\]]*\][^{}]*\}'
                match = re.search(pattern, real_response, re.DOTALL)
                extracted_triples = eval(match.group())["triples"]
                valid_triples = filter_invalid_triples(triples=extracted_triples)
                
                # Check for new entities in the triples and filter them out
                all_triples = []
                filtered_count = 0
                
                for triple in valid_triples:
                    # Find new entities that weren't in the original input
                    new_entities = self.check_new_entities(triple, unique_entities)
                    
                    # If new entities were found, skip this triple
                    if new_entities:
                        filtered_count += 1
                    else:
                        all_triples.append(triple)
                
                # Log a summary if any triples were filtered
                # if filtered_count > 0:
                #     logger.warning(f"[{chunk_key}] Filtered {filtered_count} out of {len(valid_triples)} triples containing new entities")
                
            except Exception as e:
                logger.warning(f"Exception in batch triple extraction for chunk {chunk_key}: {e}")
                metadata.update({'error': str(e)})
                all_triples = []
        else:
            all_triples = []
        
        # Process all triples with text_processing before returning
        processed_triples = [[text_processing(t) for t in triple] for triple in all_triples]
        
        # Success
        return TripleRawOutput(
            chunk_id=chunk_key,
            response=raw_response, 
            metadata=metadata,
            triples=processed_triples
        )

    def openie(self, chunk_key: str, passage: str, skip_triples=True) -> Dict[str, Any]:
        """
        Full OpenIE process with proposition extraction.
        
        Args:
            chunk_key: Identifier for the chunk
            passage: The text passage to process
            skip_triples: If True, skip triple extraction and use only propositions
            
        Returns:
            Dictionary with NER and proposition extraction results (and optionally triples)
        """
        # Step 1: Extract named entities
        ner_output = self.ner(chunk_key=chunk_key, passage=passage)
        
        # Step 2: Extract propositions using the named entities
        proposition_output = self.proposition_extractor.extract_propositions(
            chunk_key=chunk_key, 
            passage=passage,
            named_entities=ner_output.unique_entities
        )
        
        result = {"ner": ner_output, "propositions": proposition_output}
        
        # Step 3: Extract triples from propositions only if not skipping triples
        if not skip_triples:
            triple_output = self.triple_extraction(
                chunk_key=chunk_key, 
                passage=passage, 
                propositions=proposition_output.propositions
            )
            result["triplets"] = triple_output
            
        return result

    def batch_openie(self, chunks: Dict[str, ChunkInfo], skip_triples=True) -> Tuple[Dict[str, NerRawOutput], Optional[Dict[str, TripleRawOutput]], Dict[str, PropositionRawOutput]]:
        """
        Conduct batch OpenIE with proposition extraction.
        
        Args:
            chunks: Dictionary of chunk IDs to chunk info
            skip_triples: If True, skip triple extraction and use only propositions
            
        Returns:
            Tuple of dictionaries with NER, (optionally) triple, and proposition extraction results
        """
        # Extract passages from the provided chunks
        chunk_passages = {chunk_key: chunk["content"] for chunk_key, chunk in chunks.items()}
        
        # Step 1: Extract named entities in batch
        ner_results_list = []
        
        # Process NER in parallel
        with ThreadPoolExecutor(max_workers=300) as executor:
            # Create initial futures
            ner_futures = {executor.submit(self.ner, chunk_key, passage): chunk_key 
                          for chunk_key, passage in chunk_passages.items()}
            
            # Process with automatic retry
            with RetryExecutor(executor, ner_futures, 
                             lambda chunk_key: ((self.ner, chunk_key, chunk_passages[chunk_key]), {}),
                             desc="Extracting named entities") as retry_exec:
                retry_exec.process(lambda future, chunk_key, pbar: ner_results_list.append(future.result()))
        
        # Convert NER list to dictionary
        ner_results_dict = {res.chunk_id: res for res in ner_results_list}
        
        # Step 2: Extract propositions in batch using named entities
        # Create a dictionary mapping chunk IDs to named entities
        named_entities_dict = {chunk_key: ner_output.unique_entities for chunk_key, ner_output in ner_results_dict.items()}
        # all_entities = [ner_output.unique_entities for chunk_key, ner_output in ner_results_dict.items()]
        # print(len(set().union(*all_entities)))
        
        # Extract propositions with the named entities
        proposition_results_dict = self.proposition_extractor.batch_extract_propositions(chunks, named_entities_dict)
        
        # If skipping triples, return early
        if skip_triples:
            return ner_results_dict, None, proposition_results_dict
            
        # Step 3: Process propositions for triples (now treating all propositions from a chunk as a group)
        triple_results_list = []
        total_prompt_tokens, total_completion_tokens, num_cache_hit = 0, 0, 0
        
        with ThreadPoolExecutor(max_workers=300) as executor:
            # Create triple extraction futures for each chunk
            triple_futures = {
                executor.submit(
                    self.triple_extraction, 
                    chunk_key, 
                    chunk_passages[chunk_key], 
                    proposition_results_dict[chunk_key].propositions
                ): chunk_key
                for chunk_key in chunk_passages.keys()
            }
            
            # Process with automatic retry
            with RetryExecutor(executor, triple_futures,
                             lambda chunk_key: (
                                 self.triple_extraction,
                                 chunk_key,
                                 chunk_passages[chunk_key],
                                 proposition_results_dict[chunk_key].propositions
                             ),
                             desc="Extracting triples from proposition groups") as retry_exec:
                def process_triple(future, chunk_key, pbar):
                    triple_result = future.result()
                    triple_results_list.append(triple_result)
                    metadata = triple_result.metadata
                    nonlocal total_prompt_tokens, total_completion_tokens, num_cache_hit
                    total_prompt_tokens += metadata.get('prompt_tokens', 0)
                    total_completion_tokens += metadata.get('completion_tokens', 0)
                    if metadata.get('cache_hit'):
                        num_cache_hit += 1
                
                retry_exec.process(process_triple)
        
        # Convert triple list to dictionary
        triple_results_dict = {res.chunk_id: res for res in triple_results_list}
        
        # Apply text_processing to all entity nodes and update proposition mappings
        processed_ner_results_dict = {}
        processed_proposition_results_dict = {}
        
        for chunk_key, ner_output in ner_results_dict.items():
            # # Process all entities
            # processed_entities = [text_processing(entity) for entity in ner_output.unique_entities]
            
            # Create new NER output with processed entities
            processed_ner_output = NerRawOutput(
                chunk_id=ner_output.chunk_id,
                response=ner_output.response,
                unique_entities=ner_output.unique_entities,
                metadata=ner_output.metadata
            )
            processed_ner_results_dict[chunk_key] = processed_ner_output
            
            # Process propositions and update their entities
            prop_output = proposition_results_dict[chunk_key]
            processed_props = []
            
            for prop in prop_output.propositions:
                # # Process each entity in the proposition
                # processed_prop_entities = [text_processing(entity) for entity in prop["entities"]]
                
                # Create new proposition with processed entities
                processed_prop = {
                    "text": prop["text"],  # Keep original text
                    "entities": prop["entities"]
                }
                processed_props.append(processed_prop)
            
            # Create new proposition output with processed propositions
            processed_prop_output = PropositionRawOutput(
                chunk_id=prop_output.chunk_id,
                response=prop_output.response,
                propositions=processed_props,
                metadata=prop_output.metadata
            )
            processed_proposition_results_dict[chunk_key] = processed_prop_output
        
        # Create a verification function to check if all triple subjects and objects exist in entities
        def verify_triple_entities_consistency():
            # For debug purposes - check if all triple subjects and objects belong to the entity set
            for chunk_key, triple_output in triple_results_dict.items():
                if chunk_key not in processed_ner_results_dict:
                    logger.warning(f"Chunk {chunk_key} has triples but no NER output")
                    continue
                    
                # Get all processed entities for this chunk
                all_chunk_entities = set(processed_ner_results_dict[chunk_key].unique_entities)
                
                # Check each triple's subject and object
                for triple in triple_output.triples:
                        
                    subject, _, obj = triple
                    
                    # Check if subject exists in entity set
                    if subject not in all_chunk_entities:
                        logger.warning(f"Triple subject not in entity set: '{subject}' in chunk {chunk_key}")
                    
                    # Check if object exists in entity set
                    if obj not in all_chunk_entities:
                        logger.warning(f"Triple object not in entity set: '{obj}' in chunk {chunk_key}")
        
        verify_triple_entities_consistency()
            
        logger.info(f"Applied text_processing to entities and propositions for {len(processed_ner_results_dict)} chunks")
        
        return processed_ner_results_dict, triple_results_dict, processed_proposition_results_dict