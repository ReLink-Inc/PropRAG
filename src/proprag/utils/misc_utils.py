from argparse import ArgumentTypeError
from dataclasses import dataclass
from hashlib import md5
from typing import Dict, Any, List, Tuple, Literal, Union, Optional, TypeVar, Generic, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from tqdm import tqdm

import numpy as np
import re
import logging
from contextlib import contextmanager


from .typing import Triple, Proposition
from .llm_utils import filter_invalid_triples

logger = logging.getLogger(__name__)

@dataclass
class PropositionRawOutput:
    chunk_id: str
    response: str
    propositions: List[Dict[str, Any]]  # List of proposition objects with text and entities
    metadata: Dict[str, Any] = None

@dataclass
class NerRawOutput:
    chunk_id: str
    response: str
    unique_entities: List[str]
    metadata: Dict[str, Any]


@dataclass
class TripleRawOutput:
    chunk_id: str
    response: str
    triples: List[List[str]]
    metadata: Dict[str, Any]

@dataclass
class LinkingOutput:
    score: np.ndarray
    type: Literal['node', 'dpr']

@dataclass
class QuerySolution:
    question: str
    docs: List[str]
    doc_scores: np.ndarray = None
    answer: str = None
    gold_answers: List[str] = None
    gold_docs: Optional[List[str]] = None


    def to_dict(self):
        return {
            "question": self.question,
            "answer": self.answer,
            "gold_answers": self.gold_answers,
            "docs": self.docs[:5],
            "doc_scores": [round(v, 4) for v in self.doc_scores.tolist()[:5]]  if self.doc_scores is not None else None,
            "gold_docs": self.gold_docs,
        }

T = TypeVar('T')
V = TypeVar('V')

import concurrent.futures
# --- Custom generator mimicking as_completed with refreshing timeout ---
def as_completed_refreshing_timeout(futures_set, timeout):
    """
    Yields futures as they complete, but applies the timeout
    to each wait for the *next* completion.

    Args:
        futures_set: A set (or list/iterable) of Future objects.
        timeout: The maximum time in seconds to wait for the *next*
                 future to complete in each iteration.

    Yields:
        Completed Future objects.

    Raises:
        concurrent.futures.TimeoutError: If no future completes within the
                                         specified timeout during a wait cycle.
    """
    pending = set(futures_set) # Work with a mutable set

    while pending:
        # wait() returns two sets: done and not_done
        # We wait for the *first* future to complete, with a timeout
        done, not_done = concurrent.futures.wait(
            pending,
            timeout=timeout,
            return_when=concurrent.futures.FIRST_COMPLETED
        )

        if not done:
            # If 'done' is empty after the timeout, no future completed in time
            raise concurrent.futures.TimeoutError(
                f"No future completed within the {timeout}s timeout."
            )

        # Yield the completed futures
        for future in done:
            yield future
            pending.remove(future) # Remove from the set of pending futures


class RetryExecutor(Generic[T, V]):
    def __init__(self, 
                 executor: ThreadPoolExecutor,
                 futures: Dict[Future[T], V],
                 submit_fn: Callable[[V], Tuple],
                 desc: str = "Processing",
                 timeout: int = 180):
        self.executor = executor
        self.futures = futures
        self.submit_fn = submit_fn
        self.timeout = timeout
        self.failed_items = set()
        self.desc = desc

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def process(self, process_fn: Callable[[Future[T], V], None]):
        while self.futures:
            try:
                pbar = tqdm(as_completed(self.futures), 
                          total=len(self.futures),
                          desc=self.desc)
                for future in pbar:
                    try:
                        process_fn(future, self.futures[future], pbar)
                    except Exception as e:
                        logger.error(f"Error processing item {self.futures[future]}: {e}")
                        self.failed_items.add(self.futures[future])
            except TimeoutError:
                remaining = {f: v for f, v in self.futures.items() if not f.done()}
                logger.warning(f"Timeout occurred. {len(remaining)} items still pending, will retry...")
                self.failed_items.update(remaining.values())
                
                # Cancel remaining futures
                for future in remaining:
                    future.cancel()
            self.futures = {}
            for item in self.failed_items:
                args, kwargs = self.submit_fn(item)
                future = self.executor.submit(*args, **kwargs)
                # Prepare for next iteration
                self.futures[future] = item
            self.failed_items.clear()

def text_processing(text):
    if isinstance(text, list):
        return [text_processing(t) for t in text]
    if not isinstance(text, str):
        text = str(text)
    return re.sub('[^A-Za-z0-9 ]', ' ', text.lower()).strip()

def reformat_openie_results(corpus_openie_results) -> (Dict[str, NerRawOutput], Dict[str, TripleRawOutput]):

    ner_output_dict = {
        chunk_item['idx']: NerRawOutput(
            chunk_id=chunk_item['idx'],
            response=None,
            metadata={},
            unique_entities=list(np.unique(chunk_item['extracted_entities']))
        )
        for chunk_item in corpus_openie_results
    }
    triple_output_dict = {
        chunk_item['idx']: TripleRawOutput(
            chunk_id=chunk_item['idx'],
            response=None,
            metadata={},
            triples=filter_invalid_triples(triples=chunk_item['extracted_triples'])
        )
        for chunk_item in corpus_openie_results
    }

    return ner_output_dict, triple_output_dict

def extract_entity_nodes(chunk_triples: List[Triple]) -> (List[str], List[List[str]]):
    chunk_triple_entities = []  # a list of lists of unique entities from each chunk's triples
    for triples in chunk_triples:
        triple_entities = set()
        for t in triples:
            if len(t) == 3:
                triple_entities.update([t[0], t[2]])
            else:
                logger.warning(f"During graph construction, invalid triple is found: {t}")
        chunk_triple_entities.append(list(triple_entities))
    graph_nodes = list(np.unique([ent for ents in chunk_triple_entities for ent in ents]))
    return graph_nodes, chunk_triple_entities

def extract_proposition_entities(chunk_propositions: List[List[Dict]]) -> Tuple[List[str], List[List[str]]]:
    """
    Extract all entity nodes from propositions and group them by chunk.
    
    Args:
        chunk_propositions: List of propositions from each chunk
        
    Returns:
        Tuple containing:
        - List of unique entity nodes across all chunks
        - List of lists containing entities for each chunk
    """
    all_entities = []
    chunk_proposition_entities = []  # Similar to chunk_triple_entities
    
    for propositions in chunk_propositions:
        chunk_entities = set()
        for prop in propositions:
            if not "entities" in prop:
                logger.warning("No entities found in proposition: ", prop)
                logger.warning(f"Proposition: {prop}")
                continue
            all_entities.extend(prop["entities"])
            chunk_entities.update(prop["entities"])
        chunk_proposition_entities.append(list(chunk_entities))
    
    return list(np.unique(all_entities)), chunk_proposition_entities

def flatten_facts(chunk_triples: List[Triple]) -> List[Triple]:
    graph_triples = []  # a list of unique relation triple (in tuple) from all chunks
    for triples in chunk_triples:
        graph_triples.extend([tuple(t) for t in triples])
    graph_triples = list(set(graph_triples))
    return graph_triples

def flatten_propositions(chunk_propositions: List[List[Dict]]) -> List[Dict]:
    """
    Flatten a list of lists of propositions into a single list.
    
    Args:
        chunk_propositions: List of propositions from each chunk
        
    Returns:
        Flattened list of all propositions
    """
    all_propositions = []
    for propositions in chunk_propositions:
        all_propositions.extend(propositions)
    
    return all_propositions

def min_max_normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """
    Compute the MD5 hash of the given content string and optionally prepend a prefix.

    Args:
        content (str): The input string to be hashed.
        prefix (str, optional): A string to prepend to the resulting hash. Defaults to an empty string.

    Returns:
        str: A string consisting of the prefix followed by the hexadecimal representation of the MD5 hash.
    """
    # return prefix + md5(content.encode()).hexdigest()
    return prefix + content


def all_values_of_same_length(data: dict) -> bool:
    """
    Return True if all values in 'data' have the same length or data is an empty dict,
    otherwise return False.
    """
    # Get an iterator over the dictionary's values
    value_iter = iter(data.values())

    # Get the length of the first sequence (handle empty dict case safely)
    try:
        first_length = len(next(value_iter))
    except StopIteration:
        # If the dictionary is empty, treat it as all having "the same length"
        return True

    # Check that every remaining sequence has this same length
    return all(len(seq) == first_length for seq in value_iter)


def string_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )
