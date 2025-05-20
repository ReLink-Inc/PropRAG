from typing import Dict, List, Union, Any, Optional
import os
import json


from ..prompts.prompt_template_manager import PromptTemplateManager
from .logging_utils import get_logger
from ..llm.openai_gpt import CacheOpenAI

logger = get_logger(__name__)


def load_qa_dataset(dataset_name: str):
    """
    Load QA dataset for evaluation.
    
    Args:
        dataset_name: Name of the dataset ('musique', 'hotpotqa', '2wikimultihopqa', or 'sample')
        
    Returns:
        Tuple of (questions, answers, gold_documents, contexts)
        
        Where:
        - gold_documents is a list of lists matching the format of get_gold_docs in main.py
          Each document is formatted as title + '\n' + content
        - answers is a list of sets matching the format of get_gold_answers in main.py
          Each set contains all possible answer strings including aliases
    """
    dataset_path = os.path.join("reproduce", "dataset", f"{dataset_name}.json")
    corpus_path = os.path.join("reproduce", "dataset", f"{dataset_name}_corpus.json")
    
    logger.info(f"Loading dataset from {dataset_path}")
    logger.info(f"Loading corpus from {corpus_path}")
    
    # Load the dataset
    try:
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        logger.error(f"Dataset file not found: {dataset_path}")
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    # Load the corpus
    try:
        with open(corpus_path, 'r') as f:
            corpus = json.load(f)
    except FileNotFoundError:
        logger.error(f"Corpus file not found: {corpus_path}")
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    
    # Extract questions, answers, and gold documents using the same format as main.py
    questions = []
    gold_documents = []
    gold_answers = []
    contexts = []
    
    # Process corpus into usable format based on structure
    corpus_dict = {}
    
    # Detect corpus format
    if isinstance(corpus, list) and len(corpus) > 0 and isinstance(corpus[0], dict) and "title" in corpus[0]:
        # Corpus is a list of documents with title and text fields
        for doc in corpus:
            corpus_dict[doc["title"]] = doc["text"]
    elif isinstance(corpus, dict):
        # Corpus is already a dict
        corpus_dict = corpus
    
    logger.info(f"Processed corpus with {len(corpus_dict)} documents")
    
    for item in dataset:
        # Extract question
        questions.append(item.get("question", ""))
        
        # Extract gold documents matching get_gold_docs format
        gold_docs = []
        
        if 'supporting_facts' in item:  # hotpotqa, 2wikimultihopqa
            gold_title = set([item[0] for item in item['supporting_facts']])
            gold_title_and_content_list = [item for item in item['context'] if item[0] in gold_title]
            if dataset_name.startswith('hotpotqa'):
                gold_docs = [item[0] + '\n' + ''.join(item[1]) for item in gold_title_and_content_list]
            else:
                gold_docs = [item[0] + '\n' + ' '.join(item[1]) for item in gold_title_and_content_list]
        elif 'contexts' in item:
            gold_docs = [item['title'] + '\n' + item['text'] for item in item['contexts'] if item['is_supporting']]
        else:
            if 'paragraphs' in item:
                gold_paragraphs = []
                for paragraph in item['paragraphs']:
                    if 'is_supporting' in paragraph and paragraph['is_supporting'] is False:
                        continue
                    gold_paragraphs.append(paragraph)
                gold_docs = [
                    paragraph['title'] + '\n' + (
                        paragraph['text'] if 'text' in paragraph else paragraph['paragraph_text']
                    ) for paragraph in gold_paragraphs
                ]
        
        # De-duplicate gold documents
        gold_docs = list(set(gold_docs))
        gold_documents.append(gold_docs)
        
        # Extract answers matching get_gold_answers format
        gold_ans = None
        if 'answer' in item or 'gold_ans' in item:
            gold_ans = item['answer'] if 'answer' in item else item['gold_ans']
        elif 'reference' in item:
            gold_ans = item['reference']
        elif 'obj' in item:
            gold_ans = set(
                [item['obj']] + [item['possible_answers']] + [item['o_wiki_title']] + [item['o_aliases']]
            )
            gold_ans = list(gold_ans)
        
        if gold_ans is None:
            gold_ans = ""
            
        if isinstance(gold_ans, str):
            gold_ans = [gold_ans]
            
        assert isinstance(gold_ans, list)
        gold_ans = set(gold_ans)
        
        if 'answer_aliases' in item:
            gold_ans.update(item['answer_aliases'])
            
        gold_answers.append(gold_ans)
        
        # Add context if available
        if "context" in item:
            contexts.append(item["context"])
        else:
            contexts.append(None)
    
    # Ensure we have non-empty gold documents
    num_questions_with_gold = sum(1 for docs in gold_documents if len(docs) > 0)
    logger.info(f"Loaded {len(questions)} questions from {dataset_name}, {num_questions_with_gold} with gold documents")
    
    return questions, gold_answers, gold_documents, contexts



def merge_elements_with_same_first_line(elements, prefix='Wikipedia Title: '):
    merged_dict = {}

    # Iterate through each element in the list
    for element in elements:
        # Split the element into lines and get the first line
        lines = element.split('\n')
        first_line = lines[0]

        # Check if the first line is already a key in the dictionary
        if first_line in merged_dict:
            # Append the current element to the existing value
            merged_dict[first_line] += "\n" + element.split(first_line, 1)[1].strip('\n')
        else:
            # Add the current element as a new entry in the dictionary
            merged_dict[first_line] = prefix + element

    # Extract the merged elements from the dictionary
    merged_elements = list(merged_dict.values())
    return merged_elements


def reason_step(dataset, prompt_template_manager: PromptTemplateManager, query: str, passages: list, thoughts: list, llm_client: CacheOpenAI):
    """
    Given few-shot samples, query, previous retrieved passages, and previous thoughts, generate the next thought with OpenAI models. The generated thought is used for further retrieval step.
    :return: next thought
    """

    prompt_user = ''
    if dataset in ['hotpotqa', 'hotpotqa_train']:
        passages = merge_elements_with_same_first_line(passages)
    for passage in passages:
        prompt_user += f'{passage}\n\n'
    prompt_user += f'Question: {query}\nThought:' + ' '.join(thoughts)

    
    messages = prompt_template_manager.render(name=f'ircot_{dataset}', prompt_user=prompt_user)

    try:
        response_message, metadata = llm_client.infer(messages=messages)
        response_content = response_message[0]["content"]
    except Exception as e:
        logger.exception("An exception occurred while calling LLM for QA!")
        return ''
    
    return response_content