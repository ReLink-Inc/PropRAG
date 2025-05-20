import os
import sys
from typing import List
import json

from src.proprag.PropRAG import PropRAG
from src.proprag.utils.misc_utils import string_to_bool
from src.proprag.utils.config_utils import BaseConfig

import argparse

import logging
logging.getLogger("httpx").setLevel(logging.WARNING)

# os.environ["LOG_LEVEL"] = "DEBUG"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging

def get_gold_docs(samples: List, dataset_name: str = None) -> List:
    gold_docs = []
    for sample in samples:
        if 'supporting_facts' in sample:  # hotpotqa, 2wikimultihopqa
            gold_title = set([item[0] for item in sample['supporting_facts']])
            gold_title_and_content_list = [item for item in sample['context'] if item[0] in gold_title]
            if dataset_name.startswith('hotpotqa'):
                gold_doc = [item[0] + '\n' + ''.join(item[1]) for item in gold_title_and_content_list]
            else:
                gold_doc = [item[0] + '\n' + ' '.join(item[1]) for item in gold_title_and_content_list]
        elif 'contexts' in sample:
            gold_doc = [item['title'] + '\n' + item['text'] for item in sample['contexts'] if item['is_supporting']]
        else:
            assert 'paragraphs' in sample, "`paragraphs` should be in sample, or consider the setting not to evaluate retrieval"
            gold_paragraphs = []
            for item in sample['paragraphs']:
                if 'is_supporting' in item and item['is_supporting'] is False:
                    continue
                gold_paragraphs.append(item)
            gold_doc = [item['title'] + '\n' + (item['text'] if 'text' in item else item['paragraph_text']) for item in gold_paragraphs]

        gold_doc = list(set(gold_doc))
        gold_docs.append(gold_doc)
    return gold_docs


def get_gold_answers(samples):
    gold_answers = []
    for sample_idx in range(len(samples)):
        gold_ans = None
        sample = samples[sample_idx]

        if 'answer' in sample or 'gold_ans' in sample:
            gold_ans = sample['answer'] if 'answer' in sample else sample['gold_ans']
        elif 'reference' in sample:
            gold_ans = sample['reference']
        elif 'obj' in sample:
            gold_ans = set(
                [sample['obj']] + [sample['possible_answers']] + [sample['o_wiki_title']] + [sample['o_aliases']])
            gold_ans = list(gold_ans)
        assert gold_ans is not None
        if isinstance(gold_ans, str):
            gold_ans = [gold_ans]
        assert isinstance(gold_ans, list)
        gold_ans = set(gold_ans)
        if 'answer_aliases' in sample:
            gold_ans.update(sample['answer_aliases'])

        gold_answers.append(gold_ans)

    return gold_answers

def main():
    parser = argparse.ArgumentParser(description="PropRAG retrieval and QA")
    parser.add_argument('--dataset', type=str, default='musique', help='Dataset name')
    # parser.add_argument('--llm_base_url', type=str, default='https://api.openai.com/v1', help='LLM base URL')
    parser.add_argument('--llm_base_url', type=str, default='https://openrouter.ai/api/v1', help='LLM base URL')
    # parser.add_argument('--llm_name', type=str, default='gpt-4o-mini', help='LLM name')
    parser.add_argument('--llm_name', type=str, default='meta-llama/llama-3.3-70b-instruct', help='LLM name')
    parser.add_argument('--embedding_name', type=str, default='nvidia/NV-Embed-v2', help='embedding model name')
    parser.add_argument('--force_index_from_scratch', type=str, default='false',
                        help='If set to True, will ignore all existing storage files and graph data and will rebuild from scratch.')
    parser.add_argument('--force_openie_from_scratch', type=str, default='false', help='If set to False, will try to first reuse openie results for the corpus if they exist.')
    parser.add_argument('--openie_mode', choices=['online', 'offline'], default='online',
                        help="OpenIE mode, offline denotes using VLLM offline batch mode for indexing, while online denotes")
    parser.add_argument('--use_propositions', action='store_true', 
                        help="Use proposition extraction before entity-relation extraction")

   
    parser.add_argument('--use_beam_search', action='store_true',
                      help="Use graph beam search for retrieval")
    parser.add_argument('--beam_width', type=int, default=4,
                      help="Width of the beam for beam search (number of paths to track)")
    parser.add_argument('--max_path_length', type=int, default=3,
                      help="Max path length for beam search")
    parser.add_argument('--second_stage_filter_k', type=int, default=40,
                      help="Second stage filter k for beam search")
    parser.add_argument('--sim_threshold', type=float, default=0.75,
                      help="Threshold for considering edges as synonyms in beam search (0.0-1.0)")


    args = parser.parse_args()

    dataset_name = args.dataset
    llm_base_url = args.llm_base_url
    llm_name = args.llm_name

    corpus_path = f"reproduce/dataset/{dataset_name}_corpus.json"
    with open(corpus_path, "r") as f:
        corpus = json.load(f)

    docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]

    force_index_from_scratch = string_to_bool(args.force_index_from_scratch)
    force_openie_from_scratch = string_to_bool(args.force_openie_from_scratch)

    # Prepare datasets and evaluation
    samples = json.load(open(f"reproduce/dataset/{dataset_name}.json", "r"))
    all_queries = [s['question'] for s in samples]

    # gold_docs = get_gold_docs(samples, dataset_name)
    if dataset_name in ['lveval', 'narrativeqa_dev_10_doc']:
        gold_docs = None
    else:
        gold_docs = get_gold_docs(samples, dataset_name)
    gold_answers = get_gold_answers(samples)
    # assert len(all_queries) == len(gold_docs) == len(gold_answers), "Length of queries, gold_docs, and gold_answers should be the same."
    assert len(all_queries) == len(gold_answers), "Length of queries and gold_answers should be the same."

    config = BaseConfig(
        llm_base_url=llm_base_url,
        llm_name=llm_name,
        dataset=dataset_name,
        embedding_model_name=args.embedding_name,
        force_index_from_scratch=force_index_from_scratch,  # ignore previously stored index, set it to False if you want to use the previously stored index and embeddings
        force_openie_from_scratch=force_openie_from_scratch,
        retrieval_top_k=200,
        linking_top_k=5,
        max_qa_steps=3,
        qa_top_k=5,
        graph_type="facts_and_sim_passage_node_unidirectional",
        embedding_batch_size=4,
        beam_width=args.beam_width,
        max_path_length=args.max_path_length,
        second_stage_filter_k=args.second_stage_filter_k,
        # max_new_tokens=None,
        corpus_len=len(corpus),
        openie_mode=args.openie_mode,
        use_propositions=args.use_propositions
    )

    logging.basicConfig(level=logging.INFO)

    proprag = PropRAG(global_config=config)

    proprag.index(docs)

    # Retrieval and QA
    proprag.rag_qa(queries=all_queries, gold_docs=gold_docs, gold_answers=gold_answers)
    # proprag.retrieve_grid_search(queries=all_queries, gold_docs=gold_docs)
    # proprag.qa()

import faulthandler
import signal
import sys

# Enable faulthandler
faulthandler.enable()

# Optional: dump tracebacks when you send SIGUSR1 (Linux/macOS only)
faulthandler.register(signal.SIGUSR1)


if __name__ == "__main__":
    main()
