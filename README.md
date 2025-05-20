# PropRAG: Guiding Retrieval with Beam Search over Proposition Paths

**PropRAG** is a novel Retrieval-Augmented Generation (RAG) framework designed to enhance multi-hop reasoning in Large Language Models (LLMs). It achieves this by:

1.  Leveraging contextually rich **propositions** as fundamental knowledge units, extracted offline using an LLM.
2.  Employing an efficient, **LLM-free online beam search algorithm over proposition paths** to explicitly discover multi-step reasoning chains.

This approach allows PropRAG to improve evidence retrieval for complex queries while avoiding the latency and cost associated with online LLM calls during the search process. PropRAG has demonstrated state-of-the-art zero-shot performance on challenging multi-hop QA benchmarks such as MuSiQue, HotpotQA, and 2WikiMultihopQA.

**Note:** This code repository is adapted from the original HippoRAG repository: [https://github.com/OSU-NLP-Group/HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG). We thank the authors for making their code available.

---

## ⚠️ Current Status & Known Issues

Please be aware that this repository is an active research codebase.
*   There might be bugs or some components may not be fully functional. Fixes and improvements are in progress.
*   We welcome contributions and bug reports! Please feel free to open an issue.

---

## Setup

### 1. Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 2. API Keys
PropRAG utilizes LLMs for offline proposition and entity extraction, and potentially as a reader for end-to-end QA, Currently only OpenRouter API is supported.

*   **OpenRouter API Key:**
    Create a file named `openrouter_api_key.txt` in the root directory of this repository and paste your OpenRouter API key into it.

### 3. Supported LLMs for Extraction/Reading
*   **Currently Supported for Extraction (as used in the paper):** Llama-3.3-70B-Instruct (via Nebius AI Studio or compatible OpenRouter endpoint).
*   **Coming Soon:** Support for GPT-4o-mini and other models via OpenRouter.

### 4. Pre-extracted Data (Optional but Recommended for Reproducibility)
To reproduce the results from the paper without re-running the potentially time-consuming offline proposition and entity extraction steps, you can use the pre-extracted data provided.

The files named `openie_results_ner_meta-llama_llama-3.3-70b-instruct.json` located in the `outputs/<dataset_name>/` directories contain the propositions and entities extracted using Llama-3.3-70B-Instruct for each respective dataset used in our paper.

Ensure these files are correctly placed if you intend to skip the extraction phase during your runs.

---

## Running PropRAG

### Evaluation / Testing
To run evaluations on a specific dataset:
```bash
python main.py --dataset <dataset_name>
```
Replace `<dataset_name>` with one of the supported dataset identifiers (e.g., `musique`, `hotpotqa`, `2wikimultihopqa`, `popqa`, `nq_rear`).

**Example:**
```bash
python main.py --dataset musique
```