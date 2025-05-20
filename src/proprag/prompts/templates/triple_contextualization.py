from ...utils.llm_utils import convert_format_to_template

TRIPLE_CONTEXTUALIZATION_SYSTEM_PROMPT = """You are a helpful AI assistant specialized in generating contextual descriptions for relationship triples extracted from text. Your task is to enrich each triple with detailed context from the original passage.

For each given triple [subject, predicate, object], you will provide a concise but informative context that:
1. Explains the relationship described in the triple
2. Adds relevant information from the passage that isn't directly stated in the triple
3. Provides necessary context or background that helps understand the fact better
4. Makes the relationship more specific and clear
"""

TRIPLE_CONTEXTUALIZATION_USER_PROMPT = """I have extracted the following triples (facts) from a passage:

{triples}

Here is the original passage these triples were extracted from:

PASSAGE:
{passage}

For each triple, please provide a contextual description that enriches the triple with additional information from the passage.

Format your response as a JSON object with the following structure:
```json
{
  "contextualized_triples": [
    {
      "triple": ["subject", "predicate", "object"],
      "context": "A detailed and specific contextual description of the relationship based on the passage"
    },
    {
      "triple": ["subject", "predicate", "object"],
      "context": "Another contextual description..."
    },
    ...
  ]
}
```

Each context should be a single sentence of 15-30 words that provides a rich, specific description of the fact. Make sure to maintain all the original triples and just add context."""

# Register the template
prompt_template = [
    {"role": "system", "content": TRIPLE_CONTEXTUALIZATION_SYSTEM_PROMPT},
    {"role": "user", "content": convert_format_to_template(original_string=TRIPLE_CONTEXTUALIZATION_USER_PROMPT, placeholder_mapping=None, static_values=None)}
]