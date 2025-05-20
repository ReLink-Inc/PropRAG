ner_system = """Your task is to fix the user provided JSON.
Respond with the fixed JSON only.
"""


prompt_template = [
    {"role": "system", "content": ner_system},
    {"role": "user", "content": "${json}"}
]