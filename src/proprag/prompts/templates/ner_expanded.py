ner_system = """Your task is to extract entities from the given paragraph. 
Respond with a JSON dictionary only, with a "entities" key that maps to an non-empty list of entities.
All named entities and dates must be included in the list.
All generic entities important to the theme of the passage must be included in the list.
All entities that is involved in a predicate relation to the above entities must be included in the list.
All dates must be included in the list.
"""

one_shot_ner_paragraph = """Radio City
Radio City is India's first private FM radio station and was started on 3 July 2001.
It plays Hindi, English and regional songs.
Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features."""


one_shot_ner_output = """{"entities":
    ["Radio City", "India", "private FM radio station", "3 July 2001", "Hindi", "English", "New Media", "May 2008", "PlanetRadiocity.com", "music portal", "news", "videos", "songs"]
}
"""


prompt_template = [
    {"role": "system", "content": ner_system},
    {"role": "user", "content": one_shot_ner_paragraph},
    {"role": "assistant", "content": one_shot_ner_output},
    {"role": "user", "content": "${passage}"}
]