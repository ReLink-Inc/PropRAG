ner_system = """Your task is to extract key entities that are in relation to other entities from the given paragraph. 
Respond with a JSON dictionary with a "key_entities" key that maps to an non-empty list of key entities. 
All named entities must be included in the list.
Output JSON only.
"""

one_shot_ner_paragraph = """Radio City
Radio City is India's first private FM radio station and was started on 3 July 2001.
It plays Hindi, English and regional songs.
Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features."""


one_shot_ner_output = """{"key_entities":
    ["Radio City", "India", "private FM radio station", "3 July 2001", "Hindi", "English", "New Media", "May 2008", "PlanetRadiocity.com", "music portal", "news", "videos", "songs"]
}
"""

two_shot_ner_paragraph = """Dishwasher
Dishwasher A dishwasher is a mechanical device for cleaning dishware and cutlery automatically. 
Unlike manual dishwashing, which relies largely on physical scrubbing to remove soiling, the mechanical dishwasher cleans by spraying hot water, typically between , at the dishes, with lower temperatures used for delicate items. 
A mix of water and dishwasher detergent is pumped to one or more rotating spray arms, which blast the dishes with the cleaning mixture."""

two_shot_ner_output = """{"key_entities":
    ["Dishwasher", "mechanical device", "dishware", "cutlery", "hot water", "rotating spray arms", "cleaning mixture"]
}
"""


prompt_template = [
    {"role": "system", "content": ner_system},
    {"role": "user", "content": one_shot_ner_paragraph},
    {"role": "assistant", "content": one_shot_ner_output},
    {"role": "user", "content": two_shot_ner_paragraph},
    {"role": "assistant", "content": two_shot_ner_output},
    {"role": "user", "content": "${passage}"}
]