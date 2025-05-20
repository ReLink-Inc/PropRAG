from typing import Dict, Any, List, TypedDict, Tuple, Union

Triple = Union[Tuple[str, str, str], List[str]]

class Proposition(TypedDict):
    text: str
    entities: List[str]
    
Path = List[Tuple[str, float]]  # List of [(entity_key, score), ...]
BeamItem = Tuple[Path, float, Dict[str, Dict]]  # (path, score, entity_connections)