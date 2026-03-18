from dataclasses import dataclass, field

@dataclass
class Document:
    id: str
    content: str
    metadata: dict = field(default_factory=dict)
    hash: str = ""
