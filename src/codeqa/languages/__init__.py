from codeqa.languages.registry import REGISTRY, LanguageSpec, detect_language, get_language
from codeqa.languages.tags import Tag, extract_tags

__all__ = [
    "LanguageSpec",
    "REGISTRY",
    "Tag",
    "detect_language",
    "extract_tags",
    "get_language",
]
