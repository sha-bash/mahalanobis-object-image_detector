"""Lazy exports: importing the Ollama HTTP client must not load PyTorch."""
from importlib import import_module

_MODULES = {
    'LLMClient': 'base', 'VLMClient': 'base',
    'GigaChatLLM': 'llm', 'StubLLM': 'llm',
    'EasyOCRClient': 'ocr', 'NullOCR': 'ocr', 'StubOCR': 'ocr',
    'ContextualVLM': 'vlm', 'GigaChatVLM': 'vlm', 'StubVLM': 'vlm',
    'load_image': 'vlm', 'VisualEncoder': 'visual_encoder',
}


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    value = getattr(import_module(f'{__name__}.{_MODULES[name]}'), name)
    globals()[name] = value
    return value

__all__ = [
    "VLMClient",
    "LLMClient",
    "StubVLM",
    "GigaChatVLM",
    "ContextualVLM",
    "StubLLM",
    "GigaChatLLM",
    "StubOCR",
    "EasyOCRClient",
    "NullOCR",
    "VisualEncoder",
]
