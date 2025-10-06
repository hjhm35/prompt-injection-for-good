from typing import Dict, Type, List
from .base import BaseProvider
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .deepseek_provider import DeepSeekProvider
from .together_provider import TogetherProvider
from .gemini_provider import GeminiProvider

# Provider registry
PROVIDERS: Dict[str, Type[BaseProvider]] = {
    'openai': OpenAIProvider,
    'anthropic': AnthropicProvider,
    'deepseek': DeepSeekProvider,
    'together': TogetherProvider,
    'gemini': GeminiProvider,
}

def get_provider(provider_name: str, **kwargs) -> BaseProvider:
    """Get provider instance by name"""
    if provider_name not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider_name}. Available: {list(PROVIDERS.keys())}")
    
    return PROVIDERS[provider_name](**kwargs)

def list_providers() -> List[str]:
    """Get list of available provider names"""
    return list(PROVIDERS.keys())

def get_all_models() -> Dict[str, List[str]]:
    """Get all available models from all providers"""
    models = {}
    for name, provider_class in PROVIDERS.items():
        try:
            # For dynamic providers, use cached models from web app
            if name in ['gemini', 'openai', 'anthropic', 'deepseek', 'together']:
                # Import the cached models function from web app
                try:
                    from ..web.app import get_cached_models
                    cached_models = get_cached_models(name)
                    if cached_models:
                        if name == 'gemini':
                            model_names = [model['base_model_id'] for model in cached_models if model.get('base_model_id')]
                        else:  # openai, anthropic, deepseek, together
                            model_names = [model['id'] for model in cached_models if model.get('id')]
                        models[name] = model_names
                    else:
                        models[name] = []
                except ImportError:
                    # Fallback to direct provider instantiation if web app not available
                    instance = provider_class()
                    provider_models = instance.get_available_models()
                    models[name] = provider_models if provider_models else []
            else:
                # For static providers, use direct instantiation
                instance = provider_class()
                provider_models = instance.get_available_models()
                models[name] = provider_models if provider_models else []
        except Exception as e:
            # If provider requires API key and it's not set, use static models if available
            if hasattr(provider_class, 'MODELS'):
                models[name] = provider_class.MODELS.copy()
            else:
                models[name] = []
    return models

__all__ = ['BaseProvider', 'OpenAIProvider', 'AnthropicProvider', 'DeepSeekProvider', 'TogetherProvider', 'GeminiProvider', 'get_provider', 'list_providers', 'get_all_models']