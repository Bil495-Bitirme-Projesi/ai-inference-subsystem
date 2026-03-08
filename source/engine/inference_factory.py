from typing import Dict, Type, Callable, Any

from config.config_manager import ConfigManager
from engine.interfaces import IInferenceEngine


class InferenceFactory:
    # Hem sınıfı hem de config tipini saklayan bir yapı
    _registry: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(cls, model_type: str, config_class: Type):
        def wrapper(engine_cls: Type):
            cls._registry[model_type] = {
                "engine": engine_cls,
                "config_schema": config_class,
            }
            return engine_cls

        return wrapper

    @classmethod
    def create(cls, model_type: str, config_path: str) -> IInferenceEngine:
        entry = cls._registry.get(model_type)
        if not entry:
            raise ValueError(f"Model '{model_type}' not found!")

        config = ConfigManager.load(config_path, entry["config_schema"])
        return entry["engine"](config)


def register_inference_engine(model_type: str):
    def decorator(creator_fn: Callable):
        InferenceFactory.register(model_type, creator_fn)
        return creator_fn

    return decorator
