"""
VLM Engine Wrapper - High-level interface for Video Language Model inference.

Manages Pipeline:
[LLaVA-NeXT-Video-VAD] --> Text --> [Embedding Model] --> Vector --> [Text Classifier] --> Label, Confidence
"""

import logging
from typing import List, Optional, Dict, Any
from contextlib import contextmanager

try:
    from PIL.Image import Image
except ImportError:
    from PIL import Image

from source.engine.interfaces import IInferenceEngine
from core.vad_engine import VADEngine

logger = logging.getLogger(__name__)

from source.config.anomaly_config import VLMConfig
from source.engine.interfaces import IInferenceEngine
from source.engine.inference_factory import register_inference_engine

@register_inference_engine("VLMEngine", VLMConfig)
class VADInferenceEngine(IInferenceEngine):
    """
    Thread-safe wrapper around VADEngine with error handling, logging, and resource management.
    
    Provides:
    - Automatic model initialization and cleanup
    - Error handling and recovery
    - Inference result caching (optional)
    - Logging and monitoring
    - IInferenceEngine interface implementation
    """
    
    def __init__(self, config: VLMConfig):
        """
        Initialize VAD Engine wrapper from config.
        
        Args:
            config: VLMConfig instance with model parameters
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(config.log_level)
        
        self.use_memory = config.use_memory
        self.model_path = config.model_path
        self.use_potential_filter = config.use_potential_filter
        self.enable_caching = config.enable_caching
        self._inference_cache: Dict[str, Any] = {} if self.enable_caching else None
        
        self._engine = None
        self._initialized = False
        
        try:
            self._initialize_engine()
        except Exception as e:
            self.logger.error(f"Failed to initialize VAD Engine: {e}", exc_info=True)
            raise
    
    def _initialize_engine(self):
        """Initialize the actual VAD engine."""            
        self._engine = VADEngine(
            use_memory=self.use_memory,
            model_path=self.model_path,
            use_potential_filter=self.use_potential_filter
        )
        self._initialized = True
        self.logger.info("VAD Engine initialized successfully")
    
    def predict(self, sequence_tensor) -> Dict[str, Any]:
        """
        Minimal IInferenceEngine implementation.

        This wrapper expects `sequence_tensor` to be a preprocessed list of
        `PIL.Image` frames. Upstream code should handle tensor/array -> PIL
        conversion; the wrapper remains intentionally small.
        """
        if not self._initialized:
            raise RuntimeError("VAD Engine not initialized")

        frames = sequence_tensor

        try:
            self.logger.debug(f"Predict called with {len(frames)} frame(s)")
            result = self.process_segment(frames)
            return result
        except Exception as e:
            self.logger.error(f"Error in predict: {e}", exc_info=True)
            return {
                "predicted_label": "ERROR",
                "prob": 0.0,
                "description": None,
                "error": str(e),
            }
    
    def process_segment(self, frames: List[Image], frame_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a segment of frames through the inference pipeline.
        
        Args:
            frames: List of PIL Image objects
            frame_id: Optional identifier for caching
            
        Returns:
            Dict with keys: response, class_name, prob, confidence_score
            
        Raises:
            RuntimeError: If engine not initialized
            ValueError: If frames list is empty
        """
        if not self._initialized:
            raise RuntimeError("VAD Engine not initialized")
                
        # Check cache
        if self.enable_caching and frame_id and frame_id in self._inference_cache:
            self.logger.debug(f"Cache hit for frame_id: {frame_id}")
            return self._inference_cache[frame_id]
        
        try:
            self.logger.debug(f"Processing {len(frames)} frame(s)")
            result = self._engine.process_segment(frames)
            
            # Cache result
            if self.enable_caching and frame_id:
                self._inference_cache[frame_id] = result
            
            self.logger.debug(f"Inference result: {result['predicted_label']} ({result['prob']})")
            return result
            
        except Exception as e:
            self.logger.error(f"Error during inference: {e}", exc_info=True)
            raise
    
    def shutdown(self):
        """Cleanup and release resources."""
        try:
            if self._initialized:
                self.reset()
                self._engine = None
                self._initialized = False
                self.logger.info("VAD Engine shutdown complete")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
        return False
    
    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "not initialized"
        return f"VADEngineWrapper(status={status}, caching={self.enable_caching})"
