import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Create mock modules to handle missing dependencies in the CI/test environment
mock_torch = MagicMock()
mock_transformers = MagicMock()

# Inject mocks into sys.modules before any project imports
sys.modules["torch"] = mock_torch
sys.modules["transformers"] = mock_transformers

# Mock torch.nn.functional
mock_torch.nn.functional.softmax.return_value = MagicMock()
# Mock torch.cuda
mock_torch.cuda.is_available.return_value = False

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Now import the project components
from source.engine.videomae_engine import VideoMAEAnomalyEngine, reverse_mapping
from source.config.anomaly_config import VideoMAEAnomalyConfig

class TestVideoMAEAnomalyEngine(unittest.TestCase):
    def setUp(self):
        # Mock the configuration
        self.config = VideoMAEAnomalyConfig(model_path="mock/path")
        
        # Reset mocks
        mock_transformers.VideoMAEForVideoClassification.from_pretrained.reset_mock()
        
        # Mock the model instance returned by from_pretrained
        self.mock_model = MagicMock()
        mock_transformers.VideoMAEForVideoClassification.from_pretrained.return_value = self.mock_model
        
        # chained calls: .to(device).eval()
        self.mock_model.to.return_value = self.mock_model
        
        # Initialize the engine
        self.engine = VideoMAEAnomalyEngine(self.config)

    def test_initialization(self):
        # Verify from_pretrained was called with correct arguments
        mock_transformers.VideoMAEForVideoClassification.from_pretrained.assert_called_once()
        args, kwargs = mock_transformers.VideoMAEForVideoClassification.from_pretrained.call_args
        self.assertEqual(args[0], "mock/path")
        self.mock_model.eval.assert_called_once()

    def test_predict(self):
        # Prepare dummy input tensor mockup
        dummy_input = MagicMock()
        
        # Mock model output
        mock_output = MagicMock()
        # mock logits
        mock_logits = MagicMock()
        mock_output.logits = mock_logits
        self.mock_model.return_value = mock_output
        
        # Mock softmax result
        mock_probs = MagicMock()
        mock_torch.nn.functional.softmax.return_value = mock_probs
        
        # Squeeze result
        mock_probs_squeezed = MagicMock()
        mock_probs.squeeze.return_value = mock_probs_squeezed
        
        # argmax result
        mock_argmax = MagicMock()
        mock_torch.argmax.return_value = mock_argmax
        mock_argmax.item.return_value = 0 # Abuse
        
        # Index result for return
        mock_probs_squeezed.__getitem__.return_value = 0.95
        
        # Run prediction
        result = self.engine.predict(dummy_input)
        
        # Verify model was called
        self.mock_model.assert_called_once()
        
        # Verify return value
        self.assertEqual(result, 0.95)

    def test_interface_compliance(self):
        # Verify init method exists
        self.assertTrue(hasattr(self.engine, "init"))
        # Should not raise any error
        self.engine.init(self.config)

if __name__ == "__main__":
    # Ensure torch constants used in code are available
    mock_torch.no_grad = MagicMock()
    mock_torch.no_grad.return_value.__enter__ = MagicMock()
    mock_torch.no_grad.return_value.__exit__ = MagicMock()
    
    unittest.main()
