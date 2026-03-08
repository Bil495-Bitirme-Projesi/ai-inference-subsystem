import torch
import time
from transformers import VideoMAEForVideoClassification

from config.anomaly_config import VideoMAEAnomalyConfig
from engine.interfaces import IInferenceEngine
from engine.inference_factory import register_inference_engine

class_mapping = {
    "Abuse": 0,
    "Arrest": 1,
    "Arson": 2,
    "Assault": 3,
    "Burglary": 4,
    "Explosion": 5,
    "Fighting": 6,
    "Normal Videos": 7,
    "Road Accidents": 8,
    "Robbery": 9,
    "Shooting": 10,
    "Shoplifting": 11,
    "Stealing": 12,
    "Vandalism": 13,
}

reverse_mapping = {v: k for k, v in class_mapping.items()}


@register_inference_engine("VideoMAE", VideoMAEAnomalyConfig)
class VideoMAEAnomalyEngine(IInferenceEngine):
    def __init__(self, config):
        self._config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = VideoMAEForVideoClassification.from_pretrained(
            config.model_path,
            label2id=class_mapping,
            id2label=reverse_mapping,
            ignore_mismatched_sizes=True,
        ).to(self.device)
        self._model.eval()

    def predict(self, sequence_tensor: torch.Tensor):
        with torch.no_grad():
            # Start per-video timer
            start_video = time.time()

            video_tensor = sequence_tensor.to(self.device)

            # Forward pass
            outputs = self._model(video_tensor)

            if self.device == "cuda":
                torch.cuda.synchronize()

            end_video = time.time()
            elapsed_video = end_video - start_video

            # Get predictions
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze()
            predicted_label = torch.argmax(probs, dim=-1).item()

            output = {
                "predicted_label": reverse_mapping[predicted_label],
                "probs": probs[predicted_label].item(),
                "elapsed_video": elapsed_video,
            }

            print(f"[LOG] VideoMAE: \n{output}")
            return output
