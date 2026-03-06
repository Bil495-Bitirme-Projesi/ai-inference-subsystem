import numpy as np
import cv2

class Preprocessor:
    def process(self, raw_frame: np.ndarray, target_size=(224, 224)) -> np.ndarray:
        h, w = raw_frame.shape[:2]
        tw, th = target_size
        ratio = min(tw / w, th / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        
        # Oranlı yeniden boyutlandırma
        resized = cv2.resize(raw_frame, (new_w, new_h))
        
        # Siyah bir canvas oluştur
        canvas = np.zeros((th, tw, 3), dtype=np.uint8)
        
        # Merkeze yerleştir
        y_offset = (th - new_h) // 2
        x_offset = (tw - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        # Normalize ve RGB dönüşümü (VideoMAE beklediği gibi)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        return canvas.astype(np.float32) / 255.0
