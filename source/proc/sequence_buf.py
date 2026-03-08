class SequenceBuffer:
    def __init__(self, sequence_length, stride=1):
        self.sequence_length = sequence_length
        self.stride = stride
        self.buffer = []
        self._new_frames_since_last_inference = 0
        self._first_window_completed = False

    def add_frame(self, frame_id, processed_frame):
        """
        Yeni bir kare ve onun indeksini ekler.
        """
        self.buffer.append((frame_id, processed_frame))
        self._new_frames_since_last_inference += 1
        
        # Buffer boyutunu koru
        if len(self.buffer) > self.sequence_length:
            self.buffer.pop(0)

    def is_ready(self):
        """
        Stride mantığına göre inference için yeterli yeni kare birikip birikmediğini kontrol eder.
        """
        # Buffer henüz dolmadıysa hazır değil
        if len(self.buffer) < self.sequence_length:
            return False
            
        # İlk pencere dolduğunda hemen hazır olur
        if not self._first_window_completed:
            return True
            
        # Sonraki pencereler için stride kadar yeni kare gelmesi gerekir
        return self._new_frames_since_last_inference >= self.stride

    def get_sequence(self):
        """
        Mevcut pencereyi (indeksler ve kareler) döndürür ve sayacı sıfırlar.
        """
        self._new_frames_since_last_inference = 0
        self._first_window_completed = True
        return self.buffer