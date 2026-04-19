class SequenceBuffer:
    """
    Stride-based frame sampling buffer for video stream processing.
    
    Stride=4 örneği: frame 0, 4, 8, 12, 16... sadece bu frameler buffer'a eklenir.
    Canlı yayında her stride karenin sadece 1'i örneklenir.
    
    Attributes:
        sequence_length: Kaç frame'in sequence oluşturmasını istediğimiz (örn. 16)
        stride: Video streamden kaç karede 1 frame sampling yapılacağı (örn. 4)
    """
    
    def __init__(self, sequence_length, stride=1):
        """
        Args:
            sequence_length: Buffer'da kaç frame tutulacağı (örn. 16 frame = 1 sequence)
            stride: Video streamden örnekleme aralığı (1=her frame, 4=her 4. frame, vs.)
        """
        self.sequence_length = sequence_length
        self.stride = stride
        self.buffer = []
        self._first_window_completed = False

    def add_frame(self, frame_id, processed_frame):
        """
        Yeni frame'i stride mantığına göre ekler.
        
        Sadece frame_id % stride == 0 olan frame'ler eklenir.
        Örneğin stride=4 ise: 0, 4, 8, 12, 16... frame'ler eklenir.
        
        Args:
            frame_id: Video stream'deki absolute frame index
            processed_frame: İşlenen frame verisi
            
        Returns:
            bool: Frame'in buffer'a eklenip eklenmediği
        """
        # Stride mantığı: sadece stride'ın katı olan frame'leri ekle
        if frame_id % self.stride != 0:
            return False
        
        self.buffer.append((frame_id, processed_frame))
        
        # Buffer boyutunu koru (sequence_length kadar tut)
        if len(self.buffer) > self.sequence_length:
            self.buffer.pop(0)
        
        return True

    def is_ready(self):
        """
        Buffer'da yeterli frame olup olmadığını kontrol eder.
        
        Stride örneklemesi sonrası sequence_length kadar frame birikince ready olur.
        
        Returns:
            bool: Inference için hazır olunup olmadığı
        """
        # Buffer henüz dolmadıysa hazır değil
        if len(self.buffer) < self.sequence_length:
            return False
        
        # Buffer dolunca hazır
        return True

    def get_sequence(self):
        """
        Mevcut sequence'i döndürür ve ilk pencere bayrağını set eder.
        
        Returns:
            list: (frame_id, processed_frame) tuple'larının listesi, 
                  stride örnekleme ile alındı
        """
        self._first_window_completed = True
        return self.buffer
    
    def get_frame_count(self):
        """Buffer'da mevcut frame sayısını döndürür."""
        return len(self.buffer)

    def flush(self):
        self.buffer.clear()
