class SequenceBuffer:
    def __init__(self, sequence_length, stride=1):
        self.sequence_length = sequence_length
        self.stride = stride
        self.current_sequence = []

    def add_frame(self, processed_frame):
        self.current_sequence.append(processed_frame)
        if len(self.current_sequence) > self.sequence_length:
            self.current_sequence.pop(0)

    def is_ready(self):
        return len(self.current_sequence) == self.sequence_length

    def get_sequence(self):
        return self.current_sequence