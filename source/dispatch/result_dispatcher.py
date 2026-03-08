from dispatch.interfaces import IDispatcher


class ResultDispatcher(IDispatcher):
    def dispatch(self, detections, info, frame):
        # Zaman aralığı bilgisini yazdır
        time_range = f"[{info.get('start_sec')}s - {info.get('end_sec')}s]"
        print(f"\n[DISPATCH] {time_range} Results: {detections}")
