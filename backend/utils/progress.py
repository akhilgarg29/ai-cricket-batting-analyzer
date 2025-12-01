"""Progress tracking for video processing"""
from typing import Callable, Optional

class ProgressTracker:
    def __init__(self, total: int, callback: Optional[Callable] = None):
        self.total = total
        self.current = 0
        self.callback = callback
    
    def update(self, increment: int = 1):
        self.current += increment
        percentage = (self.current / self.total) * 100
        
        if self.callback:
            self.callback(self.current, self.total, percentage)
        
        return percentage
    
    def reset(self):
        self.current = 0
