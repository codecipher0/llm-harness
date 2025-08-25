from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Abstract base class for all models in the harness."""

    @abstractmethod
    def load(self, config: dict):
        """Load model weights, configs, etc."""
        pass

    @abstractmethod
    def infer(self, image, **kwargs):
        """Run inference on a single image (or batch)."""
        pass

    @abstractmethod
    def unload(self):
        """Cleanup resources (GPU memory, file handles)."""
        pass