import torch  # type: ignore
import torch.nn.functional as F  # type: ignore
from PIL import Image
import io

def get_imagenet_classes():
    """Fallback utility to load standard ImageNet class names if needed."""
    try:
        from torchvision.models import ResNet50_Weights  # type: ignore
        return ResNet50_Weights.DEFAULT.meta["categories"]
    except Exception:
        return [f"Class_{i}" for i in range(1000)]

class InferenceEngine:
    """Encapsulates the model, transforms, and classes for robust inference."""
    def __init__(self, model: torch.nn.Module, transforms, classes: list = None, device: str = "cpu"):
        self.device = device
        self.model = model.to(self.device)
        self.model.eval()
        self.transforms = transforms
        self.classes = classes if classes is not None else get_imagenet_classes()

    def predict(self, image_bytes: bytes):
        """
        Runs an inference prediction on a single image.
        Returns: tuple: (predicted_class_name, confidence_percentage)
        """
        # Load image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Preprocess image
        tensor = self.transforms(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            # Get top prediction
            top_prob, top_class_idx = torch.max(probabilities, 1)
            
            confidence = top_prob.item() * 100
            predicted_class = self.classes[top_class_idx.item()]
            
        return predicted_class, confidence

