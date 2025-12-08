"""
Vision Tool for Dheera
Connect to image classification models.
"""

from typing import Dict, Any, Optional
from pathlib import Path


class VisionTool:
    """Image analysis capabilities for Dheera."""
    
    def __init__(self, model_name: str = "resnet18"):
        self.model = None
        self.model_name = model_name
        self.available = False
        self._load_model()
    
    def _load_model(self):
        """Load a pretrained vision model."""
        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as transforms
            
            # Load pretrained model
            if self.model_name == "resnet18":
                self.model = models.resnet18(pretrained=True)
            elif self.model_name == "resnet50":
                self.model = models.resnet50(pretrained=True)
            else:
                self.model = models.resnet18(pretrained=True)
            
            self.model.eval()
            
            # Standard ImageNet transforms
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
            
            # Load ImageNet labels
            self.labels = self._load_labels()
            self.available = True
            print(f"  ✓ Vision tool loaded: {self.model_name}")
            
        except ImportError as e:
            print(f"  ⚠ Vision tool not available: {e}")
            print("    Install: pip install torch torchvision pillow")
    
    def _load_labels(self):
        """Load ImageNet class labels."""
        # Simplified top labels
        return {i: f"class_{i}" for i in range(1000)}
    
    def analyze_image(self, image_path: str, top_k: int = 5) -> Dict[str, Any]:
        """Analyze an image and return predictions."""
        if not self.available:
            return {"error": "Vision model not loaded"}
        
        try:
            from PIL import Image
            import torch
            
            # Load and transform image
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probabilities, top_k)
            
            predictions = []
            for prob, idx in zip(top_probs, top_indices):
                predictions.append({
                    "class_id": idx.item(),
                    "confidence": prob.item(),
                })
            
            return {
                "success": True,
                "predictions": predictions,
                "image_path": image_path,
            }
            
        except Exception as e:
            return {"error": str(e)}


def register_vision_tool(tool_registry):
    """Register vision tool with Dheera."""
    vision = VisionTool()
    
    if vision.available:
        def analyze(image_path: str) -> str:
            result = vision.analyze_image(image_path)
            if "error" in result:
                return f"Error: {result['error']}"
            
            preds = result.get("predictions", [])
            if preds:
                top = preds[0]
                return f"I see: class {top['class_id']} with {top['confidence']*100:.1f}% confidence"
            return "Could not analyze image"
        
        tool_registry.register(
            name="analyze_image",
            function=analyze,
            description="Analyze an image and describe what's in it",
        )
    
    return vision
