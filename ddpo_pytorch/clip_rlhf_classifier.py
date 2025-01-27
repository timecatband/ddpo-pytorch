import torch
import torch.nn as nn

from transformers import CLIPProcessor, CLIPModel

# Model Definition
class LinearClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes=2):
        super(LinearClassifier, self).__init__()
        self.hidden_dim = embedding_dim
        self.linear = nn.Linear(embedding_dim, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.relu = nn.ReLU()
        self.output = nn.Linear(self.hidden_dim, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.net = nn.Sequential(self.linear, self.relu, self.linear2, self.relu, self.output)

    def forward(self, embeddings):
        return self.net(embeddings)


class CLIPRLHFClassifier(nn.Module):
    def __init__(self, dtype, weights, clip_model="ViT-B/32"):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.classifier = LinearClassifier(768)
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")        self.classifier.load_state_dict(torch.load(weights))
        self.classifier = self.classifier.to(self.device)
        self.classifier.eval()
        self.dtype = dtype
    
    @torch.no_grad()
    def __call__(self, images):
        device = next(self.parameters()).device
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.dtype).to(device) for k, v in inputs.items()}
        embed = self.clip.get_image_features(**inputs)
        outputs = self.classifier(embed)
        outputs = outputs.squeeze(1)
        scores = outputs[:, 1] - outputs[:, 0]
        return scores
        