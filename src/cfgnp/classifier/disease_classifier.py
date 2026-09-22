import torch
from cfgnp.classifier.model.classifier import Classifier
from easydict import EasyDict as edict
import json
import torch.nn as nn
from cfgnp.util.util import DEVICE
from cfgnp.util.data_paths import ChexpertPath


class DiseaseClassifier(torch.nn.Module):
    def __init__(self, cfg_path=None):
        if cfg_path is None:
            cfg_path = ChexpertPath.get_disease_config_file()
        with open(cfg_path, 'r') as f:
            cfg = edict(json.load(f))
        super(DiseaseClassifier, self).__init__()
        self.num_classes = 2
        self.pretrained_model = Classifier(cfg)
        self.pretrained_model.load_state_dict(torch.load(ChexpertPath.get_disease_pretrained_model_path(), weights_only=True, map_location=DEVICE))
        # self.classifier = nn.Linear(5, 2)
        self.linear1 = nn.Sequential(
            nn.Linear(5, 128),
            nn.ReLU()
        )
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4)
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        # with torch.no_grad():  # Freeze the pretrained model
        (features, logits_map) = self.pretrained_model(x)
        features = torch.concat(features, dim=-1)
        features = self.linear1(features)
        features, _ = self.attention(features, features, features)
        output = self.classifier(features)
        return output