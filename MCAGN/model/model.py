from model.cross_graph import CrossGraph
from transformers import ViTModel, BertModel
import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, n_heads=12, cross_mlp_dim=1024, n_fusion_layers=3):
        super().__init__()
        self.ViT = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.cross_graph = CrossGraph
        self.n_heads = n_heads
        self.cross_mlp_dim = cross_mlp_dim
        self.n_fusion_layers = n_fusion_layers
        self.fusion = nn.ModuleList([self.cross_graph(768, self.n_heads, self.cross_mlp_dim)
                                     for _ in range(self.n_fusion_layers)])
        self.s_head = nn.Sequential(nn.Linear(1536, 1536), nn.ReLU(),
                                    nn.Linear(1536, 3))
        self.e_head = nn.Sequential(nn.Linear(1536, 1536), nn.ReLU(),
                                    nn.Linear(1536, 6))
        self.d_head = nn.Sequential(nn.Linear(1536, 1536), nn.ReLU(),
                                    nn.Linear(1536, 7))

    def forward(self, txt, img):
        image_output_ = self.ViT(img)
        image_output = image_output_[0][:,1:,:]
        output = self.bert(**txt)[0]
        for layer in self.fusion:
            output = layer(output, image_output)
        cls = output[:, 0, :]
        cls = torch.cat((cls, image_output_[1]), dim=1)
        s = self.s_head(cls)
        e = self.e_head(cls)
        d = self.d_head(cls)
        return s,e,d


class TxtOnly(nn.Module):
    def __init__(self, n_heads=12, cross_mlp_dim=1024, n_fusion_layers=3):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.s_head = nn.Sequential(nn.Linear(768, 768), nn.ReLU(),
                                    nn.Linear(768, 3))
        self.e_head = nn.Sequential(nn.Linear(768, 768), nn.ReLU(),
                                    nn.Linear(768, 6))
        self.d_head = nn.Sequential(nn.Linear(768, 768), nn.ReLU(),
                                    nn.Linear(768, 7))

    def forward(self, txt, img):
        output = self.bert(**txt)[0]
        cls = output[:, 0, :]
        s = self.s_head(cls)
        e = self.e_head(cls)
        d = self.d_head(cls)
        return s,e,d


class VisOnly(nn.Module):
    def __init__(self, n_heads=12, cross_mlp_dim=1024, n_fusion_layers=3):
        super().__init__()
        self.ViT = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.s_head = nn.Sequential(nn.Linear(768, 768), nn.ReLU(),
                                    nn.Linear(768, 3))
        self.e_head = nn.Sequential(nn.Linear(768, 768), nn.ReLU(),
                                    nn.Linear(768, 6))
        self.d_head = nn.Sequential(nn.Linear(768, 768), nn.ReLU(),
                                    nn.Linear(768, 7))

    def forward(self, txt, img):
        image_output_ = self.ViT(img)
        cls = image_output_[0][:, 0, :]
        s = self.s_head(cls)
        e = self.e_head(cls)
        d = self.d_head(cls)
        return s,e,d


class NoCrossAttn(nn.Module):
    def __init__(self, n_heads=12, cross_mlp_dim=1024, n_fusion_layers=3):
        super().__init__()
        self.ViT = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.s_head = nn.Sequential(nn.Linear(1536, 1536), nn.ReLU(),
                                    nn.Linear(1536, 3))
        self.e_head = nn.Sequential(nn.Linear(1536, 1536), nn.ReLU(),
                                    nn.Linear(1536, 6))
        self.d_head = nn.Sequential(nn.Linear(1536, 1536), nn.ReLU(),
                                    nn.Linear(1536, 7))

    def forward(self, txt, img):
        image_output_ = self.ViT(img)
        image_output = image_output_[0][:,1:,:]
        output = self.bert(**txt)[0]
        cls = output[:, 0, :]
        cls = torch.cat((cls, image_output_[1]), dim=1)
        s = self.s_head(cls)
        e = self.e_head(cls)
        d = self.d_head(cls)
        return s,e,d