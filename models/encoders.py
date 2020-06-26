import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class EmbeddingFinalLayer(nn.Module):
    def __init__(self, cnn, embed_size):
        super().__init__()
        self.cnn = cnn
        self.embed_size = embed_size
        _, last_module = list(cnn.named_children())[-1]
        self.proj = nn.Linear(last_module.out_features, embed_size)

    def forward(self, x):
        x = self.cnn(x)
        return self.proj(x)[:, None, :] # new shape: (batch_size, seq_len, embed_size)

    def train(self, mode=True):
        self.cnn.train(mode)
        self.proj.train(mode)
        for param in self.cnn.parameters():
            param.requires_grad_(mode)


class EmbeddingGridInception(models.Inception3):
    def __init__(self, embed_size):
        super().__init__(init_weights=False)

        from torchvision.models.inception import load_state_dict_from_url, model_urls
        state_dict = load_state_dict_from_url(model_urls['inception_v3_google'],
                                              progress=True)
        self.load_state_dict(state_dict)

        del self.fc, self.AuxLogits
        self.aux_logits = False

        self.embed_size = embed_size
        self.proj = nn.Linear(2048, embed_size)

    def _forward(self, x):
        '''Change forward for inception model.
        For more details look at:
        https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py
        '''

        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)

        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)

        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)

        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)

        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8

        return x

    def forward(self, x):
        inception_out = self._forward(x)
        projected_out = self.proj(inception_out.permute(0, 2, 3, 1).view(-1, 64, 2048))
        return projected_out  # N, seq_len=64, embed_size

    '''def train(self, mode=True):
        self.proj.train(mode)
        self.train(mode)
        for param in self.parameters():
            param.requires_grad_(mode)'''
