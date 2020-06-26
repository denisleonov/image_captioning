import torchvision.models as models
import torch.nn as nn
import torch
import os
import os.path
from torch.distributions.categorical import Categorical

class EncoderCNN(nn.Module):
    def __init__(self, embed_size = 1024):
        super(EncoderCNN, self).__init__()
#         self.model = models.mobilenet_v2(pretrained=True)
#         self.model.classifier = nn.Linear(1280, 1024)
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(2048, 1024)
        self.embed = nn.Linear(in_features=1024, out_features=embed_size)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.init_weights()
        
    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.embed.bias.data.fill_(0)
        
    def forward(self, images):
        model_outputs = self.dropout(self.relu(self.model(images)))
        embeddings = self.embed(model_outputs)
        return embeddings.unsqueeze(1)
    
    def load_weights(self, path):
        with open(path, 'rb') as f:
            self.load_state_dict(torch.load(f))
            
    def save_weights(self, epoch, step, path='drive/My Drive/AU/ImageCaptioning'):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, f'encoder-{epoch}-{step}.pth'))

class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, start_id, eos_id, embed_size=1024, hidden_size=1024, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.start_id = start_id
        self.eos_id = eos_id
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.lstm_cell = nn.LSTM(input_size=embed_size, hidden_size=hidden_size,
                                     num_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)
        self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_size)
        self.softmax = nn.Softmax(dim=-1)
        self.device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
    
    def forward(self, features, captions):
        captions_embed = self.embed(captions)
        inputs = torch.cat((features, captions_embed), dim=1)
        hidden_state, cell_state = self.lstm_cell(inputs) 
        return self.fc_out(hidden_state)

    def sample(self, features, states=None, max_length=20):
        output = []
        start_ids = torch.ones((features.size(0), 1), device=self.device, dtype=torch.long) * self.start_id
        start_embeds = self.embed(start_ids)
        inputs = torch.cat((features, start_embeds), dim=1)
        with torch.no_grad():
            for t in range(max_length):
                inputs, states = self.lstm_cell(inputs, states)
                inputs = inputs[:, -1, :]
                out = self.fc_out(inputs)
                out = self.top_k(out)
                if out.item() == self.eos_id: break
                output.append(out.item())
                inputs = self.embed(out).unsqueeze(1)
        return output

    def top_k(self, out, k=5):
        probs = self.softmax(out)
        sorted_probs, sorted_idxs = torch.sort(probs, descending=True)

        sorted_probs[:, k:] = 0
        sorted_probs /= sorted_probs.sum(dim=1, keepdim=True)

        sample = Categorical(sorted_probs).sample()
        sample_id = sorted_idxs.gather(1, sample.unsqueeze(1)).squeeze(1)
        return sample_id

    def top_p(self, out, p=0.9):
        probs = self.softmax(out)
        sorted_probs, sorted_idxs = torch.sort(probs, descending=True)

        cumulative_probs = torch.cumsum(sorted_probs, dim=1)
        sorted_probs[cumulative_probs > p] = 0.
        sorted_probs /= sorted_probs.sum(dim=1, keepdim=True)

        sample = Categorical(sorted_probs).sample()
        sample_id = sorted_idxs.gather(1, sample.unsqueeze(1)).squeeze(1)
        return sample_id
    
    def load_weights(self, path):
        with open(path, 'rb') as f:
            self.load_state_dict(torch.load(f))
            
    def save_weights(self, epoch, step, path='drive/My Drive/AU/ImageCaptioning'):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, f'decoder-{epoch}-{step}.pth'))