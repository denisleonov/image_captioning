import torch.nn as nn
import torch
from torch.distributions.categorical import Categorical

class Sampler:
    def __init__(self, model, top_p=0.8, max_desc_len=40):
        self.model = model
        self.top_p = top_p
        self.max_desc_len = max_desc_len
        self.eos_id = model.tokenizer.eos_token_id
        self.softmax = nn.Softmax(dim=-1)

    def initialize(self, img_data):
        self.context = self.model.get_context_embeds(img_data.to(self.model.device))
        self.desc_len = 1
        self.desc_ids_batch = []

    def run(self):
        for context in self.context:
            desc_ids = []
            context = context.unsqueeze(0)
            while self.desc_len <= self.max_desc_len:
                out = self.model.gpt2(inputs_embeds=context)[0]

                # get output fot last word
                out = out[:, -1, :]

                sample_id = self.sample(out).item()
                desc_ids.append(sample_id)
                if sample_id == self.eos_id:
                    break
                self.desc_len += 1

                sample_embeds = self.model.gpt2.transformer.wte(
                    torch.tensor([sample_id]).to(self.model.device)
                )

                context = torch.cat([context, sample_embeds[:, None, :]], dim=1)

            self.desc_ids_batch.append(desc_ids)

        return self.desc_ids_batch

    def sample(self, out):

        probs = self.softmax(out)
        sorted_probs, sorted_idxs = torch.sort(probs, descending=True)

        cumulative_probs = torch.cumsum(sorted_probs, dim=1)
        sorted_probs[cumulative_probs > self.top_p] = 0.
        sorted_probs[sorted_probs[:, 0] == 0, 0] = 1.
        sorted_probs /= sorted_probs.sum(dim=1, keepdim=True)

        sample = Categorical(sorted_probs).sample()
        sample_id = sorted_idxs.gather(1, sample.unsqueeze(1)).squeeze(1)

        return sample_id
