from tqdm.auto import tqdm, trange
import torch.nn as nn
import torch
import numpy as np

class Validation:
    def __init__(self, encoder, decoder):
        self.epoch = 0
        self.encoder = encoder
        self.decoder = decoder
        self.val_losses = []
        self.avr_val_losses = []
        self.criterion = nn.CrossEntropyLoss()

    def run(self, val_loader):
        self.epoch += 1
        vloader = tqdm(enumerate(val_loader), total=len(val_loader))
        vloader.set_description('Validation')
        for i_step, (val_images, val_captions) in vloader:
            with torch.no_grad():
                self.encoder.eval()
                self.decoder.eval()
                captions_target = val_captions[:, 1:].to(self.decoder.device)
                captions_train = val_captions[:, :-1].to(self.decoder.device)
                val_images = val_images.to(self.decoder.device)
                features = self.encoder(val_images)
                outputs = self.decoder(features, captions_train)[:, 1:, :]
                val_loss = self.criterion(outputs.contiguous().view(-1, outputs.size(-1)), captions_target.contiguous().view(-1))
            
            self.val_losses.append(val_loss.item())
            vloader.set_postfix(loss=val_loss.item())
        
        self.avr_val_losses.append(np.mean(self.val_losses[-len(val_loader):]))
        np.save('avr_val_losses', np.array(self.avr_val_losses))
        print(f'Average validation loss: {avr_val_losses[-1]}')