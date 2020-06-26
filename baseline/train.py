import torch.optim as optim
from tqdm.auto import tqdm, trange
import torch.nn as nn
import numpy as np

class Train:
    def __init__(self, encoder, decoder, decoder_lr=0.001, encoder_lr=0.0001):
        self.epoch = 0
        self.encoder = encoder
        self.decoder = decoder
        self.losses = []
        self.criterion = nn.CrossEntropyLoss()
        self.decoder_optimizer = optim.Adam(decoder.parameters(), lr=decoder_lr)
        self.encoder_optimizer = optim.Adam(encoder.parameters(), lr=encoder_lr)
        self.decoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.decoder_optimizer,
                                                                mode='min',
                                                                factor=0.5,
                                                                patience=1)
        self.encoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.encoder_optimizer,
                                                         mode='min',
                                                         factor=0.5,
                                                         patience=1)

    def run(self, train_loader, avr_val_loss=None, accum_interval = 5, save_every = None):
        if avr_val_loss is not None:
            self.decoder_scheduler.step(avr_val_loss)
            self.encoder_scheduler.step(avr_val_loss)
        self.epoch += 1
        if save_every is None:
            save_every = len(train_loader)//2
        self.decoder.zero_grad()
        self.encoder.zero_grad()
        tloader = tqdm(enumerate(train_loader), total=len(train_loader))
        tloader.set_description('Train')
        for i_step, (images, captions) in tloader:
            self.encoder.train()
            self.decoder.train()

            captions_target = captions[:, 1:].to(self.decoder.device)
            captions_train = captions[:, :-1].to(self.decoder.device)
            images = images.to(self.decoder.device)
            features = self.encoder(images)
            outputs = self.decoder(features, captions_train)[:, 1:, :]

            loss = self.criterion(outputs.contiguous().view(-1, outputs.size(-1)), captions_target.contiguous().view(-1))
            loss.backward()
            if (i_step + 1) % accum_interval == 0 or (i_step + 1) == len(train_loader):
                self.decoder_optimizer.step()
                self.encoder_optimizer.step()
                self.decoder.zero_grad()
                self.encoder.zero_grad()
                
            self.losses.append(loss.item())
            tloader.set_postfix(loss=loss.item())
                
            if (i_step + 1) % save_every == 0:
                self.decoder.save_weights(self.epoch, i_step//save_every)
                self.encoder.save_weights(self.epoch, i_step//save_every)
        np.save('losses', np.array(self.losses))
        p1, = plt.plot([np.mean(self.losses[i:i+300]) for i in range(len(self.losses) - 300)], label='Train loss')
        plt.legend(handles=[p1], loc='upper right')
        plt.show()