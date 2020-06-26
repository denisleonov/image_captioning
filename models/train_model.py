import torch
from tqdm.autonotebook import tqdm
import os
from noam_scheduler import NoamScheduler
from transformers.optimization import AdamW

def clip_grad(parameters, clip_value):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    for p in (p for p in parameters if p.grad is not None):
        p.grad.data.clamp_(min=-clip_value, max=clip_value)


def train_epoch(dataloader, model, optimizer, scheduler,
                PATH_TO_SAVE, save_interval=50):
    losses = []
    torch.cuda.empty_cache()

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
    for idx_batch, batch in pbar:

        imgs, ids, mask = [x.to(model.device) for x in batch]

        '''print(ids.shape)
        print(model.tokenizer.decode(ids[0]))
        display(transforms.ToPILImage()(imgs[0].cpu()))'''

        loss, *_ = model(imgs, ids, attn_mask=mask)
        loss.backward()

        if (idx_batch + 1) % optimizer.accum_interval == 0 or (idx_batch + 1) == len(dataloader):
            clip_grad(model.parameters(), optimizer.clip_value)
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()

        losses.append(loss.item())
        pbar.set_description(f'loss:{loss.item():.4f}, lr: {scheduler.noam_lr:.4f}')

        if (idx_batch + 1) % save_interval == 0:
            model.save(os.path.join(PATH_TO_SAVE, 'weights/temp'))


def train(model, dataloader,
          PATH_TO_SAVE,
          n_epochs=5,
          clip_value=1.0,
          accum_interval=5,
          lr=1e-3,
          warmup=50,
          timestep=0,
          save_interval=50):
    # model.load(os.path.join(PATH_TO_SAVE, 'weights/temp'))
    model.train()

    '''for param in model.embedding.parameters():
        print(param.requires_grad)'''

    optimizer = AdamW(
        [param for param in model.parameters()
         if param.requires_grad],
        lr=lr
    )
    optimizer.clip_value = clip_value
    optimizer.accum_interval = accum_interval
    scheduler = NoamScheduler(optimizer, warmup)
    scheduler.timestep = timestep

    for epoch in range(n_epochs):
        train_epoch(dataloader, model, optimizer, scheduler, PATH_TO_SAVE, save_interval=save_interval)
        print(f'Epoch #{epoch} finished')
        model.save(os.path.join(PATH_TO_SAVE, f'weights/full_trained_{epoch}'))