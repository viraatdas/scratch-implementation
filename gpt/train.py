from model import Transformer
import torch
import torch.nn as nn


device = "cpu"
lr = 1e-9
num_epochs = 100

model = Transformer()

optimizer = torch.optim.Adam(model.parameters(), lr, eps=1e-9)
loss_fn = nn.CrossEntropyLoss().to(device)


for epoch in range(num_epochs):
  torch.cuda.empty_cache()
  model.train()
  batch_iterator = None
  for batch in batch_iterator:
    encoder_output = model.encode()
    decoder_output = model.decode(encoder_output)

    proj_output = model.project(decoder_output)

    loss = loss_fn(...)
    loss.backward()
    optimizer.step()
