import torch
import torch.nn.functional as F

from utils import reshape_image

def train_model(
    model,
    criterion,
    optimizer,
    Is_tr,
    fs_tr,
    epochs: int = 100,
    print_every: int = 10,
    bin_size = 20,
    up_factor = 10,
    scheduler = None
):
    losses = []

    for epoch in range(epochs):
        total_loss = 0
        for Is, fs in zip(Is_tr, fs_tr):
            batch_size = Is.shape[0]
            loss = torch.zeros(batch_size).to(model.device)
            model.reset(batch_size)
            
            # Is has shape [B, seq_len]
            for i in range(Is.shape[1]):
                # Is[:, i] has shape [B], convert to [B,1]
                # model output has shape [B,1], convert to [B]
                f = model(Is[:, i].unsqueeze(1)).squeeze()
                
                # up-weight loss for non-zero firing rate
                alpha = torch.ones(batch_size).to(model.device)
                alpha[torch.logical_or(fs[:, i] > 0, f > 0)] = up_factor
                loss += alpha * criterion(f * bin_size, fs[:, i] * bin_size)
            
            mean_loss = torch.mean(loss)
            optimizer.zero_grad()
            mean_loss.backward(retain_graph=True)
            optimizer.step()

            total_loss += mean_loss.item()
        losses.append(total_loss)

        if scheduler is not None:
            scheduler.step()
        
        if (epoch+1) % print_every == 0:
            if scheduler is None:
                print(f"Epoch {epoch+1} | Loss: {total_loss}")
            else:
                curr_lr = scheduler.get_last_lr()
                print(f"Epoch {epoch+1} | Loss: {total_loss} | lr: {curr_lr}")

        if len(losses) >= 3 and losses[-1] == losses[-2] == losses[-3]:
            return losses
        
    return losses

def fit_activation(
    actv,
    criterion,
    optimizer,
    Is,
    fs,
    epochs: int = 1000
):
    losses = []
    for _ in range(epochs):
        total_loss = 0
        for current, fr in zip(Is, fs):
            current = current.reshape(1, 1)
            pred_fr = actv(current)
            loss = criterion(pred_fr * actv.bin_size, fr.reshape(1, 1) * actv.bin_size)
            total_loss += loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        losses.append(total_loss.item())
    return losses

def train_network(model, train_loader, epochs=30, lr=0.005, variant="p", C=1, device=None):        
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 8, 12, 30, 40, 50, 60, 70], gamma=0.5)
    
    for epoch in range(epochs):
        total_loss = 0

        for x, label in train_loader:
            x = reshape_image(x, variant=variant).to(device)
            
            # sequentially send input into network
            model.reset(x.shape[0])
            for i in range(x.shape[1]):
                model(x[:, i, :])
                
            loss = 0
            for _ in range(5):
                pred_y = model(model.zero_input(x.shape[0]))
                loss += criterion(pred_y, F.one_hot(label, num_classes=10).to(torch.float32).to(device))
            loss /= 5
            loss += C * model.reg()

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            total_loss += loss
        scheduler.step()
        if (epoch+1) % 1 == 0:
            print(f"Epoch {epoch+1} | Loss: {total_loss} | lr: {scheduler.get_last_lr()}")