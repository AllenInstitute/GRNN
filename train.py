import torch
import torch.nn.functional as F

from utils import reshape_image

def train_model(
    model,
    criterion,
    optimizer,
    Is_tr,
    fs_tr,
    Is_te,
    fs_te,
    epochs: int = 100,
    print_every: int = 10,
    bin_size = 20,
    C = 0
):
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        total_loss = 0

        n = 0
        for Is, fs in zip(Is_tr, fs_tr):
            batch_size = Is.shape[0]
            loss = torch.zeros(batch_size).to(model.device)
            model.reset(batch_size)
            
            # Is has shape [B, seq_len]
            for i in range(Is.shape[1]):
                # Is[:, i] has shape [B], convert to [B,1]
                # model output has shape [B,1], convert to [B]
                f = model(Is[:, i].unsqueeze(1)).squeeze()
                loss += criterion(f * bin_size, fs[:, i] * bin_size)
            
            # L0 regularization
            reg = C * model.reg(p=0)
            mean_loss = torch.mean(loss)
            obj = mean_loss + reg
            optimizer.zero_grad()
            obj.backward(retain_graph=True)
            optimizer.step()

            # don't include reg term in total loss? only PoissonNLLLoss
            # normalize by seq length
            total_loss += mean_loss.item() / Is.shape[1]
            n += 1

        # normalize by number of batches
        train_losses.append(total_loss / n)

        n = 0
        total_test_loss = 0
        with torch.no_grad():
            for Is, fs in zip(Is_te, fs_te):
                batch_size = Is.shape[0]
                loss = torch.zeros(batch_size).to(model.device)
                model.reset(batch_size)
                
                # Is has shape [B, seq_len]
                for i in range(Is.shape[1]):
                    # Is[:, i] has shape [B], convert to [B,1]
                    # model output has shape [B,1], convert to [B]
                    f = model(Is[:, i].unsqueeze(1)).squeeze()
                    loss += criterion(f * bin_size, fs[:, i] * bin_size)
                
                mean_loss = torch.mean(loss)
                # normalize by seq length
                total_test_loss += mean_loss.item() / Is.shape[1]
                n += 1
            test_losses.append(total_test_loss / n)
        
        if (epoch+1) % print_every == 0:
            print(f"Epoch {epoch+1} | Loss: {total_loss}")

        if len(train_losses) >= 3 and train_losses[-1] == train_losses[-2] == train_losses[-3]:
            return train_losses, test_losses
        
    return train_losses, test_losses

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

def train_network(model, train_loader, epochs=30, lr=1e-3, variant="p", device=None):        
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for epoch in range(epochs):
        total_loss = 0

        for x, label in train_loader:
            x = reshape_image(x, variant=variant).to(device)
            
            # sequentially send input into network
            model.reset(x.shape[0])
            for i in range(x.shape[1]):
                model(x[:, i, :])
                
            pred_y = model(model.zero_input(x.shape[0]))
            loss = criterion(pred_y, F.one_hot(label, num_classes=10).to(torch.float32).to(device))

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=False)
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch+1) % 1 == 0:
            print(f"Epoch {epoch+1} | Loss: {total_loss}")
        losses.append(total_loss)
    return losses