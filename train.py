import torch

def train_model(
    model,
    criterion,
    optimizer,
    Is_tr,
    fs_tr,
    epochs: int = 100,
    print_every: int = 10,
    loss_fn = "huber",
    bin_size = 20,
    up_factor = 10,
    ws = None,
    closed = True,
    C = 0,
    scheduler = None
):
    if ws is None:
        ws = [1 for _ in range(len(Is_tr))]
    losses = []
    k, l, m, n = model.k, model.l, model.m, model.n
    p = max(k, l, m, n)
    for epoch in range(epochs):
        total_loss = 0
        for currents, firing_rates, w in zip(Is_tr, fs_tr, ws):
            loss = 0
            pred_fs = firing_rates[:p]
            for i in range(p, len(currents)):
                if closed:
                    fs = pred_fs[:i]
                    f = model(currents[:i+1], fs)
                    pred_fs = torch.cat((pred_fs, f.reshape(1)))
                else:
                    f = model(currents[:i+1], firing_rates[:i])
                
                # up-weight loss for non-zero firing rate
                alpha = up_factor if firing_rates[i] > 0 or f > 0 else 1
                if loss_fn == "poisson":
                    loss += alpha * criterion(f * bin_size, firing_rates[i] * bin_size)
                else:
                    loss += alpha * criterion(f, firing_rates[i])
            loss += C * model.norm(p=1)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1) # prevent gradient explosion
            optimizer.step()
            total_loss += w * loss.item()
        if scheduler is not None:
            scheduler.step()
        losses.append(total_loss)
        if (epoch+1) % print_every == 0:
            if scheduler is None:
                print(f"Epoch {epoch+1} / Loss: {total_loss}")
            else:
                curr_lr = scheduler.get_last_lr()
                print(f"Epoch {epoch+1} / Loss: {total_loss} / lr: {curr_lr}")
        if len(losses) >= 3 and losses[-1] == losses[-2] == losses[-3]:
            return losses
    return losses

def fit_activation(
    actv,
    criterion,
    optimizer,
    Is,
    fs,
    epochs: int = 1000,
    C = 0.1,
    loss_fn = "huber"
):
    losses = []
    for _ in range(epochs):
        total_loss = 0
        for current, fr in zip(Is, fs):
            pred_fr = actv(current)
            if loss_fn == "poisson":
                loss = criterion(pred_fr * actv.bin_size, fr * actv.bin_size)
            else:
                loss = criterion(pred_fr, fr)
            total_loss += loss
        
        # L2 regularization
        total_loss += C * torch.mean(torch.pow(actv.poly_coeff[1:], 2))
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        losses.append(total_loss.item())
    return losses