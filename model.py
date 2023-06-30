import pickle
import torch
import torch.nn.functional as F
import numpy as np

class FiringRateModel(torch.nn.Module):
    def __init__(
        self, 
        g, # activation function
        k: int = 0 # number of previous timesteps for current I
    ):
        super().__init__()
        self.g = g
        self.a = torch.nn.Parameter(torch.ones(k) * 1e-1)
        self.b = torch.nn.Parameter(torch.randn(1)[0])
        self.k = k
        
    def forward(
        self,
        currents, # currents tensor, size k+1
        prev_f # previous firing rate
    ):
        if self.k > 0:
            x = currents[:-1] @ self.a
            return self.g(currents[-1] + x - self.b * prev_f)
        else:
            return self.g(currents[0] - self.b * prev_f)

class PolynomialActivation(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, z):
        x = (z - self.b) / self.max_current
        #poly = F.relu(self.poly_coeff) @ x.pow(self.p) # relu to ensure monotonicity
        poly = (self.poly_coeff ** 2) @ x.pow(self.p)
        tan = self.max_firing_rate * F.tanh(poly) # ceil is the max firing rate
        return F.relu(tan)
    
    # slightly ad hoc parameter initialization
    def init_params(self, degree, max_current, max_firing_rate, Is, fs, C=0):
        self.degree = degree # polynomial degree
        self.max_current = max_current # used for normalization
        self.max_firing_rate = max_firing_rate
        self.p = torch.tensor([d for d in range(degree+1)])
        self.C = C
        
        x1, x2, y1, y2 = tuple([torch.tensor(0.0)] * 4)
        xs, ys = map(list, zip(*sorted(zip(Is, fs), key=lambda x: x[0])))
        x2, y2 = xs[-1], ys[-1]
        for i in range(0, len(ys)):
            if ys[i] > 0:
                x1, y1 = xs[i], ys[i]
                break
        self.b = torch.nn.Parameter(x1.clone())
        self.poly_coeff = torch.ones(self.degree + 1) * 1e-1 # to make sure there is some gradient
        self.poly_coeff[1] = (y2 - y1) / (x2 - x1) * self.max_current * torch.abs(torch.randn(1)[0] * 7 + 10)
        self.poly_coeff = torch.nn.Parameter(self.poly_coeff)
        
    def init_from_file(self, filename):
        try:
            with open(filename, "rb") as file:
                d = pickle.load(file)
        except:
            print("Error")
        finally:
            self.max_current = d["max_current"]
            self.max_firing_rate = d["max_firing_rate"]
            self.poly_coeff = d["poly_coeff"]
            self.b = d["b"]
            self.C = d["C"]
            self.degree = len(self.poly_coeff) - 1
            self.p = torch.tensor([d for d in range(self.degree+1)])
            self.p = torch.nn.Parameter(self.p, requires_grad=False)
        
    def save_params(self, filename):
        d = {
            "max_current": self.max_current,
            "max_firing_rate": self.max_firing_rate,
            "poly_coeff": self.poly_coeff,
            "b": self.b,
            "C": self.C
        }
        
        with open(filename, 'wb') as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_max_firing_rate(data, cell_id):
    diffs = np.concatenate([np.diff(d["spike_times"]) for d in data[cell_id][-1]])
    return np.max(1 / diffs) / 1000 # return in ms^-1

def train_model(
    model,
    criterion,
    optimizer,
    Is_tr,
    fs_tr,
    k: int,
    epochs: int = 100,
    print_every: int = 10
):
    for epoch in range(epochs):
        total_loss = 0
        for currents, firing_rates in zip(Is_tr, fs_tr):
            f = firing_rates[0] # initialize firing rate to t=0
            loss = 0
            n = 0
            for i in range(k+1, len(currents)):
                currs = currents[i-k:i+1]
                f = model(currs, f)
                loss += criterion(f, firing_rates[i])
                n += 1
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1) # prevent gradient explosion
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch+1) % print_every == 0:
            print(f"Epoch {epoch+1} / Loss: {total_loss}")

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