import pickle
import torch
import torch.nn.functional as F
import numpy as np

class FiringRateModel(torch.nn.Module):
    def __init__(
        self, 
        g, # activation function
        k: int = 0, # number of previous timesteps for current I
        l: int = 0, # number of timesteps for firing rate
        a = None,
        b = None,
        static_g: bool = True
    ):
        super().__init__()
        self.g = g
        self.k = k
        self.l = l
        self.a = torch.nn.Parameter(torch.zeros(k) if a is None else a)
        self.b = torch.nn.Parameter(torch.zeros(l) if b is None else b)
        
        # freeze activation parameters
        if static_g:
            for _, p in self.g.named_parameters():
                p.requires_grad = False
        
    def forward(
        self,
        currents, # currents tensor, size k+1 (t-k,...,t)
        fs # firing rates, size l (t-l,...,t-1)
    ):
        if self.k > 0:
            return self.g(currents[-1] + self.a @ currents[:-1] - self.b @ fs)
        else:
            return self.g(currents[0] - self.b @ fs)
    
    def init_from_params(self, params):
        self.a = torch.nn.Parameter(params["a"])
        self.b = torch.nn.Parameter(params["b"])
        self.g = PolynomialActivation()
        self.g.init_from_params(params["g"])
        self.k = len(self.a)
        self.l = len(self.b)

        for _, p in self.g.named_parameters():
            p.requires_grad = False

    def get_params(self):
        return {
            "a": self.a.clone(),
            "b": self.b.clone(),
            "g": self.g.get_params()
        }

    def predict(self, Is):
        k, l = self.k, self.l
        pad = max(k, l)
        Is_pad = F.pad(Is, (pad, 0), "constant")
        with torch.no_grad():
            fs = torch.zeros(pad)
            pred_fs = []
            for i in range(pad, len(Is_pad)):
                currs = Is_pad[i-k:i+1]
                f = self(currs, fs[i-l:i])
                fs = torch.cat((fs, f.reshape(1)))
                pred_fs.append(f)
        return np.array([f.item() for f in pred_fs])

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
                params = pickle.load(file)
        except:
            print("Error")
        finally:
            self.init_from_params(params)

    def init_from_params(self, params):
        self.max_current = params["max_current"]
        self.max_firing_rate = params["max_firing_rate"]
        self.poly_coeff = params["poly_coeff"]
        self.b = params["b"]
        self.C = params["C"]
        self.degree = len(self.poly_coeff) - 1
        self.p = torch.tensor([d for d in range(self.degree+1)])
        self.p = torch.nn.Parameter(self.p, requires_grad=False)
    
    def get_params(self):
        return {
            "max_current": self.max_current,
            "max_firing_rate": self.max_firing_rate,
            "poly_coeff": self.poly_coeff,
            "b": self.b,
            "C": self.C
        }

    def save_params(self, filename):
        d = self.get_params()
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
    epochs: int = 100,
    print_every: int = 10,
    loss_fn = "huber",
    bin_size = 20,
    up_factor = 10,
    ws = None
):
    if ws is None:
        ws = [1 for _ in range(len(Is_tr))]
    losses = []
    k, l = model.k, model.l
    for epoch in range(epochs):
        total_loss = 0
        for currents, firing_rates, w in zip(Is_tr, fs_tr, ws):
            pred_fs = firing_rates[:max(k, l)]
            loss = 0
            for i in range(max(k, l), len(currents)):
                # up-weight loss for non-zero firing rate
                m = up_factor if firing_rates[i] > 0 else 1
                currs = currents[i-k:i+1]
                fs = pred_fs[i-l:i]
                #print(pred_fs, fs, model.b)
                f = model(currs, fs)
                pred_fs = torch.cat((pred_fs, f.reshape(1)))

                if loss_fn == "poisson":
                    loss += m * criterion(f * bin_size, firing_rates[i] * bin_size)
                else:
                    loss += m * criterion(f, firing_rates[i])
                
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1) # prevent gradient explosion
            optimizer.step()
            total_loss += w * loss.item()

        losses.append(total_loss)
        if (epoch+1) % print_every == 0:
            print(f"Epoch {epoch+1} / Loss: {total_loss}")
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