import pickle
import torch
import torch.nn.functional as F
import numpy as np

class FiringRateModel(torch.nn.Module):
    def __init__(
        self, 
        g, # activation function
        ds,
        bin_size = 20,
        device = None,
        freeze_g = True
    ):
        super().__init__()
        self.g = g
        self.bin_size = bin_size
        self.dt = bin_size / 1000
        self.device = device
        
        self.ds = torch.nn.Parameter(torch.tensor(ds).to(torch.float32), requires_grad=False)
        self.n = len(self.ds)
        self.a = torch.nn.Parameter(torch.ones(self.n) + torch.randn(self.n) * 0.001)
        self.b = torch.nn.Parameter(torch.randn(self.n) * 0.001)
        self.w = torch.nn.Parameter((torch.ones(self.n) + torch.randn(self.n) * 0.001) / self.n).reshape(-1, 1).to(device)
        
        # freeze activation parameters
        if freeze_g:
            for _, p in self.g.named_parameters():
                p.requires_grad = False
    
    # outputs a tensor of shape [B], firing rate predictions at time t
    def forward(
        self,
        currents, # shape [B], currents for time t
        fs # shape [B], firing rates for time t-1
    ):
        x = torch.outer(currents, self.a) # shape [B, n]
        y = 1000 * torch.outer(fs, self.b) # shape [B, n]
        self.v =  (1 - self.ds) * self.v + x + y # shape [B, n]
        return self.g(self.v @ self.w).reshape(-1) # shape [B]
    
    def reset(self, batch_size):
        self.v = torch.zeros(batch_size, self.n).to(self.device)

    def init_from_params(self, params, freeze_g=True):
        self.a = torch.nn.Parameter(params["a"])
        self.b = torch.nn.Parameter(params["b"])
        self.g = PolynomialActivation()
        self.g.init_from_params(params["g"])
        self.ds = torch.nn.Parameter(params["ds"], requires_grad=False)
        self.w = torch.nn.Parameter(params["w"])
        self.n = len(self.ds)
        
        if freeze_g:
            for _, p in self.g.named_parameters():
                p.requires_grad = False

    def get_params(self):
        return {
            "a": self.a.detach().cpu(),
            "b": self.b.detach().cpu(),
            "g": self.g.get_params(),
            "w": self.w.detach().cpu(),
            "ds": self.ds.detach().cpu()
        }

    # Is: shape [seq_length]
    def predict(self, Is):
        pred_fs = []
        vs = []
        f = torch.zeros(1).to(self.device)
        
        with torch.no_grad():
            self.reset(1)
            for i in range(len(Is)):
                f = self.forward(Is[i].reshape(1), f.reshape(1))
                vs.append(self.v.clone())
                pred_fs.append(f.clone())
        return torch.stack(pred_fs).squeeze(), torch.stack(vs).squeeze()

class PolynomialActivation(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    # z: shape [B, 1]
    def forward(self, z):
        x = (z - self.b) / self.max_current # shape [B, 1]
        poly = x.pow(self.p) @ (self.poly_coeff ** 2).reshape(-1, 1) # shape [B, degree]
        tan = self.max_firing_rate * F.tanh(poly) # ceil is the max firing rate
        return F.relu(tan) # shape [B, 1]
    
    # slightly ad hoc parameter initialization
    def init_params(self, bin_size, degree, max_current, max_firing_rate, Is, fs):
        self.bin_size = bin_size
        self.degree = degree # polynomial degree
        self.max_current = max_current # used for normalization
        self.max_firing_rate = max_firing_rate
        self.p = torch.nn.Parameter(torch.tensor([d for d in range(degree+1)]), requires_grad=False)
        
        x1, x2, y1, y2 = tuple([torch.tensor(0.0)] * 4)
        xs, ys = map(list, zip(*sorted(zip(Is.cpu(), fs.cpu()), key=lambda x: x[0])))
        i = np.argmax(ys)
        x2, y2 = xs[i], ys[i]
        for i in range(0, len(ys)):
            if ys[i] > 0.01:
                x1, y1 = (xs[i-1], ys[i-1]) if i - 1 > 0 else (xs[i], ys[i])
                break
        self.b = torch.nn.Parameter(x1.clone())
        self.poly_coeff = torch.randn(self.degree + 1) * 1e-1 # to make sure there is some gradient
        self.poly_coeff[1] = np.abs((y2 - y1) / (x2 - x1) * self.max_current) #* torch.abs(torch.randn(1)[0] * 7 + 15)
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
        self.bin_size = params["bin_size"]
        self.max_current = params["max_current"]
        self.max_firing_rate = params["max_firing_rate"]
        self.poly_coeff = torch.nn.Parameter(params["poly_coeff"])
        self.b = torch.nn.Parameter(params["b"])
        self.degree = len(self.poly_coeff) - 1
        self.p = torch.tensor([d for d in range(self.degree+1)])
        self.p = torch.nn.Parameter(self.p, requires_grad=False)
    
    def get_params(self):
        return {
            "max_current": self.max_current,
            "max_firing_rate": self.max_firing_rate,
            "poly_coeff": self.poly_coeff.detach().cpu(),
            "b": self.b.detach().cpu(),
            "bin_size": self.bin_size
        }

    def save_params(self, filename):
        d = self.get_params()
        with open(filename, 'wb') as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
def load_model(params, freeze_g=True):
    g = PolynomialActivation()
    g.init_from_params(params["g"])
    model = FiringRateModel(g, [])
    model.init_from_params(params, freeze_g=freeze_g)
    return model