import torch
import torch.nn.functional as F
import numpy as np

class BatchGFR(torch.nn.Module):
    def __init__(self, models, freeze_g=True, device=None):
        super().__init__()
        self.device = device
        self.n_models = len(models)
        self.g = BatchPolynomialActivation([model.g for model in models])
        self.bin_size = [model.bin_size for model in models]
        
        self.ds = torch.nn.Parameter(models[0].ds.detach().cpu(), requires_grad=False)
        self.n_hidden = len(self.ds)
        
        # [n_models, n_hidden]
        self.a = torch.nn.Parameter(torch.cat([model.a.detach().cpu() for model in models], dim=0))
        self.b = torch.nn.Parameter(torch.cat([model.b.detach().cpu() for model in models], dim=0))

        if freeze_g: self.g.freeze_parameters()
    
    def reset(self, batch_size):
        self.v = torch.zeros(batch_size, self.n_models, self.n_hidden).to(self.device)
        self.fs = torch.zeros(batch_size, self.n_models).to(self.device)
    
    # currents shape [B, n_models]
    def forward(self, currents):
        x = torch.einsum("ij,jk->ijk", currents, self.a) # shape [B, n_models, n_hidden]
        y = 1000 * torch.einsum("ij,jk->ijk", self.fs, self.b) # shape [B, n_models, n_hidden]
        self.v = torch.einsum("k,ijk->ijk", 1 - self.ds, self.v) + x + y # shape [B, n_models, n_hidden]
        self.fs = self.g(torch.mean(self.v, dim=2))
        return self.fs # shape [B, n_models]
    
    def freeze_parameters(self):
        for _, p in self.named_parameters():
            p.requires_grad = False
            
    def unfreeze_parameters(self): # problematic
        for _, p in self.named_parameters():
            p.requires_grad = True
    
class GFR(torch.nn.Module):
    def __init__(
        self, 
        g, # activation function
        ds,
        bin_size,
        freeze_g = True,
        device = None
    ):
        super().__init__()
        self.g = g
        self.bin_size = bin_size
        self.device = device
        
        self.ds = torch.nn.Parameter(ds.clone().detach(), requires_grad=False)
        self.n = len(self.ds)
        
        a = torch.zeros(self.n)
        a[0] = self.n
        b = torch.zeros(self.n)
        self.a = torch.nn.Parameter(a.reshape(1, self.n))
        self.b = torch.nn.Parameter(b.reshape(1, self.n))
        
        if freeze_g: self.g.freeze_parameters()
            
    
    # outputs a tensor of shape [B, 1], firing rate predictions at time t
    def forward(
        self,
        currents # shape [B, 1], currents for time t
    ):
        x = torch.einsum("ij,jk->ijk", currents, self.a) # shape [B, n_models, n_hidden]
        y = 1000 * torch.einsum("ij,jk->ijk", self.fs, self.b) # shape [B, n_models, n_hidden]
        self.v =  torch.einsum("k,ijk->ijk", 1 - self.ds, self.v) + x + y # shape [B, n_models, n_hidden]
        self.fs = self.g(torch.mean(self.v, dim=2))
        return self.fs # shape [B, n_models]
    
    def reset(self, batch_size):
        self.v = torch.zeros(batch_size, 1, self.n).to(self.device)
        self.fs = torch.zeros(batch_size, 1).to(self.device)
    
    def reg(self, p=1):
        return self.a.norm(p=p) + self.b.norm(p=p)

    @classmethod
    def default(cls, freeze_g=True, device=None):
        g = PolynomialActivation.default()
        ds = torch.tensor([1.0000, 0.6321, 0.3935, 0.1813])
        a = torch.zeros(ds.shape[0])
        a[0] = ds.shape[0]
        b = torch.zeros(ds.shape[0])
        model = cls(g, ds, 1, freeze_g=freeze_g, device=device)
        model.a = torch.nn.Parameter(a.unsqueeze(dim=0))
        model.b = torch.nn.Parameter(b.unsqueeze(dim=0))
        return model

    @classmethod
    def from_params(cls, params, freeze_g=True, device=None):
        g = PolynomialActivation.from_params(params["g"])
        model = cls(
            g, 
            torch.tensor(params["ds"]), 
            params["bin_size"], 
            freeze_g=freeze_g, 
            device=device
        )
        model.a = torch.nn.Parameter(torch.tensor(params["a"]))
        model.b = torch.nn.Parameter(torch.tensor(params["b"]))
        return model

    def get_params(self):
        return {
            "a": self.a.tolist(),
            "b": self.b.tolist(),
            "g": self.g.get_params(),
            "ds": self.ds.tolist(),
            "bin_size": self.bin_size
        }
    
    def freeze_parameters(self):
        for _, p in self.named_parameters():
            p.requires_grad = False
            
    def unfreeze_parameters(self): # problematic
        for _, p in self.named_parameters():
            p.requires_grad = True
    
    # Is: shape [seq_length]
    def predict(self, Is):
        pred_fs = []
        vs = []
        
        with torch.no_grad():
            self.reset(1)
            for i in range(len(Is)):
                f = self.forward(Is[i].reshape(1, 1)).reshape(1)
                vs.append(self.v.clone().reshape(1, -1))
                pred_fs.append(f.clone())
        return torch.stack(pred_fs).squeeze(), torch.stack(vs).squeeze()
    
class BatchPolynomialActivation(torch.nn.Module):
    def __init__(self, gs):
        super().__init__()
        self.n = len(gs) # out dim
        self.degree = max([g.degree for g in gs])
        self.max_current = torch.nn.Parameter(torch.tensor([g.max_current for g in gs]), requires_grad=False)
        self.max_firing_rate = torch.nn.Parameter(torch.tensor([g.max_firing_rate for g in gs]), requires_grad=False)
        self.bin_size = [g.bin_size for g in gs]
        self.p = torch.nn.Parameter(torch.tensor([d for d in range(self.degree + 1)]), requires_grad=False)
        self.b = torch.nn.Parameter(torch.tensor([g.b.item() for g in gs]))
        
        poly_coeff = torch.zeros(self.n, self.degree + 1)
        for i, g in enumerate(gs):
            poly_coeff[i,:g.degree+1] = g.poly_coeff.detach().cpu()
        self.poly_coeff = torch.nn.Parameter(poly_coeff)
        
    # z: shape [B, n]
    def forward(self, z):
        x = (z - self.b) / self.max_current # shape [B, n]
        poly = torch.einsum("ijk,jk->ij", x.unsqueeze(dim=2).pow(self.p.reshape(1, 1, -1)), self.poly_coeff ** 2) # shape [B, n]
        tan = self.max_firing_rate * F.tanh(poly) # ceil is the max firing rate
        return F.relu(tan).to(torch.float32) # shape [B, n]
    
    def freeze_parameters(self):
        for _, p in self.named_parameters():
            p.requires_grad = False
            
    def unfreeze_parameters(self):
        for _, p in self.named_parameters():
            p.requires_grad = True

    def get_params(self):
        return {
            "max_current": self.max_current.tolist(),
            "max_firing_rate": self.max_firing_rate.tolist(),
            "poly_coeff": self.poly_coeff.tolist(),
            "b": self.b.tolist(),
            "bin_size": self.bin_size
        }

class PolynomialActivation(torch.nn.Module):
    def __init__(self, degree, max_current, max_firing_rate, bin_size):
        super().__init__()
        self.degree = degree
        self.max_current = torch.nn.Parameter(torch.tensor(max_current).reshape(1), requires_grad=False)
        self.max_firing_rate = torch.nn.Parameter(torch.tensor(max_firing_rate).reshape(1), requires_grad=False)
        self.bin_size = bin_size
        
        self.p = torch.nn.Parameter(torch.tensor([d for d in range(degree+1)]), requires_grad=False)
        self.poly_coeff = torch.nn.Parameter(torch.randn(1, self.degree + 1))
        self.b = torch.nn.Parameter(torch.tensor([0.0]))
    
    # z: shape [B, 1]
    def forward(self, z):
        x = (z - self.b) / self.max_current # shape [B, n=1]
        poly = torch.einsum("ijk,jk->ij", x.unsqueeze(dim=2).pow(self.p.reshape(1, 1, -1)), self.poly_coeff ** 2) # shape [B, n]
        tan = self.max_firing_rate * F.tanh(poly) # ceil is the max firing rate
        return F.relu(tan).to(torch.float32) # shape [B, n]
    
    @classmethod
    def from_params(cls, params):
        poly_coeff = torch.nn.Parameter(torch.tensor(params["poly_coeff"]))
        degree = poly_coeff.shape[1] - 1
        max_current = params["max_current"]
        max_firing_rate = params["max_firing_rate"]
        bin_size = params["bin_size"]
        g = cls(degree, max_current, max_firing_rate, bin_size)
        g.poly_coeff = poly_coeff
        g.b = torch.nn.Parameter(torch.tensor(params["b"]))
        return g
    
    @classmethod
    def default(cls):
        poly_coeff = torch.nn.Parameter(torch.tensor([0.0, 1.0]))
        degree = 1
        max_current = 1
        max_firing_rate = 1
        bin_size = 1
        g = cls(degree, max_current, max_firing_rate, bin_size)
        g.poly_coeff = poly_coeff
        g.b = torch.nn.Parameter(torch.tensor(0.0))
        return g

    def get_params(self):
        return {
            "max_current": self.max_current.tolist(),
            "max_firing_rate": self.max_firing_rate.tolist(),
            "poly_coeff": self.poly_coeff.tolist(),
            "b": self.b.tolist(),
            "bin_size": self.bin_size
        }
    
    def freeze_parameters(self):
        for _, p in self.named_parameters():
            p.requires_grad = False
            
    def unfreeze_parameters(self):
        for _, p in self.named_parameters():
            p.requires_grad = True

class LSTM(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(1, hidden_size, 1, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, 1)

    def forward(self, Is):
        # Is has shape [B, seq_len]
        out, _ = self.lstm(Is.unsqueeze(-1)) # [B, L, hidden_size]
        fs_pred = F.relu(self.fc(out)).squeeze(dim=2)  # [B, L, 1] so squeeze
        return fs_pred