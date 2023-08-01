import pickle
import torch
import torch.nn.functional as F
import numpy as np

# assume ds the same for all models, otherwise composition becomes quite complicated
class BatchEKFR(torch.nn.Module):
    def __init__(self, models, freeze_g=True, device=None):
        super().__init__()
        self.device = device
        self.n_models = len(models)
        self.g = BatchPolynomialActivation([model.g for model in models])
        self.bin_size = [model.bin_size for model in models]
        
        self.ds = torch.nn.Parameter(models[0].ds.detach().cpu(), requires_grad=False)
        self.n_hidden = len(self.ds)
        
        # [n_models, n_hidden]
        self.a = torch.nn.Parameter(torch.stack([model.a.detach().cpu() for model in models]))
        self.b = torch.nn.Parameter(torch.stack([model.b.detach().cpu() for model in models]))
        self.w = torch.nn.Parameter(torch.stack([model.w.detach().cpu() for model in models])).reshape(self.n_models, self.n_hidden)
        
        if freeze_g: self.g.freeze_parameters()
    
    def reset(self, batch_size):
        self.v = torch.zeros(batch_size, self.n_models, self.n_hidden).to(self.device)
        self.fs = torch.zeros(batch_size, self.n_models).to(self.device)
    
    # currents shape [B, n_models]
    def forward(self, currents):
        x = torch.einsum("ij,jk->ijk", currents, self.a) # shape [B, n_models, n_hidden]
        y = 1000 * torch.einsum("ij,jk->ijk", self.fs, self.b) # shape [B, n_models, n_hidden]
        self.v =  torch.einsum("k,ijk->ijk", 1 - self.ds, self.v) + x + y # shape [B, n_models, n_hidden]
        self.fs = self.g(torch.einsum("ijk,jk->ij", self.v, self.w))
        return self.fs # shape [B, n_models]
    
    def freeze_parameters(self):
        for _, p in self.named_parameters():
            p.requires_grad = False
            
    def unfreeze_parameters(self):
        for _, p in self.named_parameters():
            p.requires_grad = True
    
class ExponentialKernelFiringRateModel(torch.nn.Module):
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
        self.a = torch.nn.Parameter(torch.ones(self.n) + torch.randn(self.n) * 0.001)
        self.b = torch.nn.Parameter(torch.randn(self.n) * 0.001)
        self.w = torch.nn.Parameter((torch.randn(self.n) * 0.001 + 1) / self.n).reshape(-1, 1).to(device)
        
        if freeze_g: self.g.freeze_parameters()
        
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
    
    @classmethod
    def from_params(cls, params, freeze_g=True, device=None):
        g = PolynomialActivation.from_params(params["g"])
        model = cls(g, params["ds"], params["bin_size"], freeze_g=freeze_g, device=device)
        model.a = torch.nn.Parameter(params["a"])
        model.b = torch.nn.Parameter(params["b"])
        model.w = torch.nn.Parameter(params["w"])
        return model

    def get_params(self):
        return {
            "a": self.a.detach().cpu(),
            "b": self.b.detach().cpu(),
            "g": self.g.get_params(),
            "w": self.w.detach().cpu(),
            "ds": self.ds.detach().cpu(),
            "bin_size": self.bin_size
        }
    
    def freeze_parameters(self):
        for _, p in self.named_parameters():
            p.requires_grad = False
            
    def unfreeze_parameters(self):
        for _, p in self.named_parameters():
            p.requires_grad = True
            
    def kernel(self, x, var="a"):
        a = self.a if var == "a" else self.b
        return torch.sum(self.w * a * torch.pow(self.ds, x))
    
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
            poly_coeff[i, :g.degree+1] = g.poly_coeff.detach().cpu()
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
    
class PolynomialActivation(torch.nn.Module):
    def __init__(self, degree, max_current, max_firing_rate, bin_size):
        super().__init__()
        self.degree = degree
        self.max_current = max_current
        self.max_firing_rate = max_firing_rate
        self.bin_size = bin_size
        
        self.p = torch.nn.Parameter(torch.tensor([d for d in range(degree+1)]), requires_grad=False)
        self.poly_coeff = torch.nn.Parameter(torch.randn(self.degree + 1))
        self.b = torch.nn.Parameter(torch.tensor(0.0))
    
    # z: shape [B, 1]
    def forward(self, z):
        x = (z - self.b) / self.max_current # shape [B, 1]
        poly = x.pow(self.p) @ (self.poly_coeff ** 2).reshape(-1, 1) # shape [B, degree]
        tan = self.max_firing_rate * F.tanh(poly) # ceil is the max firing rate
        return F.relu(tan) # shape [B, 1]
    
    # initialize based on linear approximation of data
    @classmethod
    def from_data(cls, degree, max_current, max_firing_rate, bin_size, Is, fs):
        g = cls(degree, max_current, max_firing_rate, bin_size)
        
        x1, x2, y1, y2 = tuple([torch.tensor(0.0)] * 4)
        xs, ys = map(list, zip(*sorted(zip(Is.cpu(), fs.cpu()), key=lambda x: x[0])))
        i = np.argmax(ys)
        x2, y2 = xs[i], ys[i]
        for i in range(0, len(ys)):
            if ys[i] > 0.01:
                x1, y1 = (xs[i-1], ys[i-1]) if i - 1 > 0 else (xs[i], ys[i])
                break
                
        g.b = torch.nn.Parameter(x1.clone())
        poly_coeff = torch.randn(degree + 1) * 1e-1
        poly_coeff[1] = np.abs((y2 - y1) / (x2 - x1) * max_current)
        g.poly_coeff = torch.nn.Parameter(poly_coeff)
        
        return g
    
    @classmethod
    def from_params(cls, params):
        poly_coeff = torch.nn.Parameter(params["poly_coeff"])
        degree = len(poly_coeff) - 1
        max_current = params["max_current"]
        max_firing_rate = params["max_firing_rate"]
        bin_size = params["bin_size"]
        
        g = cls(degree, max_current, max_firing_rate, bin_size)
        g.poly_coeff = poly_coeff
        g.b = torch.nn.Parameter(params["b"])
        
        return g

    def get_params(self):
        return {
            "max_current": self.max_current,
            "max_firing_rate": self.max_firing_rate,
            "poly_coeff": self.poly_coeff.detach().cpu(),
            "b": self.b.detach().cpu(),
            "bin_size": self.bin_size
        }
    
    def freeze_parameters(self):
        for _, p in self.named_parameters():
            p.requires_grad = False
            
    def unfreeze_parameters(self):
        for _, p in self.named_parameters():
            p.requires_grad = True