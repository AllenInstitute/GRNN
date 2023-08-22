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
        self.k = max(model.k for model in models)
        self.l = max(model.l for model in models)
        
        # [n_models, k (or l)]
        a = torch.nn.utils.rnn.pad_sequence([model.a.detach().cpu().flip(dims=(0,)) for model in models]).flip(dims=(0,)).T
        b = torch.nn.utils.rnn.pad_sequence([model.b.detach().cpu().flip(dims=(0,)) for model in models]).flip(dims=(0,)).T
        self.a = torch.nn.Parameter(a)
        self.b = torch.nn.Parameter(b)
        
        if freeze_g: self.g.freeze_parameters()
    
    def reset(self, batch_size):
        self.currents = torch.zeros(batch_size, self.n_models, self.k).to(self.device)
        self.fs = torch.zeros(batch_size, self.n_models, self.l).to(self.device)
    
    # currents shape [B, n_models]
    def forward(self, currents):
        self.currents = torch.cat([self.currents, currents.unsqueeze(dim=2)], dim=2)[:,:,1:]
        x = torch.einsum("ijk,jk->ij", self.currents, self.a) # shape [B, n_models]
        y = 1000 * torch.einsum("ijk,jk->ij", self.fs, self.b) # shape [B, n_models]
        f = self.g(x + y) # shape [B, n_models]
        self.fs = torch.cat([self.fs, f.unsqueeze(dim=2)], dim=2)[:,:,1:]
        return f
    
    def freeze_parameters(self):
        for _, p in self.named_parameters():
            p.requires_grad = False
            
    def unfreeze_parameters(self):
        for _, p in self.named_parameters():
            p.requires_grad = True
            
    def smoothness_reg(self):
        a = torch.cat([torch.zeros(self.n_models, 1), self.a], dim=1) # shape [n_models, k+1]
        b = torch.cat([torch.zeros(self.n_models, 1), self.b], dim=1) # shape [n_models, l+1]
        i = torch.arange(self.k, 0, -1).to(torch.float32) # shape [k]
        j = torch.arange(self.l, 0, -1).to(torch.float32) # shape [l]
        smooth_a = torch.mean(torch.einsum("ij,j->i", torch.diff(a) ** 2, i ** 2)) / self.k
        smooth_b = torch.mean(torch.einsum("ij,j->i", torch.diff(b) ** 2, j ** 2)) / self.l
        return smooth_a + smooth_b

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
        self.v =  torch.einsum("k,ijk->ijk", 1 - self.ds, self.v) + x + y # shape [B, n_models, n_hidden]
        self.fs = self.g(torch.mean(self.v, dim=2))
        return self.fs # shape [B, n_models]
    
    def freeze_parameters(self):
        for _, p in self.named_parameters():
            p.requires_grad = False
            
    def unfreeze_parameters(self): # problematic
        for _, p in self.named_parameters():
            p.requires_grad = True
            
    def get_params(self):
        return {
            "a": self.a.detach().cpu(),
            "b": self.b.detach().cpu(),
            "g": self.g.get_params(),
            "ds": self.ds.detach().cpu(),
            "bin_size": self.bin_size
        }
    
    def kernel(self, x, var="a"):
        a = self.a if var == "a" else self.b
        return torch.einsum("ij,j->i", a, torch.pow(1 - self.ds, x))
    
class GeneralizedFiringRateModel(torch.nn.Module):
    def __init__(
        self, 
        g, # activation function
        k: int, # number of previous timesteps for current I
        l: int, # number of timesteps for firing rate
        bin_size = 0,
        freeze_g: bool = True,
        device = None
    ):
        super().__init__()
        self.g = g
        self.k = k
        self.l = l
        self.a = torch.nn.Parameter(torch.zeros(k))
        self.b = torch.nn.Parameter(torch.zeros(l))
        self.bin_size = bin_size
        self.device = device
        
        # freeze activation parameters
        if freeze_g: g.freeze_parameters()
        
    '''
    Input:
    currents: Tensor([batch_size])
    ---
    Output:
    f: Tensor([batch_size])
    '''
    def forward(self, currents):
        self.currents = torch.cat([self.currents, currents.reshape(-1, 1)], dim=1)[:,1:]
        x = torch.einsum("j,ij->i", self.a, self.currents)
        y = 1000 * torch.einsum("j,ij->i", self.b, self.fs)
        f = self.g(x + y)
        self.fs = torch.cat([self.fs, f.reshape(-1, 1)], dim=1)[:,1:]
        return f
    
    def reset(self, batch_size):
        self.currents = torch.zeros(batch_size, self.k).to(self.device)
        self.fs = torch.zeros(batch_size, self.l).to(self.device)
    
    def smoothness_reg(self):
        a = torch.cat([torch.tensor([0.0]), self.a])
        b = torch.cat([torch.tensor([0.0]), self.b])
        i = torch.arange(len(self.a), 0, -1).to(torch.float32)
        j = torch.arange(len(self.b), 0, -1).to(torch.float32)
        smooth_a = (torch.diff(a) ** 2) @ (i ** 2) / len(self.a)
        smooth_b = (torch.diff(b) ** 2) @ (j ** 2) / len(self.b)
        return smooth_a + smooth_b
            
    @classmethod
    def from_params(cls, params, freeze_g=True):
        g = PolynomialActivation.from_params(params["g"])
        model = cls(g, len(model.a), len(model.b), params["bin_size"], freeze_g=freeze_g)
        model.a = torch.nn.Parameter(params["a"])
        model.b = torch.nn.Parameter(params["b"])
        model.k = len(model.a)
        model.l = len(model.b)
        return model
    
    @classmethod
    def from_ekfr(cls, ekfr_model, k, l, freeze_g=True):
        g = PolynomialActivation.from_params(ekfr_model.g.get_params())
        model = cls(g, k, l, ekfr_model.bin_size, freeze_g=freeze_g)
        a = torch.tensor([ekfr_model.kernel(i, var="a") for i in range(k-1, -1, -1)]).to(torch.float32)
        b = torch.tensor([ekfr_model.kernel(i, var="b") for i in range(l-1, -1, -1)]).to(torch.float32)
        model.a = torch.nn.Parameter(a)
        model.b = torch.nn.Parameter(b)
        return model

    def get_params(self):
        return {
            "a": self.a.detach().cpu(),
            "b": self.b.detach().cpu(),
            "g": self.g.get_params(),
            "bin_size": self.bin_size
        }
    
    def freeze_parameters(self):
        for _, p in self.named_parameters():
            p.requires_grad = False
            
    def unfreeze_parameters(self):
        for _, p in self.named_parameters():
            p.requires_grad = True
    
    '''
    Input:
    Is: Tensor([seq_len])
    '''
    def predict(self, Is):
        with torch.no_grad():
            self.reset(1)
            pred_fs = []
            for i in range(Is.shape[0]):
                pred_fs.append(self(Is[i]))
        return torch.stack(pred_fs), None
    
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
        
        temp = 10
        a = torch.exp(-temp * torch.arange(self.n))
        a = a / a.sum() * self.n
        
        b = torch.randn(self.n) * 0.001
        b[0:] = 0
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
    def from_params(cls, params, freeze_g=True, device=None):
        g = PolynomialActivation.from_params(params["g"])
        model = cls(g, params["ds"], params["bin_size"], freeze_g=freeze_g, device=device)
        model.a = torch.nn.Parameter(params["a"])
        model.b = torch.nn.Parameter(params["b"])
        return model

    def get_params(self):
        return {
            "a": self.a.detach().cpu(),
            "b": self.b.detach().cpu(),
            "g": self.g.get_params(),
            "ds": self.ds.detach().cpu(),
            "bin_size": self.bin_size
        }
    
    def freeze_parameters(self):
        for _, p in self.named_parameters():
            p.requires_grad = False
            
    def unfreeze_parameters(self): # problematic
        for _, p in self.named_parameters():
            p.requires_grad = True
            
    def kernel(self, x, var="a"):
        a = self.a if var == "a" else self.b
        return torch.sum(a * torch.pow(1-self.ds, x))
    
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
            "max_current": self.max_current.detach().cpu(),
            "max_firing_rate": self.max_firing_rate.detach().cpu(),
            "poly_coeff": self.poly_coeff.detach().cpu(),
            "b": self.b.detach().cpu(),
            "bin_size": self.bin_size
        }

class PolynomialActivation(torch.nn.Module):
    def __init__(self, degree, max_current, max_firing_rate, bin_size):
        super().__init__()
        self.degree = degree
        self.max_current = torch.nn.Parameter(torch.tensor([max_current]), requires_grad=False)
        self.max_firing_rate = torch.nn.Parameter(torch.tensor([max_firing_rate]), requires_grad=False)
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
                
        g.b = torch.nn.Parameter(x1.clone().reshape(1))
        poly_coeff = torch.randn(degree + 1) * 1e-1
        poly_coeff[1] = np.abs((y2 - y1) / (x2 - x1) * max_current)
        poly_coeff = poly_coeff.reshape(1, -1)
        g.poly_coeff = torch.nn.Parameter(poly_coeff)
        
        return g
    
    @classmethod
    def from_params(cls, params):
        poly_coeff = torch.nn.Parameter(params["poly_coeff"])
        degree = poly_coeff.shape[1] - 1
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