import pickle
import torch
import torch.nn.functional as F
import numpy as np

from utils import compute_derivative

class FiringRateModel(torch.nn.Module):
    def __init__(
        self, 
        g, # activation function
        k: int = 0, # number of previous timesteps for current I
        l: int = 0, # number of timesteps for firing rate
        m: int = 0, # include current derivatives up to order m
        n: int = 0, # include firing rate derivatives up to order n
        bin_size = 0,
        static_g: bool = True
    ):
        super().__init__()
        self.g = g
        self.k = k
        self.l = l
        self.m = m
        self.n = n
        self.a = torch.nn.Parameter(torch.zeros(k))
        self.b = torch.nn.Parameter(torch.zeros(l))
        self.c = torch.nn.Parameter(torch.zeros(m))
        self.d = torch.nn.Parameter(torch.zeros(n))

        assert bin_size > 0
        self.bin_size = bin_size
        self.dt = bin_size / 1000
        
        # freeze activation parameters
        if static_g:
            for _, p in self.g.named_parameters():
                p.requires_grad = False
        
    def forward(
        self,
        currents, # currents tensor, up to time t
        fs # firing rates, up to time t-1
    ):
        dIs = torch.tensor([compute_derivative(currents, i, self.dt) for i in range(1, self.m+1)])
        dfs = torch.tensor([compute_derivative(fs, i, self.dt) for i in range(1, self.n+1)])
        x = 1000 * self.b @ (fs[-self.l:] if self.l > 0 else torch.tensor([]))
        y = self.c @ dIs
        z = self.d @ dfs
        w = x + y + z
        return self.g(currents[-1] + self.a @ currents[-self.k-1:-1] + w)
    
    def norm(self, p=1):
        return torch.sum(torch.tensor([torch.norm(x, p=p) for x in [self.a, self.b, self.c, self.d]]))

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
            "c": self.c.clone(),
            "d": self.d.clone(),
            "g": self.g.get_params() if callable(getattr(self.g, "get_params", None)) else []
        }

    def predict(self, Is, closed=True, fs=None):
        k, l, m, n = self.k, self.l, self.m, self.n
        pad = max(k, l, m, n)
        Is_pad = F.pad(Is, (pad, 0), "constant")
        
        if closed:
            with torch.no_grad():
                fs1 = torch.zeros(pad)
                pred_fs = []
                for i in range(pad, len(Is_pad)):
                    f = self.forward(Is_pad[:i+1], fs1[:i])
                    fs1 = torch.cat((fs1, f.reshape(1)))
                    pred_fs.append(f)
            return np.array([f.item() for f in pred_fs])
        else:
            with torch.no_grad():
                pred_fs = []
                for i in range(pad, len(Is_pad)):
                    f = self.forward(Is_pad[:i+1], fs[:i])
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