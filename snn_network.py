import torch
import snntorch as snn
from snntorch import surrogate


class SNNNetwork(torch.nn.Module):
    """SNN baseline using LIF neurons with surrogate gradients.

    Architecture mirrors network.Network (GFR-RNN) exactly:
        fc1  : input  -> hidden  (feedforward)
        fc2  : hidden -> hidden  (recurrent)
        lif  : LIF spiking neuron layer
        fc3  : hidden -> output  (readout)

    The only difference is the neuron model:
        GFR-RNN  -> BatchGFR (multi-timescale, polynomial activation)
        SNN      -> snn.Leaky LIF (single beta decay, surrogate gradient)
    """

    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        beta=0.95,
        num_readout_steps=1,
        device=None,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.device = device
        self.num_readout_steps = num_readout_steps

        spike_grad = surrogate.atan()

        self.fc1 = torch.nn.Linear(in_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.lif = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc3 = torch.nn.Linear(hidden_dim, out_dim)

    def reset(self, batch_size):
        device = self.fc1.weight.device
        dtype = self.fc1.weight.dtype
        self.spk = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
        self.mem = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)

    def zero_input(self, batch_size):
        return torch.zeros(batch_size, self.in_dim, device=self.fc1.weight.device, dtype=self.fc1.weight.dtype)

    def _advance_from_current(self, input_current):
        recurrent_current = self.fc2(self.spk)
        current = input_current + recurrent_current
        self.spk, self.mem = self.lif(current, self.mem)
        return self.fc3(self.spk)

    def forward_zero_input(self):
        return self.forward(self.zero_input(self.spk.shape[0]))

    def forward_sequence(self, sequence):
        batch_size, n_steps, _ = sequence.shape
        self.reset(batch_size)
        input_current = self.fc1(sequence.reshape(batch_size * n_steps, self.in_dim))
        input_current = input_current.reshape(batch_size, n_steps, self.hidden_dim)
        for step_current in input_current.unbind(dim=1):
            self._advance_from_current(step_current)
        return self.forward_zero_input()

    # x: [batch_size, in_dim]  — identical call signature to Network.forward
    def forward(self, x):
        input_current = self.fc1(x)
        return self._advance_from_current(input_current)


class SNNNetworkSynaptic(torch.nn.Module):
    """SNN baseline using Synaptic (2nd-order) LIF neurons.

    This adds a synaptic current state on top of the membrane potential,
    giving the neuron two time-constants — closer in spirit to the
    multi-timescale dynamics of GFR neurons.
    """

    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        alpha=0.9,
        beta=0.95,
        device=None,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.device = device

        spike_grad = surrogate.atan()

        self.fc1 = torch.nn.Linear(in_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.lif = snn.Synaptic(alpha=alpha, beta=beta, spike_grad=spike_grad)
        self.fc3 = torch.nn.Linear(hidden_dim, out_dim)

    def reset(self, batch_size):
        device = self.fc1.weight.device
        dtype = self.fc1.weight.dtype
        self.spk = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
        self.syn = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
        self.mem = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)

    def zero_input(self, batch_size):
        return torch.zeros(batch_size, self.in_dim, device=self.fc1.weight.device, dtype=self.fc1.weight.dtype)

    def _advance_from_current(self, input_current):
        recurrent_current = self.fc2(self.spk)
        current = input_current + recurrent_current
        self.spk, self.syn, self.mem = self.lif(current, self.syn, self.mem)
        return self.fc3(self.spk)

    def forward_zero_input(self):
        return self.forward(self.zero_input(self.spk.shape[0]))

    def forward_sequence(self, sequence):
        batch_size, n_steps, _ = sequence.shape
        self.reset(batch_size)
        input_current = self.fc1(sequence.reshape(batch_size * n_steps, self.in_dim))
        input_current = input_current.reshape(batch_size, n_steps, self.hidden_dim)
        for step_current in input_current.unbind(dim=1):
            self._advance_from_current(step_current)
        return self.forward_zero_input()

    def forward(self, x):
        input_current = self.fc1(x)
        return self._advance_from_current(input_current)
