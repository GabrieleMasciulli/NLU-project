import torch
from torch.optim.optimizer import Optimizer
import copy


class NTAvSGD(Optimizer):
    """Implements Non-monotonically Triggered Averaged Stochastic Gradient Descent (NT-AvSGD).

    Based on the paper 'Regularizing and Optimizing LSTM Language Models'.
    This implementation assumes the trigger condition (based on validation performance)
    is checked externally in the training loop. The optimizer performs SGD steps
    until `start_averaging()` is called, after which it maintains and updates
    an average of the parameters.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        n (int, optional): non-monotonic interval for the external trigger condition (default: 5)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        t0 (int, optional): point at which to start averaging (only relevant if averaging is started manually, default: 0)
    """

    def __init__(self, params, lr=1e-2, n=5, weight_decay=0, t0=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= weight_decay:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))
        if not 0 <= n:
            raise ValueError("Invalid non-monotonic interval n: {}".format(n))

        defaults = dict(lr=lr, n=n, weight_decay=weight_decay, t0=t0)
        super(NTAvSGD, self).__init__(params, defaults)

        # State initialization (will be populated in the first step)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['T'] = None  # Step k when averaging was triggered
                state['ax'] = None  # Averaged parameters

    def start_averaging(self):
        """
        Call this method from the training loop when the trigger condition is met.
        Initializes the averaged parameters `ax` with the current parameters
        and records the current step `k` as the trigger point `T`.
        """
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if state['T'] is None:  # Only trigger once
                    print(
                        f"NTAvSGD: Starting averaging at step {state['step']}")
                    state['T'] = state['step']
                    state['ax'] = torch.zeros_like(p.data)
                    state['ax'].copy_(p.data)

    def swap_parameters(self):
        """
        Swaps the current parameters with the averaged parameters `ax`.
        Should be called before evaluation if averaging has started.
        Returns the original parameters for swapping back later.
        """
        original_params = {}
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if state['T'] is not None:  # Only swap if averaging has started
                    # Store original param before overwriting
                    original_params[p] = p.data.clone()
                    # Swap current param with averaged param
                    p.data.copy_(state['ax'])
        return original_params

    def load_original_params(self, original_params):
        """
        Restores the original parameters saved by `swap_parameters`.
        Should be called after evaluation if parameters were swapped.
        """
        for p, original_data in original_params.items():
            p.data.copy_(original_data)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                # Increment step counter
                state['step'] += 1
                k = state['step']

                # Perform standard SGD step
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                p.data.add_(grad, alpha=-lr)

                # Update averaged parameters if averaging has been triggered
                if state['T'] is not None:
                    T = state['T']
                    if state['ax'] is None:  # Should have been initialized by start_averaging
                        raise RuntimeError(
                            "Optimizer averaging triggered but 'ax' is not initialized.")
                    # Update the average using the formula: ax_k = ax_{k-1} + (w_k - ax_{k-1}) / (k - T + 1)
                    # Avoid division by zero if T == k (first step after trigger)
                    if k > T:
                        state['ax'].add_(p.data.sub(
                            state['ax']).div(k - T + 1))
                    # If k == T, ax was just initialized to p.data, so no update needed yet.

        return loss

    # Override state dict handling to include 'ax'
    def state_dict(self):
        state_dict = super().state_dict()
        # Move 'ax' tensors to CPU before saving
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if state.get('ax') is not None:
                    state['ax'] = state['ax'].cpu()
        state_dict['param_groups'] = copy.deepcopy(
            self.param_groups)  # Ensure defaults like 'n' are saved
        return state_dict

    def load_state_dict(self, state_dict):
        # Move 'ax' tensors back to the correct device after loading
        super().load_state_dict(state_dict)
        # Get device from first param
        device = self.param_groups[0]['params'][0].device
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if state.get('ax') is not None:
                    state['ax'] = state['ax'].to(device)
