import torch
from torch.optim.optimizer import Optimizer


class NTAvSGD(Optimizer):
    """Implements Non-monotonically Triggered Averaged Stochastic Gradient Descent (NT-AvSGD).

    Based on the paper 'Regularizing and Optimizing LSTM Language Models'.
    This implementation assumes the trigger condition (based on validation performance)
    is checked externally in the training loop. The optimizer performs
    traditional SGD steps until `start_averaging()` is called, after which it maintains and updates
    an average of the parameters.

    Args:
        - params (iterable): iterable of parameters to optimize
        - lr (float, optional): learning rate (default: 1e-2)
        - weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(self, params, lr=1e-2, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= weight_decay:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, weight_decay=weight_decay)
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
        This method will be called from the training loop when the trigger
        condition is met.
        It initializes the averaged parameters `ax` with the current parameters
        and records the current step `k` as the trigger point `T`.
        """
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if state['T'] is None:  # Ensures this is the first and only time averaging is triggered
                    print(
                        f"NTAvSGD: Starting averaging at step {state['step']}")
                    state['T'] = state['step']
                    # Initialize ax
                    if state.get('ax') is None:
                        state['ax'] = torch.zeros_like(p.data)
                    # Copy current params to start average
                    state['ax'].copy_(p.data)

    def is_averaging(self):
        """
        Checks if the optimizer is currently in the averaging phase.
        Returns True if averaging has been triggered (state['T'] is not None), False otherwise.
        """
        if not self.param_groups:
            return False  # No parameters to optimize
        first_param = self.param_groups[0]['params'][0]
        state = self.state[first_param]
        return state.get('T') is not None

    def swap_parameters(self, model):
        """
        Swaps the current parameters with the averaged parameters `ax`.
        Should be called before evaluation if averaging has started.
        Returns the original parameters for swapping back later.
        Accepts a model instance to swap its parameters with.
        """
        original_params = {}
        params_to_swap = model.parameters()

        for p in params_to_swap:
            if p not in self.state:
                # If parameter not in optimizer state (e.g., frozen layers)
                # Or if the state wasn't properly initialized.
                continue

            state = self.state[p]
            # Only swap if averaging and ax exists
            if self.is_averaging() and state.get('ax') is not None:
                # Store original param before overwriting
                original_params[p] = p.data.clone()
                # Swap current param with averaged param
                p.data.copy_(state['ax'])

        # If no parameters were swapped (e.g., averaging not started or ax not ready), return empty dict
        if not original_params and self.is_averaging():
            print("Warning: swap_parameters called while averaging, but no parameters were swapped. 'ax' might not be initialized yet.")

        return original_params

    def load_original_params(self, original_params):
        """
        Restores the original parameters saved by `swap_parameters`.
        Should be called after evaluation if parameters were swapped.
        """
        params_to_restore = original_params.keys()

        for p in params_to_restore:
            if p in original_params:
                p.data.copy_(original_params[p])

    @torch.no_grad()
    def step(self):
        """Performs a single optimization step."""
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
                    if k >= T:
                        state['ax'].add_(p.data.sub(
                            state['ax']).div(k - T + 1))
