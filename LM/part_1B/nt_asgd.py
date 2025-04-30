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
                    # Initialize ax only if it doesn't exist or needs re-initialization
                    if state.get('ax') is None:
                         state['ax'] = torch.zeros_like(p.data)
                    state['ax'].copy_(p.data) # Copy current params to start average

    def is_averaging(self):
        """
        Checks if the optimizer is currently in the averaging phase.
        Returns True if averaging has been triggered (state['T'] is not None), False otherwise.
        """
        # Check the state of the first parameter in the first group is sufficient
        # as averaging starts for all parameters simultaneously.
        if not self.param_groups:
            return False # No parameters to optimize
        first_param = self.param_groups[0]['params'][0]
        state = self.state[first_param]
        return state.get('T') is not None

    def swap_parameters(self, model_override=None):
        """
        Swaps the current parameters with the averaged parameters `ax`.
        Should be called before evaluation if averaging has started.
        Returns the original parameters for swapping back later.
        Optionally accepts a model instance to swap its parameters instead of the optimizer's default ones.
        """
        original_params = {}
        target_model = model_override if model_override is not None else None # Use override if provided

        # Determine the parameters to swap
        params_to_swap = []
        if target_model:
            params_to_swap = target_model.parameters()
        else:
            # Default: use parameters from the optimizer's param_groups
            for group in self.param_groups:
                params_to_swap.extend(group['params'])

        for p in params_to_swap:
            if p not in self.state:
                # If using model_override, some params might not be in optimizer state (e.g., frozen layers)
                # Or if the state wasn't properly initialized.
                # print(f"Warning: Parameter not found in optimizer state during swap. Skipping.")
                continue

            state = self.state[p]
            if state.get('T') is not None and state.get('ax') is not None:  # Only swap if averaging and ax exists
                # Store original param before overwriting
                original_params[p] = p.data.clone()
                # Swap current param with averaged param
                p.data.copy_(state['ax'])
            # else:
                # print(f"Debug: Not swapping param. T={state.get('T')}, ax_exists={state.get('ax') is not None}")


        # If no parameters were swapped (e.g., averaging not started or ax not ready), return empty dict
        if not original_params and self.is_averaging():
             print("Warning: swap_parameters called while averaging, but no parameters were swapped. 'ax' might not be initialized yet.")

        return original_params


    def load_original_params(self, original_params, model_override=None):
        """
        Restores the original parameters saved by `swap_parameters`.
        Should be called after evaluation if parameters were swapped.
        Optionally accepts a model instance to restore its parameters.
        """
        target_model = model_override if model_override is not None else None

        # Determine the parameters to restore (must match those swapped)
        params_to_restore = original_params.keys() # Use keys from the dict passed in

        for p in params_to_restore:
            if p in original_params:
                 p.data.copy_(original_params[p])
            # else: # This case should ideally not happen if original_params is correct
            #     print(f"Warning: Parameter {p} found in model but not in original_params dict during restore.")


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
