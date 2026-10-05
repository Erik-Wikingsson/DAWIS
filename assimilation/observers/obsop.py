import torch

"""Base class for ObsOp object

Input is a StateVector from a Model, returns ObsVector
"""


class ObsOp():
    """Base class for ObsOp objects

    Attributes:
        observe (function): Operator that takes state_vector and outputs new obs_vector.
    """

    def __init__(self, **kwargs):
        pass

    def observe(self, state_vector):
        """Apply observation operator to state_vector

        Args:
            state_vector (StateVector): StateVector object from a Model.

        Returns:
            obs_vector (ObsVector): ObsVector object after applying observation operator.
        """
        raise NotImplementedError("Subclasses should implement this method.")


# NOTE: Don't include the obs_mask here
# NOTE: The specific obs_fns are chosen to have the range match that of x.
OBS_FNS = {
    'linear': lambda x: x,
    'arctan': lambda x: torch.arctan(x),
    'arctan_scaled': lambda x: 15*torch.arctan(x/7),
    'abs': lambda x: torch.abs(x),
    'square': lambda x: torch.square(x),
    'square_scaled': lambda x: torch.square(x/7),
    'exp': lambda x: torch.exp(x),
    'exp_scaled': lambda x: torch.exp(x/7),
    'log_abs_scaled': lambda x: 6*torch.log(torch.abs(x)+0.1),
}
