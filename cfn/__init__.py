"""Conservative Flux Network (CFN) for learning 1D conservation-law fluxes."""

from cfn.sampling import build_epoch_dataset
from cfn.models import CFN, Flux, ResidualBlock
from cfn.padding import input_circular_padding, input_nonperiodic_padding
from cfn.solvers import euler_numpy, rhs_flux_numpy, tvd_rk3_numpy
from cfn.theoretical import build_flux, flux_function
from cfn.training import apply_model, train_CFN, train_epoch

__all__ = [
    "CFN",
    "Flux",
    "ResidualBlock",
    "apply_model",
    "build_epoch_dataset",
    "build_flux",
    "euler_numpy",
    "flux_function",
    "input_circular_padding",
    "input_nonperiodic_padding",
    "rhs_flux_numpy",
    "train_CFN",
    "train_epoch",
    "tvd_rk3_numpy",
]
__version__ = "0.1.0"
