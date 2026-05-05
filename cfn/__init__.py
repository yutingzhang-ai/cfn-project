"""Conservative Flux Network (CFN) for learning 1D conservation-law fluxes."""

from cfn.models import CFN, Flux, ResidualBlock
from cfn.padding import input_circular_padding, input_nonperiodic_padding
from cfn.data import build_epoch_dataset
from cfn.training import apply_model, train_epoch, train_CFN
from cfn.theoretical import build_flux, flux_function
from cfn.solvers import rhs_flux_numpy, tvd_rk3_numpy, euler_numpy

__version__ = "0.1.0"
