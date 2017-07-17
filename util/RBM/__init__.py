from .bbrbm import BBRBM
from .gbrbm import GBRBM
from .bgrbm import BGRBM


# default RBM
RBM = BBRBM

__all__ = [RBM, BBRBM, GBRBM, BGRBM]
