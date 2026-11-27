"""Array input and reproducible sampling types shared by the numerical APIs."""
from typing import Union
import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import spmatrix

MatrixInput = Union[ArrayLike, spmatrix]
Matrix = Union[np.ndarray, spmatrix]
Seed = Union[int, np.random.Generator, None]
