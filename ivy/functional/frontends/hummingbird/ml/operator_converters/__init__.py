# Register constants used within Hummingbird converters.
from . import constants as converter_constants
from .. import supported as hummingbird_constants
from .._utils import _Constants


# Add constants in scope.
constants = _Constants(converter_constants, hummingbird_constants)

from .sklearn import decision_tree