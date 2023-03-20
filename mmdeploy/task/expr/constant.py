# Copyright (c) OpenMMLab. All rights reserved.
from enum import Enum


class ExprMode(Enum):
    """The expression running mode."""
    EXPORT = 'Export'
    INFERENCE = 'Inference'


class ReturnType(Enum):
    """The expression return type."""
    DICT = 'Dict'
    SEQUENCE = 'Sequence'
    OTHER = 'Other'
    UNKNOWN = 'Unknown'
