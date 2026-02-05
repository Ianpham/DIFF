"""
Adapters package - intelligent middleware between datasets and encoders.
"""

from .encoder_adapter import (
    EncoderAdapter,
    EncoderConfig,
    AdapterConfiguration,
    create_adapter,
    quick_build,
)

__all__ = [
    'EncoderAdapter',
    'EncoderConfig',
    'AdapterConfiguration',
    'create_adapter',
    'quick_build',
]