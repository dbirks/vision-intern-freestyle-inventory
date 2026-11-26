"""Shelf geometry and inventory counting."""

from .aggregate import aggregate_counts, fuse_counts
from .shelves import assign_shelves, suspicious_gaps, suspicious_overlaps

__all__ = ['aggregate_counts', 'assign_shelves', 'fuse_counts', 'suspicious_gaps',
           'suspicious_overlaps']
