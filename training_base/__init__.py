"""
Team-friendly training base for domain-transfer audio classification.

The package is organized so different teammates can own clear slices:
- data: dataset loading, manifests, augmentations, batching
- models: encoders, classifiers, Lightning system
- evaluation: metrics and epoch summaries
"""

