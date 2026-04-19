# MIR Class Spring 2026, Final Project, Instrument Classification

This repo serves as a basis for our project, we need to implement the different components.

- `training_base/data/`
  Dataset building hooks, augmentations, collate function, and dataloaders.
- `training_base/models/encoders.py`
  Encoder selection and encoder forward.
- `training_base/models/classifiers.py`
  Classifier selection.
- `training_base/models/system.py`
  Training, validation, and test loop.
- `training_base/evaluation/metrics.py`
  Evaluation metrics.
- `training_base/cli.py`
  `pl.Trainer` entry point.

## Dataset Options

The scaffold supports two basic dataset paths:

- `type: "slakh"`
  Placeholder class for the Slakh dataset.
- `type: "openmic"`
  Placeholder class for the OpenMIC dataset.


## Running

```bash
python3 -m training_base.cli --config configs/dummy_synthetic_to_real.json
```

In the config file we should specify all the hyperparameters and details of our training and then everything will be loaded in the code.