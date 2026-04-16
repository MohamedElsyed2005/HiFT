aic4-uav-tracker/
├── configs/
│   └── hiFT_finetune.yaml          # Configuration file
├── data/
│   ├── contest_release/
│   │   ├── dataset1/               # Videos + annotations
│   │   ├── dataset2/
│   │   ├── dataset3/
│   │   ├── dataset4/
│   │   ├── dataset5/
│   │   └── metadata/
│   │       ├── contestant_manifest.json
│   │       └── sample_submission.csv
│   ├── processed/
│   │   ├── crop511/                # Cropped images for training
│   │   │   ├── dataset1/
│   │   │   ├── dataset2/
│   │   │   └── ...
│   │   └── train.json              # Converted annotation file
├── pysot/
│   ├── core/
│   │   ├── config.py              # Updated config
│   │   └── __init__.py
│   ├── datasets/
│   │   ├── dataset.py             # Updated TrkDataset
│   │   ├── anchortarget.py        # AnchorTarget class
│   │   ├── augmentation.py        # Augmentation class
│   │   └── __init__.py
│   ├── models/
│   │   ├── backbone/
│   │   │   ├── alexnet.py         # AlexNet backbone
│   │   │   └── __init__.py
│   │   ├── utile/
│   │   │   ├── utile.py           # HiFT module
│   │   │   ├── tran.py            # Transformer module
│   │   │   └── __init__.py
│   │   ├── model_builder.py       # ModelBuilder class
│   │   ├── loss.py                # Loss functions
│   │   └── __init__.py
│   ├── tracker/
│   │   ├── base_tracker.py        # BaseTracker, SiameseTracker
│   │   ├── hift_tracker.py        # HiFTTracker class
│   │   └── __init__.py
│   ├── utils/
│   │   ├── bbox.py                # BBox utilities
│   │   ├── lr_scheduler.py        # Learning rate schedulers
│   │   ├── model_load.py          # Model loading utilities
│   │   ├── log_helper.py          # Logging utilities
│   │   ├── misc.py                # Misc utilities
│   │   ├── xcorr.py               # Cross-correlation
│   │   ├── location_grid.py       # Location grid
│   │   ├── average_meter.py       # AverageMeter class
│   │   └── __init__.py
│   └── __init__.py
├── tools/
│   ├── train.py                   # Main training script
│   ├── eval.py                    # Evaluation script
│   ├── submit.py                  # Submission generation script
│   ├── preprocess_data.py         # Data preprocessing script
│   └── __init__.py
├── snapshot/                      # Saved model checkpoints
├── logs/                          # Training logs
├── pretrained/
│   └── back.pth                   # Pretrained AlexNet weights
├── requirements.txt
├── run_train.sh                   # Training launcher
├── run_eval.sh                    # Evaluation launcher
└── README.md