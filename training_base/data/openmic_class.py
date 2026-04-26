class OpenMicDataset(Dataset):
    """
    PyTorch Dataset for OpenMIC-2018 instrument classification.

    Expected directory layout:
        root/
            audio/
                000/
                    000046_3840.ogg
            openmic-2018-aggregated-labels.csv
            partitions/
                split01_train.csv
                split01_test.csv
    """

    def __init__(
            self,
            split_cfg: Any,
            task_type: str,
            num_classes: int,
            sample_rate: int,
            clip_num_samples: int,
            train_mode: bool,
            relevance_threshold: float = 0.5,
    ) -> None:
        self.task_type        = task_type
        self.num_classes      = num_classes
        self.sample_rate      = sample_rate
        self.clip_num_samples = clip_num_samples
        self.train_mode       = train_mode

        root  = Path(split_cfg["root"])
        split = split_cfg.get("split", "train")

        # Load split keys
        split_file = "split01_train.csv" if split == "train" else "split01_test.csv"
        split_keys = set((root / "partitions" / split_file).read_text().splitlines())

        # Load and filter labels
        import pandas as pd
        df = pd.read_csv(root / "openmic-2018-aggregated-labels.csv")
        df = df[df["relevance"] >= relevance_threshold]
        df = df[df["sample_key"].astype(str).isin(split_keys)]

        # Skip known corrupted files
        corrupted = {
            "071826", "071827", "087435", "095253", "095259",
            "095263", "102144", "113025", "113604", "138485"
        }
        df = df[~df["sample_key"].astype(str).isin(corrupted)]

        # Build class index from unique instruments
        unique_instruments = sorted(df["instrument"].unique())
        self.class_to_idx  = {cls: i for i, cls in enumerate(unique_instruments)}

        # Build sample list
        self.samples: List[Dict] = []
        for _, row in df.iterrows():
            key        = str(row["sample_key"])
            instrument = row["instrument"]
            class_id   = self.class_to_idx.get(instrument, -1)
            if class_id < 0 or class_id >= self.num_classes:
                continue
            audio_path = root / "audio" / key[:3] / f"{key}.ogg"
            if not audio_path.exists():
                continue
            self.samples.append({
                "audio_path": str(audio_path),
                "label":      class_id,
                "instrument": instrument,
                "sample_key": key,
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        if self.task_type == "instrument_classification":
            excerpt = audio.load_waveform(item["audio_path"], self.sample_rate)
            excerpt = audio.trim_or_pad(excerpt, self.clip_num_samples, self.train_mode)
            label   = torch.tensor(item["label"], dtype=torch.long)
            return excerpt, label
        raise ValueError(f"Unknown task_type: {self.task_type!r}")
