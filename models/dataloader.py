from __future__ import annotations

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split


class TensorRegressionDataset(Dataset):
    def __init__(
        self,
        size: int,
        input_dim: int,
        output_dim: int,
        noise: float,
        seed: int,
    ) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.x = torch.randn(size, input_dim, generator=generator)
        weights = torch.randn(input_dim, output_dim, generator=generator)
        bias = torch.randn(output_dim, generator=generator)
        self.y = self.x @ weights + bias
        self.y = self.y + noise * torch.randn(self.y.shape, generator=generator)

    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


class RegressionDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        input_dim: int = 32,
        output_dim: int = 1,
        train_size: int = 4096,
        valid_size: int = 1024,
        test_size: int = 1024,
        noise: float = 0.05,
        seed: int = 3407,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.train_size = train_size
        self.valid_size = valid_size
        self.test_size = test_size
        self.noise = noise
        self.seed = seed

    def setup(self, stage: str | None = None) -> None:
        total_size = self.train_size + self.valid_size + self.test_size
        dataset = TensorRegressionDataset(
            size=total_size,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            noise=self.noise,
            seed=self.seed,
        )
        generator = torch.Generator().manual_seed(self.seed)
        self.train_data, self.valid_data, self.test_data = random_split(
            dataset,
            [self.train_size, self.valid_size, self.test_size],
            generator=generator,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_data, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.valid_data, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_data, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def _loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )
