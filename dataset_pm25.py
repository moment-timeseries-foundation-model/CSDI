from torch.utils.data import DataLoader, Dataset
import torch


class PM25Dataset(Dataset):
    def __init__(
        self,
        eval_length: int = 36,
        target_dim: int = 36,
        mode: str = "train",
        validindex: int = 0,
        path_to_mean_std: str = "./data/pm25/pm25_meanstd.pk",
        path_to_ground_truth: str = "./data/pm25/Code/STMVL/SampleData/pm25_ground.txt",
        path_to_missing: str = "./data/pm25/Code/STMVL/SampleData/pm25_missing.txt",
    ):
        """
        Args:
            eval_length (int): Length of the evaluation sequence.
            target_dim (int): Dimension of the target variable.
            mode (str): Mode of the dataset ('train', 'valid', 'test').
            validindex (int): Index for validation.
            path_to_mean_std (str): Path to the mean and standard deviation file.
            path_to_ground_truth (str): Path to the ground truth data file.
            path_to_missing (str): Path to the missing data file.
        """
        # TODO: Implement the __init__ method
        raise NotImplementedError(
            "Please implement the __init__ method in the subclass"
        )

    def __getitem__(self, index):
        # TODO: Implement the __getitem__ method
        raise NotImplementedError("Please implement __getitem__ method in the subclass")

    def __len__(self):
        # TODO: Implement the __len__ method
        raise NotImplementedError("Please implement __len__ method in the subclass")


def get_dataloader(batch_size, device, validindex=0):
    dataset = PM25Dataset(mode="train", validindex=validindex)
    train_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=1, shuffle=True
    )
    dataset_test = PM25Dataset(mode="test", validindex=validindex)
    test_loader = DataLoader(
        dataset_test, batch_size=batch_size, num_workers=1, shuffle=False
    )
    dataset_valid = PM25Dataset(mode="valid", validindex=validindex)
    valid_loader = DataLoader(
        dataset_valid, batch_size=batch_size, num_workers=1, shuffle=False
    )

    scaler = torch.from_numpy(dataset.train_std).to(device).float()
    mean_scaler = torch.from_numpy(dataset.train_mean).to(device).float()

    return train_loader, valid_loader, test_loader, scaler, mean_scaler
