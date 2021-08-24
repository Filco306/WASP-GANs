r"""
data_module.py
Contains PyTorch-Lightning's datamodule and dataloaders 
"""
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from src.utilities.data_utils import Normalize
from torchvision import transforms, datasets


class Scale1Minus1(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = (x / (254.0))*2 - 1
        return x


class Rescale(torch.nn.Module):
    def __init__(self, old_min, old_max, new_min, new_max):
        super(Rescale, self).__init__()
        self.old_min = old_min
        self.old_max = old_max
        self.new_min = new_min
        self.new_max = new_max

    def forward(self, x):
        x = (x - self.old_min)/(self.old_max - self.old_min) * \
            (self.new_max - self.new_min) + self.new_min
        return x


class MyDataModule(pl.LightningDataModule):

    def __init__(self, hparams):
        r"""
        Args:
            hparams (argparse.Namespace)
        """
        super().__init__()
        self.hparams.update(vars(hparams))

        self.normalizer = Normalize(hparams.normalizing_values_mu, hparams.normalizing_values_sigma)

        # self.collate_fn = CustomCollate()

    def prepare_data(self):
        r"""
        Data preparation / Download Dataset
        """
        # Example
        # MNIST(self.data_dir, train=True, download=True)
        # MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        r"""
        Set train, val and test dataset here

        Args:
            stage (string, optional): fit, test based on plt.Trainer state. 
                                    Defaults to None.
        """
        # Example:
        # # Assign train/val datasets for use in dataloaders
        # if stage == 'fit' or stage is None:
        #     mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
        #     self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        #     # Optionally...
        #     # self.dims = tuple(self.mnist_train[0][0].shape)

        # # Assign test dataset for use in dataloader(s)
        # if stage == 'test' or stage is None:
        #     self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        #     # Optionally...
        #     # self.dims = tuple(self.mnist_test[0][0].shape)

        # raise NotImplementedError("Add your dataloaders first and remove this line")

        traintransform = transforms.Compose([
            # transforms.RandomRotation(self.hparams.training_random_rotation),
            # transforms.RandomHorizontalFlip(),            
            transforms.Resize(self.hparams.image_resized),
            transforms.CenterCrop(self.hparams.image_resized),
            self.normalizer,
        ])

        # testtransform = transforms.Compose([
        #     transforms.Resize(self.hparams.image_resized),
        #     transforms.ToTensor(),
        #     transforms.Normalize(self.hparams.normalizing_values_mu, self.hparams.normalizing_values_sigma)
        # ])

        # self.train_data = CustomDataset()
        # self.val_data = CustomDataset()
        # self.test_data = CustomDataset()
        self.train_data = datasets.ImageFolder(self.hparams.data_path, transform=traintransform)
        # x=  (x / 255.0)
        # x = (x/ (255.0))*2 - 1
        # x = (x - old_min)/(old_max - old_min)*(new_max - new_min) + new_min
        # self.val_data = datasets.ImageFolder(self.hparams.data_path)
        # self.test_data = datasets.ImageFolder(self.hparams.data_path)

        # self.train_data = datasets.ImageFolder(os.path.join(self.hparams.data_path, "train"))
        # self.val_data = datasets.ImageFolder(os.path.join(self.hparams.data_path, "val"))
        # self.test_data = datasets.ImageFolder(os.path.join(self.hparams.data_path, "test"))

    def train_dataloader(self):
        r"""
        Load trainset dataloader
        Returns:
            (torch.utils.data.DataLoader): Train Dataloader
        """
        return DataLoader(self.train_data, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers, pin_memory=True)

    # def val_dataloader(self):
    #     r"""
    #     Load Validation dataloader
    #     Returns:
    #         (torch.utils.data.DataLoader): Validation Dataloader
    #     """

    #     return DataLoader(self.val_data, batch_size=self.hparams.batch_size, collate_fn=self.collate_fn,
    #                       num_workers=self.hparams.num_workers, pin_memory=True)

    # def test_dataloader(self):
    #     r"""
    #     Load Test dataloader
    #     Returns:
    #         (torch.utils.data.DataLoader): Test Dataloader
    #     """
    #     return DataLoader(self.test_data, batch_size=self.hparams.batch_size, collate_fn=self.collate_fn,
    #                       num_workers=self.hparams.num_workers, pin_memory=True)
