import numpy as np
import torch
from torchvision.utils import make_grid
from .base_trainer import BaseTrainer
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, model, loss, metrics, optimizer, resume, config,
                 data_loader, valid_data_loader=None, lr_scheduler=None, train_logger=None):

        super(Trainer, self).__init__(model, loss, metrics, optimizer, resume, config, train_logger)

        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        # Initialize lists to store metrics
        self.train_losses = []
        self.train_metrics = [[] for _ in range(len(metrics))]
        self.val_losses = []
        self.val_metrics = [[] for _ in range(len(metrics))]

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.
        """
        self.model.train()
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        self.writer.set_step(epoch)

        _trange = tqdm(self.data_loader, leave=True, desc='')

        for batch_idx, batch in enumerate(_trange):
            batch = [b.to(self.device) for b in batch]
            data, target = batch[:-1], batch[-1]
            data = data if len(data) > 1 else data[0]

            self.optimizer.zero_grad()
            output = self.model(data)

            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_metrics += self._eval_metrics(output, target)

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                _str = 'Train Epoch: {} Loss: {:.6f}'.format(epoch, loss.item())
                _trange.set_description(_str)

        # Calculate and log average loss and metrics
        loss = total_loss / len(self.data_loader)
        metrics = (total_metrics / len(self.data_loader)).tolist()

        self.writer.add_scalar('loss', loss)
        for i, metric in enumerate(self.metrics):
            self.writer.add_scalar("%s" % metric.__name__, metrics[i])

        self.train_losses.append(loss)
        for i in range(len(metrics)):
            self.train_metrics[i].append(metrics[i])

        if self.config['data']['format'] == 'image':
            self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        log = {
            'loss': loss,
            'metrics': metrics
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))

        self.writer.set_step(epoch, 'valid')

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.valid_data_loader):
                batch = [b.to(self.device) for b in batch]
                data, target = batch[:-1], batch[-1]
                data = data if len(data) > 1 else data[0]

                output = self.model(data)
                loss = self.loss(output, target)

                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(output, target)

            # Average over batches
            val_loss = total_val_loss / len(self.valid_data_loader)
            val_metrics = (total_val_metrics / len(self.valid_data_loader)).tolist()

            self.val_losses.append(val_loss)
            for i in range(len(val_metrics)):
                self.val_metrics[i].append(val_metrics[i])

            self.writer.add_scalar('loss', val_loss)
            for i, metric in enumerate(self.metrics):
                self.writer.add_scalar("%s" % metric.__name__, val_metrics[i])

            if self.config['data']['format'] == 'image':
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        return {
            'val_loss': val_loss,
            'val_metrics': val_metrics
        }

    def plot_metrics(self, save_path=None):
        """
        Plot the training and validation metrics.
        Optionally save the plot to the specified path.
        """
        epochs = range(1, len(self.train_losses) + 1)

        plt.figure(figsize=(12, 10))

        # Loss plot
        plt.subplot(2, 2, 1)
        plt.plot(epochs, self.train_losses, label='Train Loss', color='blue')
        plt.plot(epochs, self.val_losses, label='Validation Loss', color='orange')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Accuracy plot
        plt.subplot(2, 2, 2)
        plt.plot(epochs, self.train_metrics[0], label='Train Accuracy', color='blue')
        plt.plot(epochs, self.val_metrics[0], label='Validation Accuracy', color='orange')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Avg Precision plot
        plt.subplot(2, 2, 3)
        plt.plot(epochs, self.train_metrics[1], label='Train Avg Precision', color='blue')
        plt.plot(epochs, self.val_metrics[1], label='Validation Avg Precision', color='orange')
        plt.title('Avg Precision')
        plt.xlabel('Epoch')
        plt.ylabel('Avg Precision')
        plt.legend()

        # Avg Recall plot
        plt.subplot(2, 2, 4)
        plt.plot(epochs, self.train_metrics[2], label='Train Avg Recall', color='blue')
        plt.plot(epochs, self.val_metrics[2], label='Validation Avg Recall', color='orange')
        plt.title('Avg Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Avg Recall')
        plt.legend()

        plt.tight_layout()

        # Save the plot if a filepath is provided
        if save_path:
            # 获取当前时间并格式化
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            # 构造新的保存路径
            save_path_with_time = f"{save_path}_{current_time}.png"
            plt.savefig(save_path_with_time)
            print(f"Plot saved to {save_path_with_time}")

        plt.show()
