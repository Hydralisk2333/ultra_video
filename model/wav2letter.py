import torch
import torch.nn as nn
import torch.nn.functional as F

class Wav2Letter(nn.Module):
    """Wav2Letter Speech Recognition model
        Architecture is based off of Facebooks AI Research paper
        https://arxiv.org/pdf/1609.03193.pdf
        This specific architecture accepts mfcc or
        power spectrums speech signals

        TODO: use cuda if available

        Args:
            num_features (int): number of mfcc features
            num_classes (int): number of unique grapheme class labels
    """

    def __init__(self, num_features, num_classes):
        super(Wav2Letter, self).__init__()

        # Conv1d(in_channels, out_channels, kernel_size, stride)
        self.layers = nn.Sequential(
            nn.Conv1d(num_features, 250, 48, 2),
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, 7),
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, 7),
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, 7),
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, 7),
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, 7),
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, 7),
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, 7),
            torch.nn.ReLU(),
            nn.Conv1d(250, 2000, 32),
            torch.nn.ReLU(),
            nn.Conv1d(2000, 2000, 1),
            torch.nn.ReLU(),
            nn.Conv1d(2000, num_classes, 1),
        )

    def forward(self, batch):
        """Forward pass through Wav2Letter network than
            takes log probability of output

        Args:
            batch (int): mini batch of data
             shape (batch, num_features, frame_len)

        Returns:
            log_probs (torch.Tensor):
                shape  (batch_size, num_classes, output_len)
        """
        # y_pred shape (batch_size, num_classes, output_len)
        y_pred = self.layers(batch)

        # compute log softmax probability on graphemes
        log_probs = F.log_softmax(y_pred, dim=1)

        return log_probs