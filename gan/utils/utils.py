import torch
import torchmetrics

"""
SegSNR implementation involving dividing the signal into segments, 
calculating the SNR for each segment, and then averaging these values
"""

class SegSNR(torchmetrics.Metric):
    def __init__(self, seg_length=160):
        super().__init__()
        self.seg_length = seg_length
        self.add_state("total_snr", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("segments", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.

        Args:
            preds (torch.Tensor): Predicted tensor
            target (torch.Tensor): Ground truth tensor
        """
        batch_size, _ = preds.shape
        for i in range(batch_size):
            pred = preds[i]
            targ = target[i]

            # Ensuring length compatibility
            min_len = min(pred.size(0), targ.size(0))
            pred = pred[:min_len]
            targ = targ[:min_len]

            # Calculating SNR for segments and updating state
            num_segments = int(torch.floor(torch.tensor(min_len / self.seg_length)))
            for j in range(num_segments):
                start = j * self.seg_length
                end = start + self.seg_length
                seg_pred = pred[start:end]
                seg_targ = targ[start:end]

                noise = seg_targ - seg_pred
                snr = 10 * torch.log10(torch.sum(seg_targ**2) / torch.sum(noise**2))
                self.total_snr += snr
                self.segments += 1

    def compute(self):
        """
        Computes the average SegSNR over all updated states.
        """
        if self.segments == 0:
            return torch.tensor(float('inf'))
        return self.total_snr / self.segments



