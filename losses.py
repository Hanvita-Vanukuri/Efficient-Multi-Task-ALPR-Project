import torch
import torch.nn as nn


class CTCLoss(nn.Module):

    def __init__(self, blank=67):
        super().__init__()

        self.loss = nn.CTCLoss(
            blank=blank,
            zero_infinity=True
        )

    def forward(
        self,
        logits,
        targets,
        target_lengths
    ):
        """
        logits:
            [batch, sequence_length, num_classes]

        targets:
            Concatenated target labels

        target_lengths:
            Length of each target sequence
        """

        # CTC expects:
        # [sequence_length, batch, classes]

        logits = logits.permute(1, 0, 2)

        log_probs = torch.log_softmax(
            logits,
            dim=2
        )

        batch_size = logits.shape[1]
        sequence_length = logits.shape[0]

        input_lengths = torch.full(
            (batch_size,),
            sequence_length,
            dtype=torch.long
        )

        # ----------------------------------------------------
        # MPS does not currently implement CTC loss.
        # Run only the CTC calculation on CPU.
        # ----------------------------------------------------

        log_probs_cpu = log_probs.float().cpu()
        targets_cpu = targets.long().cpu()
        input_lengths_cpu = input_lengths.cpu()
        target_lengths_cpu = target_lengths.long().cpu()

        loss = self.loss(
            log_probs_cpu,
            targets_cpu,
            input_lengths_cpu,
            target_lengths_cpu
        )

        # Move the scalar loss back to the model device
        return loss.to(logits.device)