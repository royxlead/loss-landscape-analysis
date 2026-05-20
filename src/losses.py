from __future__ import annotations

import torch


def to_one_hot(targets: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
    """Convert class indices into one-hot vectors."""
    one_hot = torch.zeros((targets.shape[0], num_classes), device=targets.device, dtype=torch.float32)
    one_hot.scatter_(1, targets.view(-1, 1), 1.0)
    return one_hot


def softmax(logits: torch.Tensor) -> torch.Tensor:
    """Numerically stable softmax by subtracting row-wise max before exponentiation."""
    shifted_logits = logits - logits.max(dim=1, keepdim=True).values
    exp_shifted = torch.exp(shifted_logits)
    probs = exp_shifted / exp_shifted.sum(dim=1, keepdim=True)
    return probs


def log_softmax(logits: torch.Tensor) -> torch.Tensor:
    """Numerically stable log-softmax using the log-sum-exp trick."""
    max_logits = logits.max(dim=1, keepdim=True).values
    shifted_logits = logits - max_logits

    # log-sum-exp trick for numerical stability:
    # log(sum(exp(z))) = max(z) + log(sum(exp(z - max(z))))
    log_sum_exp = max_logits + torch.log(torch.exp(shifted_logits).sum(dim=1, keepdim=True))
    log_probs = logits - log_sum_exp
    return log_probs


class MSELoss:
    name = "MSE"
    short_name = "mse"

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = softmax(logits)
        one_hot = to_one_hot(targets, num_classes=logits.shape[1]).type_as(probs)

        # Per-sample squared error summed across classes, then averaged over batch.
        loss = torch.sum((probs - one_hot) ** 2, dim=1).mean()
        return loss


class CrossEntropyLoss:
    name = "CrossEntropy"
    short_name = "ce"

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = log_softmax(logits)
        one_hot = to_one_hot(targets, num_classes=logits.shape[1]).type_as(log_probs)

        # Negative log-likelihood for one-hot targets, averaged over the batch.
        loss = -torch.sum(one_hot * log_probs, dim=1).mean()
        return loss
