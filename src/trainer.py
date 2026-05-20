from __future__ import annotations

from typing import Dict

import torch


def compute_gradient_norm(model: torch.nn.Module) -> float:
    """Compute global L2 norm over all parameter gradients."""
    total_sq = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        total_sq += torch.sum(grad * grad).item()
    return float(total_sq ** 0.5)


class Trainer:
    def __init__(self, model: torch.nn.Module, loss_fn, optimizer: torch.optim.Optimizer, device: str = "cpu") -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self, loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        total_grad_norm = 0.0
        num_batches = 0

        self.optimizer.zero_grad()

        for inputs, targets in loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            logits = self.model(inputs)
            loss = self.loss_fn(logits, targets)
            loss.backward()

            # Capture gradient norm after backward and before zeroing as required.
            grad_norm = compute_gradient_norm(self.model)

            self.optimizer.step()
            self.optimizer.zero_grad()

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_grad_norm += grad_norm
            total_samples += batch_size
            num_batches += 1

            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == targets).sum().item()

        avg_loss = total_loss / max(total_samples, 1)
        avg_grad_norm = total_grad_norm / max(num_batches, 1)
        accuracy = total_correct / max(total_samples, 1)

        return {"loss": float(avg_loss), "grad_norm": float(avg_grad_norm), "accuracy": float(accuracy)}

    def evaluate(self, loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                logits = self.model(inputs)
                loss = self.loss_fn(logits, targets)

                batch_size = targets.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                preds = torch.argmax(logits, dim=1)
                total_correct += (preds == targets).sum().item()

        avg_loss = total_loss / max(total_samples, 1)
        accuracy = total_correct / max(total_samples, 1)
        return {"loss": float(avg_loss), "accuracy": float(accuracy)}

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        epochs: int,
    ) -> Dict[str, list]:
        history = {
            "train_loss": [],
            "test_loss": [],
            "train_accuracy": [],
            "test_accuracy": [],
            "grad_norms": [],
            "epochs": [],
        }

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            test_metrics = self.evaluate(test_loader)

            history["epochs"].append(epoch)
            history["train_loss"].append(train_metrics["loss"])
            history["test_loss"].append(test_metrics["loss"])
            history["train_accuracy"].append(train_metrics["accuracy"])
            history["test_accuracy"].append(test_metrics["accuracy"])
            history["grad_norms"].append(train_metrics["grad_norm"])

        return history
