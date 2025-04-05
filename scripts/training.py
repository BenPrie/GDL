# Imports as always...
from tqdm.notebook import tqdm

import torch
import torch.nn as nn


def train(model, train_loader, optimizer, criterion, device, verbose=False):
    # Into train mode.
    model.train()
    running_loss = 0.0

    for inputs, labels in (tqdm(train_loader, desc='Train.') if verbose else train_loader):
        # Reset gradients.
        optimizer.zero_grad()

        # Move to device.
        inputs, labels = inputs.to(device), labels.to(device)

        # Model on outputs and loss.
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backprop.
        loss.backward()
        optimizer.step()

        # Update loss.
        running_loss += loss.item()

    # Return average loss over the epoch.
    return running_loss / len(train_loader)


# Evaluation step.
def evaluate(model, test_loader, criterion, device, verbose=False):
    # Into eval mode.
    model.eval()
    running_loss = 0.0
    correct = 0

    with torch.no_grad():
        for inputs, labels in (tqdm(test_loader, desc='Eval.') if verbose else test_loader):
            # Move to device.
            inputs, labels = inputs.to(device), labels.to(device)

            # Model on outputs and loss.
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Update loss.
            running_loss += loss.item()

            # Argmax to get predicted label.
            pred = outputs.argmax(dim=1)

            # Update accuracy.
            correct += (pred == labels).sum().item()

    # Return average loss and accuracy over the epoch.
    return running_loss / len(test_loader), correct / len(test_loader.dataset)


# Full training cycle for a given model and dataset.
def experiment(model, train_loader, val_loader, device, n_epochs=10, lr=1e-3, verbose=False, print_interval=1):
    # Optimiser.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Loss function.
    criterion = nn.CrossEntropyLoss()

    # Track the losses and val accuracies.
    train_losses = []
    val_losses = []
    val_accs = []

    for epoch in tqdm(range(1, n_epochs + 1), desc='Experiment.'):
        # Train.
        train_loss = train(model, train_loader, optimizer, criterion, device, verbose)

        # Evaluate.
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, verbose)

        # Print (use interval = -1 to skip printing stats).
        if print_interval != -1:
            if epoch % print_interval == 0:
                print(f'Epoch {epoch}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # Appends.
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

    # Return losses and val accuracies.
    return train_losses, val_losses, val_accs
