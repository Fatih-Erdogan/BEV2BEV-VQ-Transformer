import os
import json
import matplotlib.pyplot as plt
from types import SimpleNamespace


def vqvae_plot_loss_curve(path_file, out_path, train_recons=True, test_recons=True, train_vq=True, test_vq=True, clip=0, show=False):
    with open(path_file, "r") as f:
        loss_dict = json.load(f)

    train_recons_losses = []
    train_vq_losses = []
    test_recons_losses = []
    test_vq_losses = []

    for epoch, data in loss_dict.items():
        train_recons_losses.append(data['train_recons_loss'])
        train_vq_losses.append(data['train_vq_loss'])
        test_recons_losses.append(data['test_recons_loss'])
        test_vq_losses.append(data['test_vq_loss'])

    # Plotting
    epochs = [i for i in range(1, len(train_recons_losses) + 1)]
    plt.figure(figsize=(10, 5))
    if train_recons:
        plt.plot(epochs[clip:], train_recons_losses[clip:], label='Train Recon Loss')
    if test_recons:
        plt.plot(epochs[clip:], test_recons_losses[clip:], label='Test Recon Loss')
    if train_vq:
        plt.plot(epochs[clip:], train_vq_losses[clip:], label='Train VQ Loss')
    if test_vq:
        plt.plot(epochs[clip:], test_vq_losses[clip:], label='Test VQ Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Losses Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path)
    if show:
        plt.show()


def transformer_plot_loss_curve(path_file, out_file_name, clip=0, show=False):
    with open(path_file, "r") as f:
        loss_dict = json.load(f)

    train_losses = []
    test_losses = []

    for epoch, data in loss_dict.items():
        train_losses.append(data['train_cross_entropy_loss'])
        test_losses.append(data['test_cross_entropy_loss'])

    # Plotting
    epochs = [i for i in range(1, len(train_losses) + 1)]
    plt.figure(figsize=(10, 6))

    plt.plot(epochs[clip:], train_losses[clip:], label='Train Cross Entropy Loss')
    plt.plot(epochs[clip:], test_losses[clip:], label='Test Cross Entropy Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Losses Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(out_file_name)
    if show:
        plt.show()


def main():
    plot_vqvae = False
    plot_transformer = True
    if plot_vqvae:
        with open("config_vqvae.json", "r") as file:
            opt = json.load(file)
            opt = SimpleNamespace(**opt)
    if plot_transformer:
        with open("config_vq_transformer.json", "r") as file:
            opt = json.load(file)
            opt = SimpleNamespace(**opt)

    assert opt.model_dir != ""
    loss_file_name = os.path.join(opt.model_dir, "loss/out.json")
    out_out_folder_name = os.path.join(opt.model_dir, "plots/")
    os.makedirs(out_out_folder_name, exist_ok=True)

    out_file_name = os.path.join(out_out_folder_name, "training_curve.png")

    if plot_vqvae:
        vqvae_plot_loss_curve(loss_file_name, out_file_name, train_recons=True, test_recons=True, train_vq=True, test_vq=True,
                              clip=5, show=True)
    if plot_transformer:
        transformer_plot_loss_curve(loss_file_name, out_file_name, show=False)


if __name__ == '__main__':
    main()
