import torch
from piq import ssim
import json
import os
import random
import numpy as np
import progressbar
from types import SimpleNamespace
from model import VQVAE3D
from dataloader.dataset import vqvae_collate_fn
from utils import load_dataset, normalize_frames
from torch.utils.data import DataLoader
import torch.nn.functional as F


def main():
    with open("config_vqvae.json", "r") as file:
        opt = json.load(file)
        opt = SimpleNamespace(**opt)

    assert opt.model_dir != "", "For evaluation, a pretrained model directory is required!"

    # load model and evaluate the model at that checkpoint
    map_loc = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    saved_model = torch.load('%s/model.pth' % opt.model_dir, map_location=map_loc)
    bs = opt.batch_size
    opt = saved_model['opt']
    opt.batch_size = bs

    # print(opt)
    # input("Do you want to continue? If not ctrl+c")

    os.makedirs('%s/evaluation/' % opt.log_dir, exist_ok=True)
    # os.makedirs('%s/evaluation/plots/' % opt.log_dir, exist_ok=True)

    dtype = torch.FloatTensor
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed = opt.seed
    if torch.cuda.is_available():
        print("Random Seed: ", seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        dtype = torch.cuda.FloatTensor
    else:
        print("CUDA WAS NOT AVAILABLE!")
        print("Didn't seed...")

    print(opt)

    encoder = saved_model['encoder']
    codebook = saved_model['codebook']
    decoder = saved_model['decoder']

    model = VQVAE3D(encoder, decoder, codebook).to(device)

    train_data, test_data = load_dataset(opt, is_vqvae=True)
    collate_fn = vqvae_collate_fn

    test_loader = DataLoader(test_data,
                             num_workers=opt.data_threads,
                             batch_size=opt.batch_size,
                             shuffle=True,
                             drop_last=True,
                             pin_memory=True,
                             collate_fn=collate_fn)

    def get_testing_batch():
        while True:
            for sequence in test_loader:
                # data between -1 and 1
                batch = (normalize_frames(dtype, sequence) - 0.5) * 2
                # returns frames
                yield batch

    testing_batch_generator = get_testing_batch()

    def calculate_mean_iou(pred, gt, standard=True):
        # for now, assume c is always 1
        """
        :param gt: tensor(bs * c * h * w) gt for n'th PREDICTION sequence, over all minibatch elements
        :param pred: tensor(bs * c * h * w) preds for n'th PREDICTION sequence, over all minibatch elements
        :param pred: bool
        :return: float: mean intersection over union
        """
        if standard:
            scale = 1.0 / gt.max()
            gt = gt * scale
            pred = pred * scale

            threshold = 0.3
            gt = gt > threshold
            pred = pred > threshold
            intersection = torch.logical_and(pred, gt).float()
            union = torch.logical_or(pred, gt).float()

            intersection = intersection.sum(dim=[1, 2, 3])  # bs
            union = union.sum(dim=[1, 2, 3])  # bs
            iou = intersection / torch.clamp(union, min=1e-6)
            return iou.mean().item()

        else:
            union = torch.max(pred, gt)
            intersection = torch.min(pred, gt)

            c_inter = intersection.sum(dim=(1, 2, 3))  # bs
            c_uni = union.sum(dim=(1, 2, 3))  # bs
            m_iou = torch.mean(c_inter / c_uni)  # mean over batch for a seq element

            return m_iou.item()

    def calculate_ssim(pred, gt):
        # for now, assume c is always 1
        """
        :param gt: tensor(bs * c * h * w) gt for n'th PREDICTION sequence, over all minibatch elements
        :param pred: tensor(bs * c * h * w) preds for n'th PREDICTION sequence, over all minibatch elements
        :return: float: ssim
        """
        pred = pred.float()
        gt = gt.float()
        ssim_value = ssim(pred, gt, kernel_size=11, data_range=1.0, reduction="mean")

        return ssim_value.item()

    def evaluate():
        epoch_size = len(test_loader)
        recon_losses = list()
        miou = list()
        ssim = list()
        model.eval()
        progress = progressbar.ProgressBar(max_value=epoch_size).start()
        with torch.no_grad():
            for i in range(epoch_size):
                progress.update(i + 1)
                x = next(testing_batch_generator)
                x_tilde, _ = model(x)
                bs, c, t, h, w = x.shape
                # convert from bs * c * t * h * w to (bs * t) * c * h * w
                # for ssim and mean iou
                x = x.cpu().permute(0, 2, 1, 3, 4).view(-1, c, h, w).clamp(-1, 1)
                x_tilde = x_tilde.cpu().permute(0, 2, 1, 3, 4).view(-1, c, h, w)

                rec_loss = F.mse_loss(x_tilde, x)
                x = (x.clamp(-1, 1) + 1) / 2
                x_tilde = (x_tilde.clamp(-1, 1) + 1) / 2
                miou.append(calculate_mean_iou(x_tilde, x))
                ssim.append(calculate_ssim(x_tilde, x))
                recon_losses.append(rec_loss)
        return np.mean(np.array(recon_losses)), np.mean(np.array(miou)), np.mean(np.array(ssim))

    print("Starting evaluation...")
    recon_loss, mean_iou, ssim_score = evaluate()
    print("Reconstruction loss: ")
    print(recon_loss, end="\n\n")
    print("MIOU: ")
    print(mean_iou, end="\n\n")
    print("SSIM: ")
    print(ssim_score, end="\n\n")

    print("Ended evaluation!\n")
    print("Saving results.\n")
    eval_f_name = '%s/evaluation/results.json' % opt.log_dir
    with open(eval_f_name, 'w') as file:
        json.dump({
            "test_recon_loss": float(recon_loss),
            "test_miou": float(mean_iou),
            "test_ssim": float(ssim_score)
        }, file, indent=4)
    print(f"Results saved to {eval_f_name}\n")


if __name__ == '__main__':
    main()
