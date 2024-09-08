import json
import os
import random
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn as nn
from model import VQTransformer
from utils import load_dataset, normalize_data, reshape_data, save_eval
from data_preprocess.preprocess_data import create_xy_bins
from torch.utils.data import DataLoader
from dataloader.dataset import TransformerCollateFunction
import progressbar
from piq import ssim


def main():
    with open("config_vq_transformer.json", "r") as file:
        opt = json.load(file)
        opt = SimpleNamespace(**opt)

    assert opt.model_dir != "", "Need to provide a model directory for evaluation."

    # load vqvae model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # load model and continue training from checkpoint
    saved_model = torch.load('%s/model.pth' % opt.model_dir, map_location=device)
    model_dir = opt.model_dir
    opt = saved_model['opt']

    # load the saved vqvae with the model
    opt_vqvae = saved_model['opt_vqvae']
    vqvae_model = saved_model['vqvae']

    opt.model_dir = model_dir

    # freeze vqvae
    for param in vqvae_model.parameters():
        param.requires_grad = False

    os.makedirs('%s/evaluation/plots' % opt.log_dir, exist_ok=True)

    dtype1 = torch.FloatTensor
    dtype2 = torch.IntTensor

    seed = opt.seed
    if torch.cuda.is_available():
        print("Random Seed: ", seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # dtype1 = torch.cuda.FloatTensor
        # dtype2 = torch.cuda.IntTensor

    else:
        print("CUDA WAS NOT AVAILABLE!")
        print("Didn't seed...")

    print(opt)

    # load the transformer
    predictor = saved_model["transformer"]

    down_to = opt_vqvae.image_size / (opt_vqvae.downsample ** 2)
    vq_transformer = VQTransformer(vqvae_model, predictor, t=int(opt_vqvae.time_window), hw_prime=int(down_to))

    min_delta, max_delta = -2, 2
    bin_lims_x, bin_lims_y = create_xy_bins(min_delta, max_delta, (opt.num_x_bins, opt.num_y_bins))
    collate_fn = TransformerCollateFunction(bin_lims_x, bin_lims_y)

    assert opt.n_future % opt_vqvae.time_window == 0
    opt.n_past = opt_vqvae.time_window
    opt.channels = opt_vqvae.channels
    train_data, test_data = load_dataset(opt, is_vqvae=False)

    test_loader = DataLoader(test_data,
                             batch_size=opt.batch_size,
                             shuffle=False,
                             drop_last=True,
                             pin_memory=True,
                             collate_fn=collate_fn)

    def get_testing_batch():
        while True:
            for sequence in test_loader:
                x, actions = normalize_data(dtype1, dtype2, sequence)
                # x: seq_len * bs * c * h * w
                # actions: seq_len * bs * 8
                batch = reshape_data(opt, x, actions)
                # returns x, actions
                # x: num_group, bs, c, t, h, w
                # actions: (num_group - 1) * bs * t * 8 or (num_group - 1) * bs * 8
                yield batch

    testing_batch_generator = get_testing_batch()

    def calculate_mean_iou(pred, gt, standard=True):
        # for now, assume c is always 1
        """
        :param gt: tensor(bs * c * h * w) gt for n'th PREDICTION sequence, over all minibatch elements
        :param pred: tensor(bs * c * h * w) preds for n'th PREDICTION sequence, over all minibatch elements
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

    reconstruction_criterion = nn.MSELoss()

    def evaluate(x, actions):
        mse = list()
        miou = list()
        ssim = list()
        with torch.no_grad():
            # x: num_group, bs, c, t, h, w     t being the n_past = time_window
            # actions: (num_group - 1) * bs * t * 8 or (num_group - 1) * bs * 8
            for group_num in range(len(actions)):
                past_x = x[group_num].to(device)
                future_x = x[group_num + 1].clone()
                past_actions = actions[group_num].to(device)

                # (bs * t * h * w * c) returns values between -1 and 1 (numpy)
                prediction = vq_transformer(past_x, past_actions)
                prediction = torch.from_numpy(prediction)
                # set next element to process as the current prediction
                x[group_num + 1] = prediction.permute(0, 4, 1, 2, 3)  # bs * c * t * h * w
                # prepare for evaluation
                prediction = prediction.permute(1, 0, 4, 2, 3)  # (t * bs * c * h * w)
                prediction = ((prediction + 1) / 2).clamp(0, 1).cpu()

                # need to adapt future_x
                future_x = future_x.permute(2, 0, 1, 3, 4)  # (t * bs * c * h * w)
                future_x = ((future_x + 1) / 2).clamp(0, 1)

                for t in range(future_x.size(0)):
                    mse.append(reconstruction_criterion(prediction[t], future_x[t]).item())
                    miou.append(calculate_mean_iou(prediction[t], future_x[t], True))
                    ssim.append(calculate_ssim(prediction[t], future_x[t]))

        return torch.tensor(mse), torch.tensor(miou), torch.tensor(ssim)

    test_mse = torch.zeros(opt.n_future)
    test_m_iou = torch.zeros(opt.n_future)
    test_ssim = torch.zeros(opt.n_future)

    vq_transformer.eval()
    with torch.no_grad():
        test_len = len(test_loader)
        progress = progressbar.ProgressBar(max_value=test_len).start()
        for i in range(test_len):
            progress.update(i + 1)
            x, actions = next(testing_batch_generator)

            t_mse, t_m_iou, t_ssim = evaluate(x, actions)
            test_mse += t_mse
            test_m_iou += t_m_iou
            test_ssim += t_ssim
    avg_test_mse, avg_test_miou, avg_test_ssim = test_mse / test_len, test_m_iou / test_len, test_ssim / test_len
    avg_test_mse = avg_test_mse.numpy().tolist()
    avg_test_miou = avg_test_miou.numpy().tolist()
    avg_test_ssim = avg_test_ssim.numpy().tolist()
    print("MSE across sequence elements: ")
    print(avg_test_mse, end="\n\n")
    print("MIOU across sequence elements: ")
    print(avg_test_miou, end="\n\n")
    print("SSIM across sequence elements: ")
    print(avg_test_ssim, end="\n\n")

    print("Ended evaluation!\n")

    # save the results
    eval_f_name = '%s/evaluation/results.json' % opt.log_dir
    plot_dir_name = '%s/evaluation/plots/' % opt.log_dir
    save_eval(eval_f_name, avg_test_mse, avg_test_miou, avg_test_ssim, plot_dir=plot_dir_name)


if __name__ == "__main__":
    main()
