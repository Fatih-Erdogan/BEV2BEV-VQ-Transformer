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
from collections import Counter
import matplotlib.pyplot as plt


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

    os.makedirs('%s/plots/' % opt.log_dir, exist_ok=True)
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

    def inspect_indices(codebook_size, save_path):
        epoch_size = len(test_loader)
        model.eval()
        all_indices = list()
        progress = progressbar.ProgressBar(max_value=epoch_size).start()
        with torch.no_grad():
            for i in range(epoch_size):
                progress.update(i + 1)
                x = next(testing_batch_generator)
                indices = model.encode_code(x).cpu().flatten().tolist()
                all_indices.extend(indices)

        counts = Counter(all_indices)
        for key in range(codebook_size):  # +1 because range is exclusive at the end
            if key not in counts:
                counts[key] = 0
        sorted_counts = dict(sorted(counts.items()))

        plt.bar(sorted_counts.keys(), sorted_counts.values(), color='blue')
        plt.xlabel('Integer Values')
        plt.ylabel('Counts')
        plt.title('Histogram of Integer Counts')

        plt.savefig(save_path)

        return sorted_counts

    print("Starting inspect!\n")
    save_dir = '%s/plots/' % opt.log_dir
    save_f = os.path.join(save_dir, "index_usage")
    inspect_indices(codebook_size=opt.codebook_num_emb, save_path=save_f)
    print("Ended inspect.\n")
    print(f"Results saved to {save_f}\n")


if __name__ == '__main__':
    main()
