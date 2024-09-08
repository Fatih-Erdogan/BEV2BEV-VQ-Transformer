from model import ActionConditionedTransformer, ConcatConditionedTransformer, VQVAE3D
import json
import os
import random
from types import SimpleNamespace
import numpy as np
import torch
import torch.optim as optim
from model import VQTransformer
from utils import load_dataset, normalize_data, reshape_data, save_transformer_losses, save_as_gif, save_as_image
from data_preprocess.preprocess_data import create_xy_bins
from torch.utils.data import DataLoader
from dataloader.dataset import TransformerCollateFunction
import progressbar


def main():
    with open("config_vq_transformer.json", "r") as file:
        opt = json.load(file)
        opt = SimpleNamespace(**opt)

    # load vqvae model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if opt.model_dir != '':
        # load model and continue training from checkpoint
        saved_model = torch.load('%s/model.pth' % opt.model_dir, map_location=device)
        epochs = opt.epochs
        optimizer = opt.optimizer
        model_dir = opt.model_dir
        opt = saved_model['opt']

        # load the saved vqvae with the model
        opt_vqvae = saved_model['opt_vqvae']
        vqvae_model = saved_model['vqvae']

        opt.epochs = epochs
        opt.optimizer = optimizer
        opt.model_dir = model_dir
        opt.log_dir = '%s/continued' % opt.log_dir

    else:
        # load the trained vqvae model
        saved_vqvae = torch.load('%s/model.pth' % opt.vqvae_model_dir, map_location=device)
        opt_vqvae = saved_vqvae['opt']
        encoder = saved_vqvae['encoder']
        codebook = saved_vqvae['codebook']
        decoder = saved_vqvae['decoder']
        vqvae_model = VQVAE3D(encoder, decoder, codebook).to(device)

        # vqvae_name = opt.vqvae_model_dir.split("/")[-1]
        model_type = "concat_cond" if opt.model_type == "concat_conditioned" else None
        model_type = "layer_cond" if opt.model_type == "layer_conditioned" else model_type
        vq_transformer_name = f"model-type={model_type}-" \
                              f"e{opt_vqvae.vq_emb_size}-s{opt_vqvae.codebook_num_emb}-" \
                              f"transformer-hid_dim={opt.transformer_hid_dim}-num_blocks={opt.transformer_blocks}-" \
                              f"attn_heads={opt.num_attention_heads}-pos_enc={opt.positional_encoding_type}-" \
                              f"action_cond={opt.is_action_conditioned}-dim_per_action={opt.dim_per_action}-" \
                              f"actions_per_frame={opt.num_actions_per_frame}-first_action_only={opt.first_action_only}-" \
                              f"action_rnn_lay={opt.action_rnn_layers}-use_vq_emb={opt.use_vq_embeddings}"
        dataset = "road_only" if opt_vqvae.channels == 1 else "with_objects"
        opt.log_dir = '%s/%s/%s' % (opt.log_dir, dataset + "_" + opt.train_on, vq_transformer_name)

    # freeze vqvae
    for param in vqvae_model.parameters():
        param.requires_grad = False

    os.makedirs('%s/gen/' % opt.log_dir, exist_ok=True)
    os.makedirs('%s/plots/' % opt.log_dir, exist_ok=True)

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

    # ---------------- optimizers ----------------
    if opt.optimizer == 'adam':
        opt.optimizer = optim.Adam
    elif opt.optimizer == 'rmsprop':
        opt.optimizer = optim.RMSprop
    elif opt.optimizer == 'sgd':
        opt.optimizer = optim.SGD
    else:
        raise ValueError('Unknown optimizer: %s' % opt.optimizer)

    # load the transformer
    if opt.model_dir != '':
        predictor = saved_model["transformer"]
    else:
        PredictorClass = ConcatConditionedTransformer if opt.model_type == "concat_conditioned" else ActionConditionedTransformer
        down_to = opt_vqvae.image_size / (opt_vqvae.downsample ** 2)
        seq_len = int((down_to ** 2) * opt_vqvae.time_window)
        vocab_size = opt_vqvae.codebook_num_emb
        the_codebook = codebook if opt.use_vq_embeddings else None
        predictor = PredictorClass(
            vocab_size=vocab_size, hid_dim=opt.transformer_hid_dim, num_blocks=opt.transformer_blocks,
            num_heads=opt.num_attention_heads, attn_drop=opt.attention_dropout, seq_len=seq_len,
            positional_type=opt.positional_encoding_type, action_condition=opt.is_action_conditioned,
            total_cond_dim=opt.dim_per_action * opt.num_actions_per_frame, dim_per_action=opt.dim_per_action,
            num_cond_bins=opt.num_x_bins * opt.num_y_bins, first_action_only=opt.first_action_only,
            action_rnn_layers=opt.action_rnn_layers, use_vq_embeddings=opt.use_vq_embeddings, codebook=the_codebook)
        predictor = predictor.to(device)

    down_to = opt_vqvae.image_size / (opt_vqvae.downsample ** 2)
    vq_transformer = VQTransformer(vqvae_model, predictor, t=int(opt_vqvae.time_window), hw_prime=int(down_to))
    optimizer = opt.optimizer(vq_transformer.parameters(), lr=opt.lr)

    min_delta, max_delta = -2, 2
    bin_lims_x, bin_lims_y = create_xy_bins(min_delta, max_delta, (opt.num_x_bins, opt.num_y_bins))
    collate_fn = TransformerCollateFunction(bin_lims_x, bin_lims_y)

    assert opt.n_future % opt_vqvae.time_window == 0
    opt.n_past = opt_vqvae.time_window
    opt.channels = opt_vqvae.channels
    train_data, test_data = load_dataset(opt, is_vqvae=False)
    train_loader = DataLoader(train_data,
                              num_workers=opt.data_threads,
                              batch_size=opt.batch_size,
                              shuffle=True,
                              drop_last=True,
                              pin_memory=True,
                              collate_fn=collate_fn)
    test_loader = DataLoader(test_data,
                             num_workers=opt.data_threads,
                             batch_size=opt.batch_size,
                             shuffle=True,
                             drop_last=True,
                             pin_memory=True,
                             collate_fn=collate_fn)

    def get_training_batch():
        while True:
            for sequence in train_loader:
                x, actions = normalize_data(dtype1, dtype2, sequence)
                # x: seq_len * bs * c * h * w
                # actions: seq_len * bs * 8
                batch = reshape_data(opt, x, actions)
                # returns x, actions
                # x: num_group, bs, c, t, h, w
                # actions: (num_group - 1) * bs * t * 8 or (num_group - 1) * bs * 8
                yield batch

    training_batch_generator = get_training_batch()

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

    def sample(epoch_num, num_sample_sequence):
        with torch.no_grad():
            vq_transformer.eval()
            for i in range(num_sample_sequence):
                x, actions = next(testing_batch_generator)
                # take the first batch elements
                x = x[:, 0].unsqueeze(1)    # num_group, 1, c, t, h, w
                actions = actions[:, 0].unsqueeze(1)

                # form gt_sequence
                num_g, _, c, t, h, w = x.shape
                gt_sequence = x.cpu().permute(0, 1, 3, 4, 5, 2).squeeze(1).view(-1, h, w, c)  # (num_groups * t) * h * w * c
                gt_sequence = gt_sequence.clamp(-1, 1)
                gt_sequence = (((gt_sequence + 1) / 2 * 255) // 1)

                # get the initial seq element
                initial_x = x[0].to(device)
                encoded_x = vq_transformer.encode_to_indices(initial_x)  # bs * (seq_len = t * h' * w')

                # flatten the quantized indices
                pred_sequences = list()
                pred_sequences.append(encoded_x)
                # predict next sequences based on the predictions, not gt of course
                for group_num in range(len(actions)):
                    past_actions = actions[group_num].to(device)
                    encoded_future_x, _ = vq_transformer.forward_on_indices(encoded_x, past_actions)
                    pred_sequences.append(encoded_future_x)
                    encoded_x = encoded_future_x

                reconstructions = list()
                for seq in pred_sequences:
                    rec = vq_transformer.decode_from_indices(seq)  # bs * t * h * w * c (bs is 1)
                    rec = torch.from_numpy(rec)
                    reconstructions.append(rec.squeeze(0))

                reconstructions = torch.concat(reconstructions, dim=0)  # (num_groups * t) * h * w * c
                reconstructions = reconstructions.clamp(-1, 1)
                reconstructions = (((reconstructions + 1) / 2 * 255) // 1)

                dir_name = '%s/gen/epoch_%.3d' % (opt.log_dir, epoch_num)
                f_name = "%s/sample_%.2d.gif" % (dir_name, i + 1)
                save_as_gif(gt=gt_sequence.to(torch.uint8), preds=reconstructions.to(torch.uint8), gif_path=f_name, fps=10)
                f_name = "%s/sample_%.2d.png" % (dir_name, i + 1)
                save_as_image(gt=gt_sequence.to(torch.uint8), preds=reconstructions.to(torch.uint8), out_path=f_name)

    def evaluate():
        epoch_size = len(test_loader)
        loss_list = list()
        progress = progressbar.ProgressBar(max_value=epoch_size).start()
        with torch.no_grad():
            for i in range(epoch_size):
                progress.update(i + 1)
                # x: num_group, bs, c, t, h, w     t being the n_past = time_window
                # actions: (num_group - 1) * bs * t * 8 or (num_group - 1) * bs * 8
                x, actions = next(testing_batch_generator)
                for group_num in range(len(actions)):
                    past_x = x[group_num].to(device)
                    future_x = x[group_num + 1].to(device)
                    past_actions = actions[group_num].to(device)
                    loss = vq_transformer.cross_entropy_loss(cond_x=past_x, future_x=future_x, actions=past_actions)
                    loss_list.append(loss.item())
        return np.mean(np.array(loss_list))

    def train(past_x, future_x, past_actions):
        # past_x, future_x -> bs, c, t, h, w
        # past_actions -> bs * t * 8 or bs * 8
        loss = vq_transformer.cross_entropy_loss(cond_x=past_x, future_x=future_x, actions=past_actions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def train_epoch():
        epoch_size = len(train_loader)
        loss_list = list()
        progress = progressbar.ProgressBar(max_value=epoch_size).start()
        for i in range(epoch_size):
            progress.update(i + 1)
            # x: num_group, bs, c, t, h, w     t being the n_past = time_window
            # actions: (num_group - 1) * bs * t * 8 or (num_group - 1) * bs * 8
            x, actions = next(training_batch_generator)
            for group_num in range(len(actions)):
                past_x = x[group_num].to(device)
                future_x = x[group_num + 1].to(device)
                past_actions = actions[group_num].to(device)
                loss = train(past_x, future_x, past_actions)
                loss_list.append(loss)
        return loss_list

    train_mini_losses = list()
    test_losses = list()
    for epoch in range(1, opt.epochs + 1):
        print(f"Epoch {epoch} / {opt.epochs}")
        vq_transformer.train()
        epoch_losses = train_epoch()
        train_mini_losses.append(epoch_losses)

        vq_transformer.eval()
        test_loss = evaluate()
        test_losses.append(test_loss)

        loss_file = os.path.join(opt.log_dir, "loss/out.json")
        save_transformer_losses(loss_file, train_mini_losses, test_losses)
        sample(epoch, 10)

        torch.save({
            'transformer': predictor,
            'opt': opt,
            'vqvae': vqvae_model,
            'opt_vqvae': opt_vqvae
        }, '%s/model.pth' % opt.log_dir)


if __name__ == "__main__":
    main()
