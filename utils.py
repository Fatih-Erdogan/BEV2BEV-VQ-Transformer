import random
from PIL import Image
from dataloader.dataset import SequencedDataset, VQVAEDataset
import torch
from torch.autograd import Variable
import numpy as np
import imageio
import json
import os
import matplotlib.pyplot as plt
import copy


def _split_samples_train_test(samples, list_of_test_towns):
    test_samples = list()
    train_samples = list()

    for sample in samples:
        town_name, _ = sample
        if any(test_town in town_name for test_town in list_of_test_towns):
            test_samples.append(sample)
        else:
            train_samples.append(sample)

    return train_samples, test_samples


def load_dataset(optt, is_vqvae=False):
    # can be all(100%), small(20%), tiny(2%)
    # samples is none only if user wants to train without extracting the dataset, possibly on a tiny dataset
    opt = copy.deepcopy(optt)
    if is_vqvae:
        # VQVAE
        Dataset = VQVAEDataset
        seq_length = opt.time_window
        step = None
    else:
        # GPT
        Dataset = SequencedDataset
        seq_length = opt.n_past + opt.n_future
        step = opt.n_past

    train_is_test = opt.train_is_test
    size = opt.train_on
    if size == "all":
        size = 1
    elif size == "small":
        size = 0.2
    elif size == "tiny":
        size = 0.02
    elif size == "few":
        size = 0.001
    else:
        assert False

    train_data = Dataset(opt.data_root, seq_length, step, opt.channels == 1)
    all_samples = train_data.samples

    if train_is_test:
        print("Train is Test! Train and test set will be the same.")
        print("So, no town splits...")
        # then the samples will not be split wrt towns
        interested_samples = all_samples[:int(len(all_samples) * size)]
        train_data.samples = interested_samples
        test_data = Dataset(opt.data_root, seq_length, step, opt.channels == 1,
                            samples=interested_samples)

    # if you know which towns will be in the test set
    elif opt.test_towns is not None:
        print(f"Test towns are provided as: {opt.test_towns}")
        print("Splitting accordingly...")
        # split all data into train and test according to the test towns specified in opt
        train_samples, test_samples = _split_samples_train_test(all_samples, opt.test_towns)
        random.seed(447)
        # shuffle the lists so that if the model is not trained on the full set,
        # the sample distribution across datasets are balanced
        random.shuffle(train_samples)
        random.shuffle(test_samples)
        # take the specified portion
        train_samples = train_samples[:int(len(train_samples) * size)]
        test_samples = test_samples[:int(len(test_samples) * size)]
        print(f"Total number of training sequences: {len(train_samples)}")
        print(f"Total number of testing sequences: {len(test_samples)}")
        # assign the samples to the datasets
        train_data.samples = train_samples
        test_data = Dataset(opt.data_root, seq_length, step, opt.channels == 1,
                            samples=test_samples)
    else:
        # split wrt to the proportion specified
        # do not shuffle this time, to not copy from train set
        # still as the towns are shared this method is not good
        print("Train is not test, and no test towns provided.")
        print("Splitting only according to the proportions...")
        train_prop = opt.train_test_ratio / (opt.train_test_ratio + 1)
        interested_samples = all_samples[:int(len(all_samples) * size)]
        train_last_idx = int(train_prop * len(interested_samples))
        train_data.samples = interested_samples[:train_last_idx]
        test_data = Dataset(opt.data_root, seq_length, step, opt.channels == 1,
                            samples=interested_samples[train_last_idx:])

    return train_data, test_data


def normalize_data(dtype1, dtype2, sequence):
    # squeeze images into -1 and 1
    frames = sequence["frames"]
    actions = sequence["actions"]
    frames = (frames / 128.0) - 1
    return frames.type(dtype1), actions.type(dtype2)


def reshape_data(opt, x, actions):
    """
    params:
    # x: seq_len * bs * c * h * w
    # actions: seq_len * bs * 8
    returns:
    x: num_group, bs, c, t, h, w
    actions: (num_group - 1) * bs * t * 8 or (num_group - 1) * bs * 8
    """
    # group to form dimension t
    seq, bs, c, h, w = x.shape
    x = x.view(-1, opt.n_past, bs, c, h, w)
    x = x.permute(0, 2, 3, 1, 4, 5)  # num_group, bs, c, t, h, w

    if opt.first_action_only:
        # take the last action from each sequence group element
        # idx:(1,3,5,7,9,11) if n_past=2 n_future=10 (among 0-11)
        # no need for the last one, nothing to predict there
        actions = actions[(opt.n_past - 1)::opt.n_past]
        actions = actions[:-1]  # (num_group - 1) * bs * 8
    else:
        # exclude the last one and the first (opt.n_past - 1)
        seq, bs, n = actions.shape
        actions = actions[(opt.n_past - 1):-1]
        actions = actions.view(-1, opt.n_past, bs, n)  # (num_group - 1) * t * bs * 8
        actions = actions.permute(0, 2, 1, 3)  # (num_group - 1) * bs * t * 8

    return x, actions


def normalize_frames(dtype, sequence):
    # squeeze images into 0-1
    frames = sequence["frames"]
    frames /= 256.0
    return frames.type(dtype)


def sequence_input(seq, dtype):
    return [Variable(x.type(dtype)) for x in seq]


# USED IN TRAIN SVG-LP
# #############################################!!! MODIFY DOWN ##############################################
def init_weights(m):
    classname = m.__class__.__name__
    if classname == 'ConvLSTMCell' or classname == 'ConvLSTM' or classname == 'ConvGaussianLSTM':
        return
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# #############################################!!! MODIFY UP ##############################################


def save_vq_losses(file_path, recons_minibatch_loss, vq_minibatch_loss, recons_test_loss, vq_test_loss):
    # Prepare the dictionary to be dumped
    loss_data = {}
    for i in range(len(recons_test_loss)):
        epoch_key = f'epoch_{i + 1}'
        loss_data[epoch_key] = {
            'train_recons_loss': sum(recons_minibatch_loss[i]) / len(recons_minibatch_loss[i]),
            'train_vq_loss': sum(vq_minibatch_loss[i]) / len(vq_minibatch_loss[i]),
            'test_recons_loss': recons_test_loss[i],
            'test_vq_loss': vq_test_loss[i],
            'train_recons_minibatch_losses': recons_minibatch_loss[i],
            'train_vq_minibatch_losses': vq_minibatch_loss[i]
        }

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as json_file:
        json.dump(loss_data, json_file, indent=4)
    print(f"Loss data saved to {file_path}")


def save_transformer_losses(file_path, train_mini_losses, test_losses):
    # Prepare the dictionary to be dumped
    loss_data = {}
    for i in range(len(test_losses)):
        epoch_key = f'epoch_{i + 1}'
        loss_data[epoch_key] = {
            'train_cross_entropy_loss': sum(train_mini_losses[i]) / len(train_mini_losses[i]),
            'test_cross_entropy_loss': test_losses[i],
            'train_minibatch_cross_entropy_losses': train_mini_losses[i]
        }
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as json_file:
        json.dump(loss_data, json_file, indent=4)
    print(f"Loss data saved to {file_path}")


def save_loss(loss_f_name, train_minibatch_losses, eval_batch_losses):
    results = {}
    num_epochs = len(train_minibatch_losses)

    for epoch in range(num_epochs):
        epoch_data = {
            "avg_train_mse": 0,
            "avg_train_kld": 0,
            "avg_eval_mse": float(eval_batch_losses[epoch]) if epoch < len(eval_batch_losses) else None,
            "train_minibatch_mses": [],
            "train_minibatch_klds": []
        }

        total_mse = 0
        total_kld = 0
        num_batches = len(train_minibatch_losses[epoch])

        for batch in train_minibatch_losses[epoch]:
            mse, kld = batch
            total_mse += mse
            total_kld += kld
            epoch_data["train_minibatch_mses"].append(float(mse))
            epoch_data["train_minibatch_klds"].append(float(kld))

        epoch_data["avg_train_mse"] = float(total_mse / num_batches)
        epoch_data["avg_train_kld"] = float(total_kld / num_batches)

        results[f"epoch_{epoch + 1}"] = epoch_data

    os.makedirs(os.path.dirname(loss_f_name), exist_ok=True)

    with open(loss_f_name, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Loss data saved to {loss_f_name}")


def save_eval(f_name, mse, miou, ssim, plot_dir=None):
    results = {"MSE": mse,
               "MIOU": miou,
               "SSIM": ssim}

    with open(f_name, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Eval data saved to {f_name}")

    if plot_dir is not None:
        sequence_index = list(range(1, len(results["MSE"]) + 1))

        plt.figure()
        plt.plot(sequence_index, results["MSE"], marker='o', linestyle='-', color='blue')
        plt.title("MSE Across Frames")
        plt.xlabel("Sequence Index")
        plt.ylabel("Score")
        plt.xticks(sequence_index)  # Set x-ticks to be explicit sequence indices
        plt.savefig(os.path.join(plot_dir, "mse_across_frames.png"))
        plt.close()

        plt.figure()
        plt.plot(sequence_index, results["MIOU"], marker='o', linestyle='-', label='MIOU', color='green')
        plt.plot(sequence_index, results["SSIM"], marker='o', linestyle='-', label='SSIM', color='red')
        plt.title("MIOU and SSIM Across Frames")
        plt.xlabel("Sequence Index")
        plt.ylabel("Score")
        plt.xticks(sequence_index)  # Set x-ticks to be explicit sequence indices
        plt.legend()
        plt.savefig(os.path.join(plot_dir, "miou_ssim_across_frames.png"))
        plt.close()


def save_as_image(gt, preds, out_path):
    """
    concat gt horizontally
    concat preds horizontally
    concat gt and preds vertically (gt on top)
    gt and preds are 0-255
    - gt (tensor): ground truth tensor with shape (seq_len, h, w, c)
    - preds (tensor): predictions tensor with shape (seq_len, h, w, c)
    """
    seq_len, h, w, c = gt.shape
    assert gt.shape == preds.shape, "gt and preds must have the same shape"

    gt_array = gt.numpy()
    preds_array = preds.numpy()

    # handle channels
    if c == 1:
        gt_array = np.concatenate([np.zeros_like(gt_array), np.zeros_like(gt_array), gt_array], axis=-1)
        preds_array = np.concatenate([np.zeros_like(preds_array), np.zeros_like(preds_array), preds_array], axis=-1)
    elif c == 2:
        gt_array = np.concatenate([np.zeros_like(gt_array[:, :, :, 0:1]), gt_array], axis=-1)
        preds_array = np.concatenate([np.zeros_like(preds_array[:, :, :, 0:1]), preds_array], axis=-1)

    # concat horizontally
    gt_combined = np.concatenate(gt_array, axis=1)
    preds_combined = np.concatenate(preds_array, axis=1)

    # concat vertically
    combined_image = np.concatenate([gt_combined, preds_combined], axis=0)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    imageio.imwrite(out_path, combined_image)


def save_as_gif(gt, preds, gif_path, fps=10):
    """
    - gt (tensor): ground truth tensor with shape (seq_len, h, w, c)
    - preds (tensor): predictions tensor with shape (seq_len, h, w, c)
    """
    seq_len, h, w, c = gt.shape
    assert gt.shape == preds.shape, "gt and preds must have the same shape"

    frames = []
    for i in range(seq_len):
        gt_frame = gt[i].numpy()
        preds_frame = preds[i].numpy()

        # type check
        if gt_frame.dtype != np.uint8 or preds_frame.dtype != np.uint8:
            raise ValueError("gt and preds must be uint8 tensors")

        # handle channels
        if c == 1:  # only blue channel is present
            gt_frame = np.concatenate((np.zeros_like(gt_frame), np.zeros_like(gt_frame), gt_frame), axis=-1)
            preds_frame = np.concatenate((np.zeros_like(preds_frame), np.zeros_like(preds_frame), preds_frame), axis=-1)
        elif c == 2:  # green and blue channels are present
            gt_frame = np.concatenate((np.zeros_like(gt_frame[:, :, 0:1]), gt_frame), axis=-1)
            preds_frame = np.concatenate((np.zeros_like(preds_frame[:, :, 0:1]), preds_frame), axis=-1)

        # concatenate horizontally
        combined_frame = np.concatenate((gt_frame, preds_frame), axis=1)

        frames.append(combined_frame)

    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    imageio.mimsave(gif_path, frames, format='GIF', fps=fps)


def save_gif(filename, inputs, duration=0.25):
    images = []
    for tensor in inputs:
        img = image_tensor(tensor, padding=0)
        img = img.cpu()
        img = img.transpose(0, 1).transpose(1, 2).clamp(0, 1)
        new_image = torch.zeros(img.shape[0], img.shape[1], 3)
        # complete other channels
        for i in range(img.shape[2]):
            new_image[:, :, 2 - i] = img[:, :, -1 - i]
        new_image = (new_image.numpy() * 255).astype(np.uint8)
        images.append(new_image)
    imageio.mimsave(filename, images, duration=duration)


def is_sequence(arg):
    return (not hasattr(arg, "strip") and
            not type(arg) is np.ndarray and
            not hasattr(arg, "dot") and
            (hasattr(arg, "__getitem__") or
             hasattr(arg, "__iter__")))


def image_tensor(inputs, padding=1):
    # assert is_sequence(inputs)
    assert len(inputs) > 0
    # print(inputs)

    # if this is a list of lists, unpack them all and grid them up
    if is_sequence(inputs[0]) or (hasattr(inputs, "dim") and inputs.dim() > 4):
        images = [image_tensor(x) for x in inputs]
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(c_dim,
                            x_dim * len(images) + padding * (len(images) - 1),
                            y_dim)
        for i, image in enumerate(images):
            result[:, i * x_dim + i * padding:
                      (i + 1) * x_dim + i * padding, :].copy_(image)

        return result

    # if this is just a list, make a stacked image
    else:
        images = [x.data if isinstance(x, torch.autograd.Variable) else x
                  for x in inputs]
        # print(images)
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(c_dim,
                            x_dim,
                            y_dim * len(images) + padding * (len(images) - 1))
        for i, image in enumerate(images):
            result[:, :, i * y_dim + i * padding:
                         (i + 1) * y_dim + i * padding].copy_(image)
        return result


def make_image(tensor):
    tensor = tensor.cpu().clamp(0, 1)
    if tensor.size(0) == 1:
        tensor = tensor.expand(3, tensor.size(1), tensor.size(2))
    # pdb.set_trace()
    from PIL import Image
    tensor = (tensor.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(tensor)


def save_image(filename, tensor):
    img = make_image(tensor)
    img.save(filename)


def save_tensors_image(filename, inputs, padding=1):
    images = image_tensor(inputs, padding)
    return save_image(filename, images)


def save_vq_image(file_name, x, x_tilde):
    # x : c * t * h * w (no bs)
    # concatenate all x and x_tilde frames side by side
    x = x.to(torch.uint8).permute(1, 0, 2, 3)
    x_tilde = x_tilde.to(torch.uint8).permute(1, 0, 2, 3)
    t = x.shape[0]
    x_concat = torch.cat([x[i] for i in range(t)], dim=2)  # horizontally
    x_tilde_concat = torch.cat([x_tilde[i] for i in range(t)], dim=2)  # horizontally

    # Concatenate x and x_tilde vertically
    final_image = torch.cat((x_concat, x_tilde_concat), dim=1)  # vertically

    final_image = final_image.permute(1, 2, 0)  # become hwc
    new_image = torch.zeros(final_image.shape[0], final_image.shape[1], 3).to(torch.uint8)
    for i in range(final_image.shape[2]):
        new_image[:, :, 2 - i] = final_image[:, :, -1 - i]

    new_image = Image.fromarray(new_image.numpy(), 'RGB')

    # Save the image
    new_image.save(file_name)


def clear_progressbar():
    # moves up 3 lines
    print("\033[2A")
    # deletes the whole line, regardless of character position
    print("\033[2K")
    # moves up two lines again
    print("\033[2A")
