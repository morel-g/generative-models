from torchmetrics.image.fid import FrechetInceptionDistance
from torch.utils.data import DataLoader
import torch
import numpy as np
import os
from ...case import Case

try:
    from src.eval.fid import evaluation
    import tensorflow as tf
    import io
    import tensorflow_gan as tfgan
    import gc
except Exception as e:
    print(
        "Exception:",
        e,
        " Could not evaluate fid v1.",
    )


def compute_fid_v3(
    net, data_module, device, batch_size=500, num_samples=50000
):
    fid = FrechetInceptionDistance(feature=2048, normalize=True)
    if device != torch.device("cpu"):
        fid.cuda()

    train_loader = DataLoader(
        data_module.train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
        pin_memory=data_module.pin_memory,
    )
    for batch in train_loader:
        fid.update((batch[0].to(device) + 1.0) / 2.0, real=True)

    nb_iters = num_samples // batch_size
    for _ in range(nb_iters):
        xi = net.sample(batch_size, return_trajectories=False)
        fid.update(xi.to(device), real=False)

    fid_score = fid.compute()

    return fid_score


# From https://github.com/yang-song/score_sde and https://github.com/nv-tlabs/CLD-SGM
def compute_fid_v1(
    net,
    path_to_stats,
    dataset,
    batch_size=250,
    num_samples=50000,
    inceptionv3=False,
):
    if dataset == Case.cifar10:
        image_size, num_channels = 32, 3
    else:
        raise NotImplementedError(
            f"Image_size and num_channels unknown for dataset {dataset}."
        )
    num_sampling_rounds = num_samples // batch_size + 1
    data_stats = evaluation.load_dataset_stats(path_to_stats)
    inception_model = evaluation.get_inception_model(inceptionv3=inceptionv3)
    all_pools = []
    for r in range(num_sampling_rounds):
        samples = net.sample(batch_size, return_trajectories=False)
        if net.model.is_augmented():
            samples, _ = samples
        samples = np.clip(
            samples.permute(0, 2, 3, 1).cpu().numpy() * 255.0, 0, 255
        ).astype(np.uint8)
        samples = samples.reshape(
            (
                -1,
                image_size,
                image_size,
                num_channels,
            )
        )

        latents = evaluation.run_inception_distributed(
            samples, inception_model, inceptionv3=inceptionv3
        )
        all_pools.append(latents["pool_3"])

    all_pools = np.concatenate(all_pools, axis=0)

    # data_stats = evaluation.load_dataset_stats(config)
    data_pools = data_stats["pool_3"]
    data_pools_mean = np.mean(data_pools, axis=0)
    data_pools_sigma = np.cov(data_pools, rowvar=False)
    all_pool_mean = np.mean(all_pools, axis=0)
    all_pool_sigma = np.cov(all_pools, rowvar=False)

    fid = evaluation.calculate_frechet_distance(
        data_pools_mean, data_pools_sigma, all_pool_mean, all_pool_sigma
    )
    return fid
    """
    for r in range(num_sampling_rounds):
        # Directory to save samples. Different for each host to avoid writing conflicts
        this_sample_dir = output_dir
        tf.io.gfile.makedirs(this_sample_dir)
        samples = net.sample(batch_size, return_trajectories=False)
        samples = np.clip(
            samples.permute(0, 2, 3, 1).cpu().numpy() * 255.0, 0, 255
        ).astype(np.uint8)
        samples = samples.reshape(
            (
                -1,
                image_size,
                image_size,
                num_channels,
            )
        )
        # Write samples to disk or Google Cloud Storage
        with tf.io.gfile.GFile(
            os.path.join(this_sample_dir, f"samples_{r}.npz"), "wb"
        ) as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, samples=samples)
            fout.write(io_buffer.getvalue())

        # Force garbage collection before calling TensorFlow code for Inception network
        gc.collect()
        latents = evaluation.run_inception_distributed(
            samples, inception_model, inceptionv3=inceptionv3
        )
        # Force garbage collection again before returning to JAX code
        gc.collect()
        # Save latent represents of the Inception network to disk or Google Cloud Storage
        with tf.io.gfile.GFile(
            os.path.join(this_sample_dir, f"statistics_{r}.npz"), "wb"
        ) as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(
                io_buffer, pool_3=latents["pool_3"], logits=latents["logits"]
            )
            fout.write(io_buffer.getvalue())
    # Compute inception scores, FIDs and KIDs.
    # Load all statistics that have been previously computed and saved for each host
    all_logits = []
    all_pools = []
    this_sample_dir = output_dir
    stats = tf.io.gfile.glob(os.path.join(this_sample_dir, "statistics_*.npz"))
    for stat_file in stats:
        with tf.io.gfile.GFile(stat_file, "rb") as fin:
            stat = np.load(fin)
            if not inceptionv3:
                all_logits.append(stat["logits"])
            all_pools.append(stat["pool_3"])

    if not inceptionv3:
        all_logits = np.concatenate(all_logits, axis=0)[:num_samples]
    all_pools = np.concatenate(all_pools, axis=0)[:num_samples]
    # print("12345")
    # Load pre-computed dataset statistics.
    data_stats = evaluation.load_dataset_stats(path_to_stats)
    data_pools = data_stats["pool_3"]

    # Compute FID/KID/IS on all samples together.
    if not inceptionv3:
        inception_score = tfgan.eval.classifier_score_from_logits(all_logits)
    else:
        inception_score = -1

    fid = tfgan.eval.frechet_classifier_distance_from_activations(
        data_pools, all_pools
    )
    # Hack to get tfgan KID work for eager execution.
    tf_data_pools = tf.convert_to_tensor(data_pools)
    tf_all_pools = tf.convert_to_tensor(all_pools)
    kid = tfgan.eval.kernel_classifier_distance_from_activations(
        tf_data_pools, tf_all_pools
    ).numpy()
    del tf_data_pools, tf_all_pools
    return fid
    """
