r"""
hparams.py
Hyper Parameters for the experiment
"""
from argparse import Namespace
import os


def create_hparams():
    r"""
    Model hyperparamters
    Returns:
        (argparse.Namespace): Hyperparameters
    """
    hparams = Namespace(

        ################################
        # Experiment Parameters        #
        ################################
        run_name="TanhReturns",
        seed=1234,
        # Important placeholder vital to load and save model
        logger=None,
        checkpoint_path="checkpoints/",
        check_val_every_n_epoch=1,
        # Can also have string "1,2,3" or list [1,2,3]
        gpus=[0],
        data_path="data",

        ################################
        # Data Parameters             #
        ################################
        num_workers=0,
        batch_size=64,

        ################################
        # Optimization Hyperparameters #
        ################################
        learning_rate=1e-3,
        weight_decay=1e-6,
        grad_clip_thresh=4.0,  # TODO: check Gradient Clipping


        ################################
        # Augmentation Parameters      #
        ################################
        training_random_rotation=30,
        normalizing_values_mu=[0.4821, 0.4787, 0.4517],
        normalizing_values_sigma=[0.3146, 0.3083, 0.3430],



        ################################
        # Model Parameters             #
        ################################
        # embedding_dim = 300 etc..
        n_channels=3,
        image_resized=(128, 128),
        latent_vector_size=100,
        feature_maps_generator=64,
        feature_maps_discriminator=64,



    )

    return hparams
