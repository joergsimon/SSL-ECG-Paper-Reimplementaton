import src.data as data
import src.finetune_to_target as ftt
import src.pretext_training as pt
import src.pretext_train_one_clf as ptoc
import src.run_example as re
import src.tests.ecgcnn_basic_tests as ecgcnn_tests
from src.constants import Constants as c
import src.augmentations as aug
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-pren", "--num-pretext-experiments", type=int, default=20,
                        help="number of pretext trail to run, default 20")
    parser.add_argument("-finen", "--num-finetune-experiments", type=int, default=20,
                        help="number of finetune trail to run, default")
    parser.add_argument("-ru", "--run-unittests", action="store_true", help="run unit tests")
    parser.add_argument("-shp", "--skip-pretext-hyperparams", action="store_true",
                        help="skip hyperparam search for pretext training")
    parser.add_argument("-rs", "--run-single-augmentation", action="store_true",
                        help="runs only one augmentation for debugging")
    parser.add_argument("-ps", "--run-single-pretraining", action="store_true",
                        help="runs pretrining with one single set of parameters")
    parser.add_argument("-shf", "--skip-finetune-hyperparams", action="store_true",
                        help="skip hyperparam search for finetuning")
    parser.add_argument("-fs", "--run-single-finetuning", action="store_true",
                        help="runs finetuning with one single set of parameters")
    parser.add_argument("-rex", "--run-example-classification", action="store_true",
                        help="runs one classification as example how to use the system")
    args = parser.parse_args()

    if not args.skip_pretext_hyperparams:
        c.use_ray = True
        pt.train_pretext_tune_task(num_samples=args.num_pretext_experiments)
    if args.run_single_augmentation:
        c.use_ray = False
        ptoc.train_pretext_full_config(pt.good_params_for_single_run, aug.AugmentationTypes.TIME_WRAP, use_tune=False)
    if args.run_single_pretraining:
        c.use_ray = False
        pt.train_pretext_full_config(pt.good_params_for_single_run, use_tune=False)
    if not args.skip_finetune_hyperparams:
        c.use_ray = True
        ftt.train_finetune_tune_task(data.DataSets.AMIGOS, 'test_123', num_samples=args.num_finetune_experiments)
    c.use_ray = False
    if args.run_single_finetuning:
        ftt.finetune_to_target_full_config(ftt.good_params_for_single_run, checkpoint_dir=None, target_dataset=data.DataSets.AMIGOS, target_id='test_123')
    if args.run_example_classification:
        re.run_example(data.DataSets.AMIGOS, 'test_123')
    if args.run_unittests:
        ecgcnn_tests.test_cnn_basic_dimensions()
        ecgcnn_tests.test_single_head_loss()
        ecgcnn_tests.test_heads_loss()
        ecgcnn_tests.test_ecg_network()
        ecgcnn_tests.test_augmentations()
