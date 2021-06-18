import src.data as data
import src.finetune_to_target as ftt
import src.pretext_training as pt
import src.pretext_train_one_clf as ptoc
import src.run_example as re
import src.tests.ecgcnn_basic_tests as ecgcnn_tests
from src.constants import Constants as c
import src.augmentations as aug

run_tests = False
run_pretrain_hyperparams = True
pre_train_single_augmentation = False
run_finetune_hyperparams = True
pre_train_single = False
fine_tune_single = False
run_example = False

if __name__ == '__main__':
    if run_pretrain_hyperparams:
        c.use_ray = True
        pt.train_pretext_tune_task(num_samples=20)
    if pre_train_single_augmentation:
        c.use_ray = False
        ptoc.train_pretext_full_config(pt.good_params_for_single_run, aug.AugmentationTypes.TIME_WRAP, use_tune=False)
    if pre_train_single:
        c.use_ray = False
        pt.train_pretext_full_config(pt.good_params_for_single_run, use_tune=False)
    if run_finetune_hyperparams:
        c.use_ray = True
        ftt.train_finetune_tune_task(data.DataSets.AMIGOS, 'test_123', num_samples=20)
    c.use_ray = False
    if fine_tune_single:
        ftt.finetune_to_target_full_config(ftt.good_params_for_single_run, checkpoint_dir=None, target_dataset=data.DataSets.AMIGOS, target_id='test_123')
    if run_example:
        re.run_example(data.DataSets.AMIGOS, 'test_123')
    if run_tests:
        ecgcnn_tests.test_cnn_basic_dimensions()
        ecgcnn_tests.test_single_head_loss()
        ecgcnn_tests.test_heads_loss()
        ecgcnn_tests.test_ecg_network()
        ecgcnn_tests.test_augmentations()
