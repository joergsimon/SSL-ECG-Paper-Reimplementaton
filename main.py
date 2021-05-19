import src.data as data
import src.finetune_to_target as ftt
import src.pretext_training as pt
import src.run_example as re
import src.tests.ecgcnn_basic_tests as ecgcnn_tests
from src.constants import Constants as c

run_tests = False
run_hyperparam = False
pre_train_single = True
fine_tune_single =True
run_example = True

if __name__ == '__main__':
    if run_hyperparam:
        pt.train_pretext_tune_task(num_samples=6)
        ftt.train_finetune_tune_task(data.DataSets.AMIGOS, 'test_123')
    else:
        c.use_ray = False
    if pre_train_single:
        pt.train_pretext_full_config(pt.good_params_for_single_run, use_tune=False)
    if fine_tune_single:
        ftt.finetune_to_target_full_config(ftt.good_params_for_single_run, checkpoint_dir=None, target_dataset=data.DataSets.AMIGOS, target_id='test_123', )
    if run_example:
        re.run_example(data.DataSets.AMIGOS, 'test_123')
    if run_tests:
        ecgcnn_tests.test_cnn_basic_dimensions()
        ecgcnn_tests.test_single_head_loss()
        ecgcnn_tests.test_heads_loss()
        ecgcnn_tests.test_ecg_network()
        ecgcnn_tests.test_augmentations()
