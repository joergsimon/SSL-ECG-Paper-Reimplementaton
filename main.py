import src.data as data
import src.finetune_to_target as ftt
import src.pretext_training as pt
import src.run_example as re
import src.tests.ecgcnn_basic_tests as ecgcnn_tests

run_tests = False
run_hyperparam = True
run_example = True

if __name__ == '__main__':
    if run_hyperparam:
        pt.train_pretext_tune_task(num_samples=6)
        ftt.train_finetune_tune_task(data.DataSets.AMIGOS, 'test_123')
    if run_example:
        re.run_example(data.DataSets.AMIGOS, 'test_123')
    if run_tests:
        ecgcnn_tests.test_cnn_basic_dimensions()
        ecgcnn_tests.test_single_head_loss()
        ecgcnn_tests.test_heads_loss()
        ecgcnn_tests.test_ecg_network()
        ecgcnn_tests.test_augmentations()
