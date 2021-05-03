import src.tests.ecgcnn_basic_tests as ecgcnn_tests
import src.data as data
import src.datasets.amigos as amigos
import src.datasets.dreamer as dreamer
import src.datasets.wesad as wesad
import src.pretext_training as pt

run_tests = False

if __name__ == '__main__':
    #w, wl = dreamer.load_ecg_windows(data.DataConstants.basepath)
    #print(len(w), len(wl))
    # w, wl = wesad.load_ecg_windows(data.DataConstants.basepath)
    pt.train_pretext_full_config()
    #dataset = amigos.ECGAmigosCachedWindowsDataset(data.DataConstants.basepath, None, None, False)
    #print(len(dataset))
    #print(dataset[1])
    #w = data.load_ecg_windows(data.DataSets.AMIGOS, data.DataConstants.basepath)
    #print('w len', len(w))
    #print('w[0] shape', w[0].shape)
    #data.load_preprocessed_data(data.DataSets.AMIGOS, data.DataConstants.basepath)
    if run_tests:
        ecgcnn_tests.test_cnn_basic_dimensions()
        ecgcnn_tests.test_single_head_loss()
        ecgcnn_tests.test_heads_loss()
        ecgcnn_tests.test_ecg_network()

        ecgcnn_tests.test_augmentations()
