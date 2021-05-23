import numpy as np
import src.utils as utils


def iterate_batches(loader, optimizer, batch_size, train_on_gpu: bool, compute_loss):
    total_loss = None
    total_accuracy = None
    for i_batch, (data, labels) in enumerate(utils.pbar(loader, leave=False)):
        if data.shape[0] != batch_size:
            #print('skipping too small batch')
            continue  # if not full batch, just continue
        if train_on_gpu:
            data, labels = data.cuda(), labels.cuda()
        optimizer.zero_grad()
        loss, accuracy = compute_loss(data, labels)
        loss.backward()
        optimizer.step()
        total_loss = utils.assign(total_loss, loss / len(labels))
        total_accuracy = utils.assign(total_accuracy, accuracy)
    l = total_loss.item()
    a = total_accuracy.item()
    return l, a


def std_train_loop(epochs, batch_size, train_loader, valid_loader, model, optimizer, compute_loss_and_accuracy, save_model, train_on_gpu: bool):
    valid_loss_min = np.Inf  # track change in validation loss

    for e in utils.pbar(range(epochs)):

        train_loss = 0.0
        valid_loss = 0.0

        train_accuracy = 0.0
        valid_accuracy = 0.0

        ###################
        # train the model #
        ###################
        model.train()
        l, a = iterate_batches(train_loader, optimizer, batch_size, train_on_gpu, compute_loss_and_accuracy)
        train_loss += l
        train_accuracy += a

        ######################
        # validate the model #
        ######################
        model.eval()
        l, a = iterate_batches(valid_loader, optimizer, batch_size, train_on_gpu, compute_loss_and_accuracy)
        valid_loss += l
        valid_accuracy += a

        # calculate average losses
        train_loss = train_loss / len(train_loader)
        valid_loss = valid_loss / len(valid_loader)

        train_accuracy = train_accuracy / len(train_loader)
        valid_accuracy = valid_accuracy / len(train_loader)

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}\n\t\tTraining Accuracy: {:.3f} \tValidation Accuracy: {:.3f}'.format(
            e, train_loss, valid_loss, train_accuracy, valid_accuracy))

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            save_model()
            valid_loss_min = valid_loss
