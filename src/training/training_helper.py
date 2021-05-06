import tqdm
import numpy as np


def iterate_batches(loader, optimizer, batch_size, train_on_gpu, compute_loss):
    def ass_loss(lt, ls):
        if lt is None:
            lt = ls
        else:
            lt += ls
        return lt
    total_loss = None
    for i_batch, (data, labels) in enumerate(tqdm.tqdm(loader, leave=False)):
        if data.shape[0] != batch_size:
            print('skipping too small batch')
            continue  # if not full batch, just continue
        if train_on_gpu:
            data, labels = data.cuda(), labels.cuda()
        optimizer.zero_grad()
        loss = compute_loss(data, labels)
        loss.backward()
        optimizer.step()
        total_loss = ass_loss(total_loss, loss / len(labels))
    l = total_loss.item() * len(data)
    return l


def std_train_loop(epochs, batch_size, train_loader, valid_loader, model, optimizer, compute_loss, save_model, train_on_gpu):
    valid_loss_min = np.Inf  # track change in validation loss

    for e in tqdm.tqdm(range(epochs)):

        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        model.train()
        train_loss += iterate_batches(train_loader, optimizer, batch_size, train_on_gpu, compute_loss)

        ######################
        # validate the model #
        ######################
        model.eval()
        valid_loss +=iterate_batches(valid_loader, optimizer, batch_size, train_on_gpu, compute_loss)

        # calculate average losses
        train_loss = train_loss / len(train_loader.sampler)
        valid_loss = valid_loss / len(valid_loader.sampler)

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            e, train_loss, valid_loss))

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            save_model()
            valid_loss_min = valid_loss
