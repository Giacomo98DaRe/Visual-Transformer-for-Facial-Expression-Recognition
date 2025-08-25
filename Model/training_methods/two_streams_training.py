# IMPORT

import torch
import time
from tqdm import tqdm

########################################################################################################################
# TWO STREAMS MODEL TRAINING FUNCTION DEFINITION

print("Training started! \n\n")

def two_streams_train_model(model, device, num_epochs, criterion, optimizer, dataloaders, loss_vect, accuracy_vect):

    since = time.time()

    # Early stopping
    last_loss = 100.0
    patience = 5
    trigger_times = 0

    # Ovviamente best_acc
    best_acc = 0.0

    # Prediction vectors for confusion matrix
    correct_val_labels = []
    labels_predictions = []

    for epoch in range(num_epochs):

        print(f'Epoch {epoch}/{num_epochs - 1}')
        print("-" * 10)

        for phase in ["train", "val"]:  # Training and validation phase per epoch

            if phase == "train":
                model.train()  # model to training mode
            else:
                model.eval()  # model to evaluate

            running_loss = 0.0
            running_corrects = 0.0

            for landmark_channel_img, features_channel_img, label in tqdm(dataloaders[phase]):
                lci = landmark_channel_img.to(device)
                fci = features_channel_img.to(device)
                label = label.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"): 
                    output = model(lci, fci)
                    _, pred = torch.max(output, 1)
                    loss = criterion(output, label)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() + lci.size(0)  # -> batch_dim
                running_corrects += torch.sum(pred == label)  # Pred che tipo è? label è una "tupla??"

                if phase == 'val':
                    # Confusion matrix vectors
                    correct_val_labels.append(label.tolist())
                    labels_predictions.append(pred.tolist())

            # LOSS
            epoch_loss = running_loss / dataloaders[phase].dataset.__len__()

            # label = label.cpu()
            # pred = pred.cpu()

            # ACCURACY
            # epoch_acc = accuracy_score(label, pred)
            epoch_acc = running_corrects / dataloaders[phase].dataset.__len__()

            # F1 SCORE
            # epoch_f1 = f1_score(label, pred, average='weighted')  # -> TODO: AVERAGE?

            print("\n{} Loss: {:.4f} Acc: {:.4f}\n".format(phase, epoch_loss, epoch_acc))  # , epoch_f1))
            # logger1.info("Epoch {} -> {} Loss: {:.4f} Acc: {:.4f}\n".format(epoch, phase, epoch_loss, epoch_acc))  # , epoch_f1))

            loss_vect[phase].append(epoch_loss)
            accuracy_vect[phase].append(epoch_acc)

            # Load the best weights into the model for the next testing phase (if any).
            if phase == 'val' and epoch_acc > best_acc:
                # best_model_weights_path = os.path.join(run_path, f"best_model_epoch_{epoch}.pth")
                # torch.save(model.state_dict(), best_model_weights_path)
                best_acc = epoch_acc

            if phase == 'val':

            # CHECKPOINT SAVING
                checkpoint_handler(model, optimizer, run_path, epoch)

            # EARLY STOPPING
                current_loss = epoch_loss

                if current_loss > last_loss:
                    trigger_times += 1
                    print('Trigger Times:', trigger_times)
                else:
                    trigger_times = 0

                if trigger_times >= patience:
                    print('Early stopping!\nTraining finished.')
                    return model, zip(correct_val_labels, labels_predictions)

                if epoch == 0:
                    last_loss = epoch_loss
                else:
                    last_loss = current_loss

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print("Best Val Acc: {:.4f}\n".format(best_acc))

    return model, zip(correct_val_labels, labels_predictions)
