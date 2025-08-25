# IMPORT

import torch
import time
from tqdm import tqdm
# from sklearn.metrics import accuracy_score, f1_score
from thesis_tools import plot_emotion_barchart, plot_emotion_barchart_plus_image, plot_emotion_scores

########################################################################################################################
# MODEL TRAINING FUNCTION DEFINITION

print("Training started! \n\n")

# """
def train_model(model, device, num_epochs, criterion, optimizer, dataloaders, loss_vect, accuracy_vect): #, best_model):
    # Time is used in model training to measure the execution time of a given piece of code
    since = time.time()

    # Early stopping
    last_loss = 100.0
    patience = 5
    trigger_times = 0

    best_acc = 0.0

    # Prediction vectors for confusion matrix
    correct_val_labels = []
    labels_predictions = []

    # Iterate for the entire number of epochs
    for epoch in range(num_epochs):

        # Print each epoch
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print("-" * 10)

	# IMPORTANT: The code is divided into "phases", namely train and val. Each phase is a dictionary label associated with            	# the corresponding dataloader.

        for phase in ["train", "val"]:  # We do training and validation phase per epoch

            if phase == "train":
                model.train()  # model to training mode
            else:
                model.eval()  # model to evaluate

            running_loss = 0.0
            running_corrects = 0.0

            for img, label in tqdm(dataloaders[phase]):
                image = img.to(device)
                label = label.to(device)

            # DATASET WITH TWO DIFFERENT TRANSFORMATIONS
            # for first_channel_img, second_channel_img, label in tqdm(dataloaders[phase]):
                # fci = first_channel_img.to(device)
                # sci = second_channel_img.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):  # no autograd makes validation go faster -> ?

                    #
                    output = model(image)

                    # Detach and convert to numpy
                    output_np = output.detach().numpy()

                    # plot_emotion_barchart(scores=output_np[0])
                    plot_emotion_scores(output_np)

		    # I assign to pred the maximum accuracy among the labels predicted by the model
                    _, pred = torch.max(output, 1)
                    # Loss calculus
                    loss = criterion(output, label)

                    if phase == 'train':
			# Update weights
                        loss.backward()
                        optimizer.step()

		# Get the cumulative loss value for each batch
                running_loss += loss.item() + image.size(0) # -> batch_dim
		# Increase the number of correctly predicted labels
                running_corrects += torch.sum(pred == label)  # Pred che tipo è? label è una "tupla??"

		# I UPDATE THE CLASS_CORRECT VECTOR WHICH WILL CONTAIN THE CORRECT PREDICTIONS FOR EACH CLASS IN THE VALIDATION SET.
		# AT THE END OF TRAINING I PRINT THE ACCURACY PERCENTAGES FOR EACH CLASS
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
            #epoch_f1 = f1_score(label, pred, average='weighted')  

            print("\n{} Loss: {:.4f} Acc: {:.4f}\n".format(phase, epoch_loss, epoch_acc)) #, epoch_f1))
            # logger1.info("Epoch {} -> {} Loss: {:.4f} Acc: {:.4f}\n".format(epoch, phase, epoch_loss, epoch_acc)) #, epoch_f1))

            loss_vect[phase].append(epoch_loss)
            accuracy_vect[phase].append(epoch_acc)

            # Load the best weights into the model for the next testing phase (if any).
            if phase == 'val': # and epoch_acc > best_acc:
                # torch.save(model.state_dict(), best_model)
                best_acc = epoch_acc

            # EARLY STOPPING
            if phase == 'val':
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

    # Print total training time
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # Print best val accuracy
    print("Best Val Acc: {:.4f}\n".format(best_acc))

    # Load the best weights into the model for the next testing phase
    # model.load_state_dict(best_model_wts)

    return model, zip(correct_val_labels, labels_predictions)

########################################################################################################################
# CHECKPOINT FUNCTION

'''
def checkpoint_handler(model, optimizer, run_path, epoch):

    checkpoint_folder = 'path_della_tua_cartella'
    checkpoint_path = os.path.join(run_path, checkpoint_folder)

    # Check if the folder already exists
    if not os.path.exists(checkpoint_path):
        # Se non esiste, crea la cartella
        os.makedirs(checkpoint_path)

    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict()
    }
'''

    torch.save(checkpoint, checkpoint_path)