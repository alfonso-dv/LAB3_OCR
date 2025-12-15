# #---------------------------------------------TASK 1 START ------------------------------------------------------------------------
# # keras imports for the dataset and building our neural network
# from keras.datasets import mnist
# from keras.models import Model, Sequential
# from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, BatchNormalization
# from keras.optimizers.schedules import ExponentialDecay
# from keras import callbacks
# from tensorflow.keras.optimizers import SGD, Adam
# #from keras.utils import np_utils
# from keras.utils import to_categorical
# from sklearn.metrics import ConfusionMatrixDisplay
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec
# import matplotlib
# from datetime import datetime
# import numpy as np
# import os
#
# def display_classification_report(classification_report, figure_path, figure_name, onscreen=True):
#     f = open(os.path.join(figure_path, figure_name+'.txt'), 'w')
#     f.write(classification_report)
#     f.close()
#
#     if onscreen:
#        print(classification_report)
#
# def display_confusion_matrix(confusion_matrix, labels, figure_path,figure_name,figure_format,onscreen=True):
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
#
#     disp.plot(cmap=plt.cm.Greys)
#
#     plt.savefig(os.path.join(figure_path,figure_name+'.'+figure_format), format=figure_format)
#
#     if onscreen:
#        if matplotlib.get_backend().lower() not in ["agg", "pdf", "svg", "ps", "png"]:
#           print("Show confusion matrix on display")
#
#           plt.show()
#        else:
#           print("Non-interactive backend; figure saved but not shown.")
#           plt.close(fig)
#     else:
#        plt.close(fig)
#
# def display_activations(input, label, activations, layer_names, figure_path, figure_name, figure_format, onscreen=True):
#     fig = plt.figure(layout='constrained', figsize=(10, 8))
#     subfigs = fig.subfigures(1, len(activations)+1) # layers + input , figsize=(row_size*2.5,len(model.layers)*1.5))
#
#     subfigs[0].subplots(1, 1)
#     subfigs[0].suptitle('Label: {}'.format(label))
#     axs = subfigs[0].get_axes()
#     axs[0].set_xticks([])
#     axs[0].set_yticks([])
#     axs[0].imshow(input, cmap='gray_r')
#
#     for layer_index in range(0,len(activations)):
#         print("layer:" +str(layer_index))
#         print(activations[layer_index].shape[-1])
#         subfigs[layer_index+1].suptitle(layer_names[layer_index])
#         subfigs[layer_index+1].subplots(activations[layer_index].shape[-1],1)
#
#     for layer_index in range(0,len(activations)):
#         print(activations[layer_index].shape)
#         #range(0,activations.shape[-1]):
#         axs = subfigs[layer_index+1].get_axes()
#         for plane_index in range(0,activations[layer_index].shape[-1]):
#             plane = activations[layer_index][0,:, :, plane_index]
#             axs[plane_index].set_xticks([])
#             axs[plane_index].set_yticks([])
#             axs[plane_index].imshow(plane, cmap='gray_r')
#
#     fig.savefig(os.path.join(figure_path,figure_name+'.'+figure_format), format=figure_format)
#     if onscreen:
#        if matplotlib.get_backend().lower() not in ["agg", "pdf", "svg", "ps", "png"]:
#           plt.show()
#        else:
#           print("Non-interactive backend; figure saved but not shown.")
#           plt.close(fig)
#     else:
#        plt.close(fig)
#
#
#
# def display_weights_column(weights, layer_names,figure_path,figure_name,figure_format,onscreen=True):
#     n_layers_with_weights = 0
#     for layer_index in range(0, len(weights)):
#         layer_weights = weights[layer_index]
#         if len(layer_weights) > 0:
#             n_layers_with_weights += 1
#
#     fig = plt.figure(figsize=(30, 15), frameon=False)
#
#     plt.subplots_adjust(wspace=0.1, hspace=0.1, top=1.0, bottom=0.0, left=0.0, right=1.0)
#     subfigs = fig.subfigures(1, n_layers_with_weights)
#     # if there is only one layer with weights, subfigures() returns a single SubFigure
#     if n_layers_with_weights == 1:
#         subfigs = [subfigs]
#     layer_index_with_weights = 0
#     print("Number of layers: "+str(len(weights)))
#     for layer_index in range(0, len(weights)):
#         layer_weights = weights[layer_index]
#
#         print("layer:" +str(layer_index))
#         # only weights (0) no biases (1)
#         if len(layer_weights) > 0 and len(layer_weights[0].shape) > 1:
#             print(" weights shape ", layer_weights[0].shape)
#
#             #subfigs[layer_index_with_weights].suptitle(layer_names[layer_index])
#             #subfigs[layer_index_with_weights].tight_layout()
#             # squeeze=False squeezing at all is done: the returned Axes object is always a 2D array containing
#             axs = subfigs[layer_index_with_weights].subplots(layer_weights[0].shape[-2], layer_weights[0].shape[-1], squeeze=False)#, sharex=True, sharey=True)
#             subfigs[layer_index_with_weights].subplots_adjust(wspace=0.1, hspace=0.1, top=1.0, bottom=0.0, left=0.0, right=1.0)
#             print(axs.shape)
#             for i in range(0, layer_weights[0].shape[-2]):
#                 for j in range(0, layer_weights[0].shape[-1]):
#                     w = layer_weights[0]
#                     axs[i,j].imshow(w[:,:,i,j], cmap='gray_r', interpolation='nearest')
#                     axs[i,j].axis("off")
#
#             layer_index_with_weights += 1
#     fig.savefig(os.path.join(figure_path,figure_name+'.'+figure_format), format=figure_format)
#
#     if onscreen:
#        if matplotlib.get_backend().lower() not in ["agg", "pdf", "svg", "ps", "png"]:
#           plt.show()
#        else:
#           print("Non-interactive backend; figure saved but not shown.")
#           plt.close(fig)
#     else:
#        plt.close(fig)
#
#
# def display_loss_function(history,figure_path,figure_name,figure_format,onscreen=True):
#     loss = history.history['loss']
#     val_loss = history.history['val_loss']
#     epochs = range(1, len(loss) + 1)
#     fig = plt.figure()
#     plt.plot(epochs, loss, color='red', label='Training loss')
#     plt.plot(epochs, val_loss, color='green', label='Validation loss')
#     plt.title('Training loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     fig.savefig(os.path.join(figure_path,figure_name+'.'+figure_format), format=figure_format)
#     if onscreen:
#        print("Show loss on display")
#        if matplotlib.get_backend().lower() not in ["agg", "pdf", "svg", "ps", "png"]:
#           plt.show()
#        else:
#           print("Non-interactive backend; figure saved but not shown.")
#           plt.close(fig)
#     else:
#        plt.close(fig)
#
# # loading the dataset
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
#
# # building the input vector from the 28x28 pixels
# X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
# X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
#
# print("X_train shape", X_train.shape)
# print("y_train shape", y_train.shape)
# print("X_test shape", X_test.shape)
# print("y_test shape", y_test.shape)
#
# # normalizing the data
# X_train /= 255
# X_test /= 255
#
# # one-hot encoding using keras' numpy-related utilities
# n_classes = 10
# print("Shape before one-hot encoding: ", y_train.shape)
# Y_train = to_categorical(y_train, n_classes)
# Y_test = to_categorical(y_test, n_classes)
# print("Shape after one-hot encoding: ", Y_train.shape)
# n_cnn1planes = 15
#
# # TASK 1: topology parameters
#
# n_cnn_layers = 3
#
# n_cnn1planes = 10
# n_cnn2planes = 20
# n_cnn3planes = 30
#
#
# n_cnn1kernel = 3
# n_poolsize = 1
#
#
# # Stride defines the step size at which the filter moves across the input during convolution.
# # A larger stride results in a reduction of the spatial dimensions of the output feature map.
# # Stride can be adjusted to control the level of downsampling in the network.
# # Stride is a critical parameter for controlling the spatial resolution of the feature maps and influencing the receptive field of the network.
# n_strides = 1
# n_dense = 100
# dropout = 0.3
#
# n_epochs=1
# # CHANGE FROM 1 TO MORE THAN 5
#
# n_epochs=20
#
# model_name = (
#     'CNN_T1_'
#     + f'layers{n_cnn_layers}'
#     + f'p1{n_cnn1planes}p2{n_cnn2planes}p3{n_cnn3planes}'
#     + '_KERNEL' + str(n_cnn1kernel)
#     + '_Epochs' + str(n_epochs)
# )
# # Now every run will produce CNN_T1_layers2_p1_10_p2_20_p3_30_..._loss.png, so you know which is which.
#
# model_name = 'CNN_Handwritten_OCR_CNN'+str(n_cnn1planes)+'_KERNEL'+str(n_cnn1kernel)+'_Epochs' + str(n_epochs)
# #model_name = 'CNN_Handwritten_OCR_CNN'+str(n_cnn1planes)+'_KERNEL'+str(n_cnn1kernel)+'_Epochs' + str(n_epochs)
# #figure_format='svg'
# figure_format='png'
# figure_path='./'
# log_path='./log'
#
# # building a linear stack of layers with the sequential model
# model = Sequential()
# # convolutional layer
# cnn1 = Conv2D(n_cnn1planes, kernel_size=(n_cnn1kernel,n_cnn1kernel), strides=(n_strides,n_strides), padding='valid', activation='relu', input_shape=(28,28,1))
# model.add(cnn1)
# model.add(MaxPool2D(pool_size=(n_poolsize,n_poolsize)))
#
# #model.add(Dropout(dropout))
#
# cnn2 = Conv2D(n_cnn1planes*2, kernel_size=(n_cnn1kernel,n_cnn1kernel), strides=(n_strides,n_strides), padding='valid', activation='relu')
# model.add(cnn2)
# model.add(MaxPool2D(pool_size=(n_poolsize,n_poolsize)))
#
# #model.add(Dropout(dropout))
#
# cnn3 = Conv2D(n_cnn1planes*4, kernel_size=(n_cnn1kernel,n_cnn1kernel), strides=(n_strides,n_strides), padding='valid', activation='relu')
# model.add(cnn3)
# model.add(MaxPool2D(pool_size=(n_poolsize,n_poolsize)))
#
# #model.add(Dropout(dropout))
#
# # ----- first conv + pooling block -----
# if n_cnn_layers >= 1:
#     cnn1 = Conv2D(
#         n_cnn1planes,
#         kernel_size=(n_cnn1kernel, n_cnn1kernel),
#         strides=(n_strides, n_strides),
#         padding='valid',
#         activation='relu',
#         input_shape=(28, 28, 1)
#     )
#     model.add(cnn1)
#     model.add(MaxPool2D(pool_size=(n_poolsize, n_poolsize)))
#     # model.add(Dropout(dropout))
#
# # ----- second conv + pooling block -----
# if n_cnn_layers >= 2:
#     cnn2 = Conv2D(
#         n_cnn2planes,
#         kernel_size=(n_cnn1kernel, n_cnn1kernel),
#         strides=(n_strides, n_strides),
#         padding='valid',
#         activation='relu'
#     )
#     model.add(cnn2)
#     model.add(MaxPool2D(pool_size=(n_poolsize, n_poolsize)))
#     # model.add(Dropout(dropout))
#
# # ----- third conv + pooling block -----
# if n_cnn_layers >= 3:
#     cnn3 = Conv2D(
#         n_cnn3planes,
#         kernel_size=(n_cnn1kernel, n_cnn1kernel),
#         strides=(n_strides, n_strides),
#         padding='valid',
#         activation='relu'
#     )
#     model.add(cnn3)
#     model.add(MaxPool2D(pool_size=(n_poolsize, n_poolsize)))
#     # model.add(Dropout(dropout))
# # flatten output of conv
# model.add(Flatten())
#
#
# model.add(Dropout(dropout))
#
# # hidden layer
# model.add(Dense(n_dense, activation='relu'))
# # output layer
# model.add(Dense(n_classes, activation='softmax'))
#
# # compiling the sequential model
#
# model_name += 'Optimzer' + 'SGD'
#
# # vary the constant learning rate
# model_name += 'LearningRate' + 'Constant'
# learning_rate = 0.001
#
# # OR use a learning rate scheduler that adapts the learning rate over the epochs of the training process
# # https://keras.io/2.15/api/optimizers/learning_rate_schedules/
#
# #model_name += 'LearningRate' + 'ExponentialDecay'
# #learning_rate = ExponentialDecay(initial_learning_rate=1e-2, decay_steps=n_epochs, decay_rate=0.9)
#
# #learning_rate=0.01
# momentum = 0.9
# optimizer=SGD(learning_rate = learning_rate, momentum = momentum)
#
# #optimizer=Adam(learning_rate = learning_rate)
#
# # vary the constant learning rate
# #learning_rate = 0.01
# #optimizer=SGD(learning_rate=learning_rate)
#
# model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)
#
# layer_names = [layer.name for layer in model.layers[:8]]
#
# weights = [layer.get_weights() for layer in model.layers[:4]]
# figure_name=model_name + '_initial_weights'
# display_weights_column(weights, layer_names, './', figure_name, figure_format, False )
#
# # training the model for n_epochs, use 10% of the training data as validation data
# history = model.fit(X_train, Y_train, validation_split = 0.1, batch_size=128, epochs=n_epochs )
#
# figure_name=model_name + '_loss'
# display_loss_function(history,'./',figure_name,figure_format)
#
#
# weights = [layer.get_weights() for layer in model.layers[:4]]
# figure_name=model_name + '_weights'
# display_weights_column(weights, layer_names, './', figure_name, figure_format, False )
#
# X_test_images = X_test[:2]
# for i in range(X_test_images.shape[0]):
#     Y_test_pred = model.predict(np.expand_dims(X_test[i], axis=0))
#
#     activation_model = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])
#     activations = activation_model.predict(np.expand_dims(X_test[i], axis=0))
#
#     figure_name=model_name + 'activations' + 'test_image_' + str(i)
#     display_activations(X_test[i],np.argmax(Y_test[i]),activations[:4], layer_names[:4], figure_path, figure_name, figure_format)
#
# y_test_pred = model.predict(X_test)
# cm = confusion_matrix(y_test, np.argmax(y_test_pred,axis=1), labels=range(0,n_classes))
#
# figure_name=model_name + '_confusion_matrix'
# display_confusion_matrix(cm, range(0,n_classes), figure_path,figure_name,figure_format)
#
# figure_name=model_name + '_classification_report'
# display_classification_report(classification_report(y_test, np.argmax(y_test_pred,axis=1), target_names=[str(c) for c in range(0,n_classes)], digits=4), figure_path, figure_name)
#
# figure_name=model_name + '_model_summary'
# stringlist = []
# model.summary(print_fn=lambda x: stringlist.append(x))
# model_summary = "\n".join(stringlist)
# display_classification_report(model_summary, figure_path, figure_name)
# #---------------------------------------------TASK 1 END ------------------------------------------------------------------------
#
#
#
#
#
#
#
#
#
#
# # ---------------------------------------------TASK 2 START ------------------------------------------------------------------------
# #keras imports for the dataset and building our neural network
# from keras.datasets import mnist
# from keras.models import Model, Sequential
# from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, BatchNormalization
# from keras.optimizers.schedules import ExponentialDecay
# from keras import callbacks
# from tensorflow.keras.optimizers import SGD, Adam
# #from keras.utils import np_utils
# from keras.utils import to_categorical
# from sklearn.metrics import ConfusionMatrixDisplay
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec
# import matplotlib
# from datetime import datetime
# import numpy as np
# import os
#
# def display_classification_report(classification_report, figure_path, figure_name, onscreen=True):
#     f = open(os.path.join(figure_path, figure_name + '.txt'), 'w', encoding='utf-8')
#     f.write(classification_report)
#     f.close()
#
#     if onscreen:
#        print(classification_report)
#
# def display_confusion_matrix(confusion_matrix, labels, figure_path,figure_name,figure_format,onscreen=True):
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
#
#     disp.plot(cmap=plt.cm.Greys)
#
#     plt.savefig(os.path.join(figure_path,figure_name+'.'+figure_format), format=figure_format)
#
#     if onscreen:
#        if matplotlib.get_backend().lower() not in ["agg", "pdf", "svg", "ps", "png"]:
#           print("Show confusion matrix on display")
#
#           plt.show()
#        else:
#           print("Non-interactive backend; figure saved but not shown.")
#           plt.close()
#     else:
#        plt.close()
#
# def display_activations(input, label, activations, layer_names, figure_path, figure_name, figure_format, onscreen=True):
#     fig = plt.figure(layout='constrained', figsize=(10, 8))
#     subfigs = fig.subfigures(1, len(activations)+1) # layers + input , figsize=(row_size*2.5,len(model.layers)*1.5))
#
#     subfigs[0].subplots(1, 1)
#     subfigs[0].suptitle('Label: {}'.format(label))
#     axs = subfigs[0].get_axes()
#     axs[0].set_xticks([])
#     axs[0].set_yticks([])
#     axs[0].imshow(input, cmap='gray_r')
#
#     for layer_index in range(0,len(activations)):
#         print("layer:" +str(layer_index))
#         print(activations[layer_index].shape[-1])
#         subfigs[layer_index+1].suptitle(layer_names[layer_index])
#         subfigs[layer_index+1].subplots(activations[layer_index].shape[-1],1)
#
#     for layer_index in range(0,len(activations)):
#         print(activations[layer_index].shape)
#         #range(0,activations.shape[-1]):
#         axs = subfigs[layer_index+1].get_axes()
#         for plane_index in range(0,activations[layer_index].shape[-1]):
#             plane = activations[layer_index][0,:, :, plane_index]
#             axs[plane_index].set_xticks([])
#             axs[plane_index].set_yticks([])
#             axs[plane_index].imshow(plane, cmap='gray_r')
#
#     fig.savefig(os.path.join(figure_path,figure_name+'.'+figure_format), format=figure_format)
#     if onscreen:
#        if matplotlib.get_backend().lower() not in ["agg", "pdf", "svg", "ps", "png"]:
#           plt.show()
#        else:
#           print("Non-interactive backend; figure saved but not shown.")
#           plt.close(fig)
#     else:
#        plt.close(fig)
#
#
#
# def display_weights_column(weights, layer_names,figure_path,figure_name,figure_format,onscreen=True):
#     n_layers_with_weights = 0
#     for layer_index in range(0, len(weights)):
#         layer_weights = weights[layer_index]
#         if len(layer_weights) > 0:
#             n_layers_with_weights += 1
#
#     fig = plt.figure(figsize=(30, 15), frameon=False)
#
#     plt.subplots_adjust(wspace=0.1, hspace=0.1, top=1.0, bottom=0.0, left=0.0, right=1.0)
#     subfigs = fig.subfigures(1, n_layers_with_weights)
#     # if there is only one layer with weights, subfigures() returns a single SubFigure
#     if n_layers_with_weights == 1:
#         subfigs = [subfigs]
#     layer_index_with_weights = 0
#     print("Number of layers: "+str(len(weights)))
#     for layer_index in range(0, len(weights)):
#         layer_weights = weights[layer_index]
#
#         print("layer:" +str(layer_index))
#         # only weights (0) no biases (1)
#         if len(layer_weights) > 0 and len(layer_weights[0].shape) > 1:
#             print(" weights shape ", layer_weights[0].shape)
#
#             #subfigs[layer_index_with_weights].suptitle(layer_names[layer_index])
#             #subfigs[layer_index_with_weights].tight_layout()
#             # squeeze=False squeezing at all is done: the returned Axes object is always a 2D array containing
#             axs = subfigs[layer_index_with_weights].subplots(layer_weights[0].shape[-2], layer_weights[0].shape[-1], squeeze=False)#, sharex=True, sharey=True)
#             subfigs[layer_index_with_weights].subplots_adjust(wspace=0.1, hspace=0.1, top=1.0, bottom=0.0, left=0.0, right=1.0)
#             print(axs.shape)
#             for i in range(0, layer_weights[0].shape[-2]):
#                 for j in range(0, layer_weights[0].shape[-1]):
#                     w = layer_weights[0]
#                     axs[i,j].imshow(w[:,:,i,j], cmap='gray_r', interpolation='nearest')
#                     axs[i,j].axis("off")
#
#             layer_index_with_weights += 1
#     fig.savefig(os.path.join(figure_path,figure_name+'.'+figure_format), format=figure_format)
#
#     if onscreen:
#        if matplotlib.get_backend().lower() not in ["agg", "pdf", "svg", "ps", "png"]:
#           plt.show()
#        else:
#           print("Non-interactive backend; figure saved but not shown.")
#           plt.close(fig)
#     else:
#        plt.close(fig)
#
#
# def display_loss_function(history,figure_path,figure_name,figure_format,onscreen=True):
#     loss = history.history['loss']
#     val_loss = history.history['val_loss']
#     epochs = range(1, len(loss) + 1)
#     fig = plt.figure()
#     plt.plot(epochs, loss, color='red', label='Training loss')
#     plt.plot(epochs, val_loss, color='green', label='Validation loss')
#     plt.title('Training loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     fig.savefig(os.path.join(figure_path,figure_name+'.'+figure_format), format=figure_format)
#     if onscreen:
#        print("Show loss on display")
#        if matplotlib.get_backend().lower() not in ["agg", "pdf", "svg", "ps", "png"]:
#           plt.show()
#        else:
#           print("Non-interactive backend; figure saved but not shown.")
#           plt.close(fig)
#     else:
#        plt.close(fig)
#
# # loading the dataset
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
#
# # building the input vector from the 28x28 pixels
# X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
# X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
#
# print("X_train shape", X_train.shape)
# print("y_train shape", y_train.shape)
# print("X_test shape", X_test.shape)
# print("y_test shape", y_test.shape)
#
# # normalizing the data
# X_train /= 255
# X_test /= 255
#
# # one-hot encoding using keras' numpy-related utilities
# n_classes = 10
# print("Shape before one-hot encoding: ", y_train.shape)
# Y_train = to_categorical(y_train, n_classes)
# Y_test = to_categorical(y_test, n_classes)
# print("Shape after one-hot encoding: ", Y_train.shape)
#
# # TASK 1: topology parameters
#
# n_cnn_layers = 3
#
# n_cnn1planes = 10
# n_cnn2planes = 20
# n_cnn3planes = 30
#
#
# n_cnn1kernel = 3
# n_poolsize = 1
#
#
# # Stride defines the step size at which the filter moves across the input during convolution.
# # A larger stride results in a reduction of the spatial dimensions of the output feature map.
# # Stride can be adjusted to control the level of downsampling in the network.
# # Stride is a critical parameter for controlling the spatial resolution of the feature maps and influencing the receptive field of the network.
# n_strides = 1
# n_dense = 100
# dropout = 0.3
#
# # CHANGE FROM 1 TO MORE THAN 5
#
# n_epochs=20
#
#
# # für task 1, auskomenntieren, wenn du task 1 testen willst
# # model_name = (
# #     'CNN_T1_'
# #     + f'layers{n_cnn_layers}'
# #     + f'_p1_{n_cnn1planes}_p2_{n_cnn2planes}_p3_{n_cnn3planes}'
# #     + '_KERNEL' + str(n_cnn1kernel)
# #     + '_Epochs' + str(n_epochs)
# # )
#
# # ============================================================
# # TASK 2: LEARNING RATE
# # ============================================================
# # Task 2.1: Learning rate controls the step size of weight updates
# # Task 2.2: Use SGD as optimizer
# # Task 2.3: Learning rate will be varied between [0.001, 0.01]
# # Task 2.4: model_name includes learning rate for identification
#
# learning_rate = 0.01   # <- diesen Wert änderst du für die 4 Runs, 0.001, 0.003, 0.005, 0.01
#
# model_name = (
#     'CNN_T2_LR_' + str(learning_rate) + '_'
#     + f'layers{n_cnn_layers}'
#     + f'_p1_{n_cnn1planes}_p2_{n_cnn2planes}_p3_{n_cnn3planes}'
#     + '_KERNEL' + str(n_cnn1kernel)
#     + '_Epochs' + str(n_epochs)
# )
#
# optimizer = SGD(learning_rate=learning_rate)
#
#
#
#
#
# # Now every run will produce CNN_T1_layers2_p1_10_p2_20_p3_30_..._loss.png, so you know which is which.
#
# #model_name = 'CNN_Handwritten_OCR_CNN'+str(n_cnn1planes)+'_KERNEL'+str(n_cnn1kernel)+'_Epochs' + str(n_epochs)
# #figure_format='svg'
# figure_format='png'
# figure_path='./'
# log_path='./log'
#
# # building a linear stack of layers with the sequential model
# model = Sequential()
# # ----- first conv + pooling block -----
# if n_cnn_layers >= 1:
#     cnn1 = Conv2D(
#         n_cnn1planes,
#         kernel_size=(n_cnn1kernel, n_cnn1kernel),
#         strides=(n_strides, n_strides),
#         padding='valid',
#         activation='relu',
#         input_shape=(28, 28, 1)
#     )
#     model.add(cnn1)
#     model.add(MaxPool2D(pool_size=(n_poolsize, n_poolsize)))
#     # model.add(Dropout(dropout))
#
# # ----- second conv + pooling block -----
# if n_cnn_layers >= 2:
#     cnn2 = Conv2D(
#         n_cnn2planes,
#         kernel_size=(n_cnn1kernel, n_cnn1kernel),
#         strides=(n_strides, n_strides),
#         padding='valid',
#         activation='relu'
#     )
#     model.add(cnn2)
#     model.add(MaxPool2D(pool_size=(n_poolsize, n_poolsize)))
#     # model.add(Dropout(dropout))
#
# # ----- third conv + pooling block -----
# if n_cnn_layers >= 3:
#     cnn3 = Conv2D(
#         n_cnn3planes,
#         kernel_size=(n_cnn1kernel, n_cnn1kernel),
#         strides=(n_strides, n_strides),
#         padding='valid',
#         activation='relu'
#     )
#     model.add(cnn3)
#     model.add(MaxPool2D(pool_size=(n_poolsize, n_poolsize)))
#     # model.add(Dropout(dropout))
# # flatten output of conv
# model.add(Flatten())
#
#
# model.add(Dropout(dropout))
#
# # hidden layer
# model.add(Dense(n_dense, activation='relu'))
# # output layer
# model.add(Dense(n_classes, activation='softmax'))
#
# # compiling the sequential model
#
# model_name += '_Optimzer_' + 'SGD'
#
# # vary the constant learning rate
# #task 1
# #model_name += '_LearningRate_' + 'Constant'
# #learning_rate = 0.001
#
#
# # OR use a learning rate scheduler that adapts the learning rate over the epochs of the training process
# # https://keras.io/2.15/api/optimizers/learning_rate_schedules/
#
# #model_name += '_LearningRate_' + 'ExponentialDecay'
# #learning_rate = ExponentialDecay(initial_learning_rate=1e-2, decay_steps=n_epochs, decay_rate=0.9)
#
# #learning_rate=0.01
# #task 1
# #momentum = 0.9
# #optimizer=SGD(learning_rate = learning_rate, momentum = momentum)
#
# #optimizer=Adam(learning_rate = learning_rate)
#
# # vary the constant learning rate
# #learning_rate = 0.01
# #optimizer=SGD(learning_rate=learning_rate)
#
# #task 1
# #model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)
#
# #task 2
# model.compile(
#     loss='categorical_crossentropy',
#     metrics=['accuracy'],
#     optimizer=optimizer
# )
#
# layer_names = [layer.name for layer in model.layers[:8]]
#
# weights = [layer.get_weights() for layer in model.layers[:4]]
# figure_name=model_name + '_initial_weights'
# display_weights_column(weights, layer_names, './', figure_name, figure_format, False )
#
# # training the model for n_epochs, use 10% of the training data as validation data
# history = model.fit(X_train, Y_train, validation_split = 0.1, batch_size=128, epochs=n_epochs )
#
# figure_name=model_name + '_loss'
# display_loss_function(history,'./',figure_name,figure_format)
#
#
# weights = [layer.get_weights() for layer in model.layers[:4]]
# figure_name=model_name + '_weights'
# display_weights_column(weights, layer_names, './', figure_name, figure_format, False )
#
# X_test_images = X_test[:2]
# for i in range(X_test_images.shape[0]):
#     Y_test_pred = model.predict(np.expand_dims(X_test[i], axis=0))
#
#     activation_model = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])
#     activations = activation_model.predict(np.expand_dims(X_test[i], axis=0))
#
#     figure_name=model_name + '_activations_' + 'test_image_' + str(i)
#     display_activations(X_test[i],np.argmax(Y_test[i]),activations[:4], layer_names[:4], figure_path, figure_name, figure_format)
#
# y_test_pred = model.predict(X_test)
# cm = confusion_matrix(y_test, np.argmax(y_test_pred,axis=1), labels=range(0,n_classes))
#
# figure_name=model_name + '_confusion_matrix'
# display_confusion_matrix(cm, range(0,n_classes), figure_path,figure_name,figure_format)
#
# figure_name=model_name + '_classification_report'
# display_classification_report(classification_report(y_test, np.argmax(y_test_pred,axis=1), target_names=[str(c) for c in range(0,n_classes)], digits=4), figure_path, figure_name)
#
# figure_name=model_name + '_model_summary'
# stringlist = []
# model.summary(print_fn=lambda x: stringlist.append(x))
# model_summary = "\n".join(stringlist)
# display_classification_report(model_summary, figure_path, figure_name)
# #---------------------------------------------TASK 2 END ------------------------------------------------------------------------
#
#
#
# #
#
#
# #---------------------------------------------TASK 3 START ------------------------------------------------------------------------
# import os
#
# # ===== Matplotlib Backend (Fenster + PNG, kein Tk) =====
# import matplotlib
# matplotlib.use("Qt5Agg")   # MUSS vor pyplot stehen
#
# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec
#
# # === Keras / TensorFlow ===
# # Importiert die benötigten Keras-Module, um ein CNN aufzubauen:
# from keras.datasets import mnist
# from keras.models import Model, Sequential
# from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, BatchNormalization
# # Conv2D erkennt Bildmerkmale, MaxPool2D reduziert Bildgröße, Flatten erzeugt Vektor, Dense führt Klassifikation durch
#
# # Importiert die Lernraten-Schedule "ExponentialDecay", die die Lernrate automatisch reduziert:
# from keras.optimizers.schedules import ExponentialDecay # SGD aktualisiert die Gewichte nach jedem Batch
# from keras import callbacks
# from tensorflow.keras.optimizers import SGD, Adam # SGD aktualisiert die Gewichte nach jedem Batch
#
# from keras.utils import to_categorical
#
# # === Evaluation ===
# from sklearn.metrics import ConfusionMatrixDisplay
# from sklearn.metrics import classification_report, confusion_matrix
#
# # === Utils ===
# from datetime import datetime
# import numpy as np
#
#
#
# def display_classification_report(classification_report, figure_path, figure_name, onscreen=True):
#     f = open(os.path.join(figure_path, figure_name + '.txt'), 'w', encoding='utf-8')
#     f.write(classification_report)
#     f.close()
#
#     if onscreen:
#        print(classification_report)
#
# def display_confusion_matrix(confusion_matrix, labels, figure_path, figure_name, figure_format, onscreen=True):
#     fig, ax = plt.subplots()
#
#     disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=labels)
#     disp.plot(cmap=plt.cm.Greys, ax=ax)
#
#     fig.savefig(os.path.join(figure_path, figure_name + '.' + figure_format),
#                 format=figure_format)
#
#     if onscreen:
#         plt.show()
#
#     plt.close(fig)
#
#
# def display_activations(input, label, activations, layer_names, figure_path, figure_name, figure_format, onscreen=True):
#     fig = plt.figure(layout='constrained', figsize=(10, 8))
#     subfigs = fig.subfigures(1, len(activations)+1) # layers + input , figsize=(row_size*2.5,len(model.layers)*1.5))
#
#     subfigs[0].subplots(1, 1)
#     subfigs[0].suptitle('Label: {}'.format(label))
#     axs = subfigs[0].get_axes()
#     axs[0].set_xticks([])
#     axs[0].set_yticks([])
#     axs[0].imshow(input, cmap='gray_r')
#
#     for layer_index in range(0,len(activations)):
#         print("layer:" +str(layer_index))
#         print(activations[layer_index].shape[-1])
#         subfigs[layer_index+1].suptitle(layer_names[layer_index])
#         subfigs[layer_index+1].subplots(activations[layer_index].shape[-1],1)
#
#     for layer_index in range(0,len(activations)):
#         print(activations[layer_index].shape)
#         #range(0,activations.shape[-1]):
#         axs = subfigs[layer_index+1].get_axes()
#         for plane_index in range(0,activations[layer_index].shape[-1]):
#             plane = activations[layer_index][0,:, :, plane_index]
#             axs[plane_index].set_xticks([])
#             axs[plane_index].set_yticks([])
#             axs[plane_index].imshow(plane, cmap='gray_r')
#
#     fig.savefig(os.path.join(figure_path,figure_name+'.'+figure_format), format=figure_format)
#     if onscreen:
#         plt.show()
#
#     plt.close(fig)
#
#
# def display_weights_column(weights, layer_names,figure_path,figure_name,figure_format,onscreen=True):
#     n_layers_with_weights = 0
#     for layer_index in range(0, len(weights)):
#         layer_weights = weights[layer_index]
#         if len(layer_weights) > 0:
#             n_layers_with_weights += 1
#
#     fig = plt.figure(figsize=(30, 15), frameon=False)
#
#     plt.subplots_adjust(wspace=0.1, hspace=0.1, top=1.0, bottom=0.0, left=0.0, right=1.0)
#     subfigs = fig.subfigures(1, n_layers_with_weights)
#     # if there is only one layer with weights, subfigures() returns a single SubFigure
#     if n_layers_with_weights == 1:
#         subfigs = [subfigs]
#     layer_index_with_weights = 0
#     print("Number of layers: "+str(len(weights)))
#     for layer_index in range(0, len(weights)):
#         layer_weights = weights[layer_index]
#
#         print("layer:" +str(layer_index))
#         # only weights (0) no biases (1)
#         if len(layer_weights) > 0 and len(layer_weights[0].shape) > 1:
#             print(" weights shape ", layer_weights[0].shape)
#
#             #subfigs[layer_index_with_weights].suptitle(layer_names[layer_index])
#             #subfigs[layer_index_with_weights].tight_layout()
#             # squeeze=False squeezing at all is done: the returned Axes object is always a 2D array containing
#             axs = subfigs[layer_index_with_weights].subplots(layer_weights[0].shape[-2], layer_weights[0].shape[-1], squeeze=False)#, sharex=True, sharey=True)
#             subfigs[layer_index_with_weights].subplots_adjust(wspace=0.1, hspace=0.1, top=1.0, bottom=0.0, left=0.0, right=1.0)
#             print(axs.shape)
#             for i in range(0, layer_weights[0].shape[-2]):
#                 for j in range(0, layer_weights[0].shape[-1]):
#                     w = layer_weights[0]
#                     axs[i,j].imshow(w[:,:,i,j], cmap='gray_r', interpolation='nearest')
#                     axs[i,j].axis("off")
#
#             layer_index_with_weights += 1
#     fig.savefig(os.path.join(figure_path, figure_name + '.' + figure_format),
#                 format=figure_format)
#
#     if onscreen:
#         plt.show()
#
#     plt.close(fig)
#
#
# def display_loss_function(history, figure_path, figure_name, figure_format, onscreen=True):
#     loss = history.history['loss']
#     val_loss = history.history['val_loss']
#     epochs = range(1, len(loss) + 1)
#
#     fig = plt.figure()
#     plt.plot(epochs, loss, color='red', label='Training loss')
#     plt.plot(epochs, val_loss, color='green', label='Validation loss')
#     plt.title('Training loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#
#     fig.savefig(os.path.join(figure_path, figure_name + '.' + figure_format),
#                 format=figure_format)
#
#     if onscreen:
#         plt.show()
#
#     plt.close(fig)
#
#
# # loading the dataset
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
#
# # building the input vector from the 28x28 pixels
# X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
# X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
#
# print("X_train shape", X_train.shape)
# print("y_train shape", y_train.shape)
# print("X_test shape", X_test.shape)
# print("y_test shape", y_test.shape)
#
# # normalizing the data
# X_train /= 255
# X_test /= 255
#
# # one-hot encoding using keras' numpy-related utilities
# n_classes = 10
# print("Shape before one-hot encoding: ", y_train.shape)
# Y_train = to_categorical(y_train, n_classes)
# Y_test = to_categorical(y_test, n_classes)
# print("Shape after one-hot encoding: ", Y_train.shape)
#
# # TASK 1: topology parameters
#
# n_cnn_layers = 3
#
# n_cnn1planes = 10
# n_cnn2planes = 20
# n_cnn3planes = 30
#
#
# n_cnn1kernel = 3
# n_poolsize = 1
#
#
# # Stride defines the step size at which the filter moves across the input during convolution.
# # A larger stride results in a reduction of the spatial dimensions of the output feature map.
# # Stride can be adjusted to control the level of downsampling in the network.
# # Stride is a critical parameter for controlling the spatial resolution of the feature maps and influencing the receptive field of the network.
# n_strides = 1
# n_dense = 100
# dropout = 0.3
#
# # CHANGE FROM 1 TO MORE THAN 5
#
# n_epochs=2
#
#
# # für task 1, auskomenntieren, wenn du task 1 testen willst
# # model_name = (
# #     'CNN_T1_'
# #     + f'layers{n_cnn_layers}'
# #     + f'p1{n_cnn1planes}p2{n_cnn2planes}p3{n_cnn3planes}'
# #     + '_KERNEL' + str(n_cnn1kernel)
# #     + '_Epochs' + str(n_epochs)
# # )
#
#
#
# # ============================================================
# # TASK 3: LEARNING RATE SCHEDULES (SGD + ExponentialDecay)
# # ============================================================
#
#
# # --- eigener Output-Ordner nur für Task 3 ---
# # Definiert den Ordnerpfad, in dem alle Grafiken von Task 3 gespeichert werden:
# figure_path = './task3/'  # Alle Loss-Plots von Task 3 landen in diesem Unterordner
#
# # Erstellt den Ordner, falls er noch nicht existiert:
# os.makedirs(figure_path, exist_ok=True)  # exist_ok=True verhindert Fehler, wenn Ordner schon da ist
#
# # Legt Dateiformat für gespeicherten Grafiken fest:
# figure_format = 'png'  # Plots werden als PNG-Bilder gespeichert
#
#
# # ------------------------------------------------------------
# # MANUELL: pro Run ändern (mind. 4 Werte insgesamt testen)
# # ------------------------------------------------------------
# # Start-Lernrate -> soll für verschiedene Runs geändert werden:
# initial_lr = 0.1   # = Bsp, auch zb: 0.001 | 0.01 | 0.05 | 0.1
#
# # Definiert die Batchgröße: beeinflusst, wie viele Trainingsschritte pro Epoche durchgeführt werden.
# batch_size = 128
#
# # Berechnet, wie viele Batches pro Epoche entstehen (wichtig für Lernraten-Schedule):
# steps_per_epoch = X_train.shape[0] // batch_size  # X_train.shape[0]-> Anzahl Trainingsbilder & batch_size: Bilder pro Batch
#
#
# # Erstellt ExponentialDecay-Schedule für Lernrate:
# # ExponentialDecay sorgt dafür, dass Lernrate im Laufe des Trainings automatisch kleiner wird
# learning_rate = ExponentialDecay(
#     initial_learning_rate=initial_lr,    # Startwert der Lernrate - pro run verändert
#     decay_steps=steps_per_epoch,         # Nach dieser Anzahl Schritte wird Lernrate reduziert
#     decay_rate=0.9                       # Multiplikator, um wie viel Lernrate sinkt (0.9 = 10% Reduktion)
# )
# # Ergebnis: zu Beginn größere Lernschritte,später kleinere+ stabilere Updates
#
#
# # ------------------------------------------------------------
# # NEUES MODELL (frische Gewichte für fairen Vergleich!)
# # ------------------------------------------------------------
#
#
# # Erstellt ein neues leeres Modell -> in das Layers nacheinander eingefügt werden
# model = Sequential() # Sequential: jede Schicht wird linear hintereinander angeordnet
#
# # Fügt erste Bild-Erkennungs-Schicht hinzu - sie erkennt einfache Bildmerkmale wie Kanten und Linien
# if n_cnn_layers >= 1:   # wird nur ausgeführt, wenn mindestens 1 CNN-Layer aktiviert ist
#     model.add(Conv2D(
#         n_cnn1planes, (n_cnn1kernel, n_cnn1kernel),
#         activation='relu',  # macht  Modell nicht-linear, damit es auch komplexe Bildmuster lernen kann
#         input_shape=(28, 28, 1)))
#
#     model.add(MaxPool2D())      # verkleinert Bild + behält nur die wichtigsten Merkmale
#
#
# # Fügt zweite Bild-Erkennungs-Schicht hinzu
# # Hier erkennt das Modell bereits komplexere Muster
# if n_cnn_layers >= 2:   #falls 2 CNN-Layer aktiviert sind
#     model.add(Conv2D(n_cnn2planes, (n_cnn1kernel, n_cnn1kernel),
#                      activation='relu'))        # 2. Convolution: erkennt komplexere Muster (zb: Kurven, Kombinationen)
#     model.add(MaxPool2D())
#
#
# # Fügt dritte Convolution-Schicht + Pooling hinzu
# # Diese Schicht erkennt sehr komplexe Merkmale wie ganze Zahlenformen
# if n_cnn_layers >= 3:
#     model.add(Conv2D(n_cnn3planes, (n_cnn1kernel, n_cnn1kernel),
#                      activation='relu'))
#     model.add(MaxPool2D())
#
#
# # wandelt 2D-Feature Maps in einen 1D-Vektor um
# model.add(Flatten())   # damit Dense-Schichten sie verarbeiten können
#
# # Fügt eine Rechen-Schicht hinzu, die alle gelernten Bildmerkmale zsmführt
# model.add(Dense(n_dense, activation='relu'))   # Diese Schicht verarbeitet die Merkmale und bereitet die Entscheidung vor.
#
#
# # Fügt letzte Schicht hinzu, die entscheidet, welche Zahl erkannt wurde:
# model.add(Dense(n_classes, activation='softmax'))   # gibt für jede Zahl von 0 bis 9 eine Wahrscheinlichkeit aus
#
#
# # ------------------------------------------------------------
# # Compile + Train
# # ------------------------------------------------------------
# # Erstellt einen eindeutigen Modellnamen zur Identifikation des Experiments
# model_name = (
#     'CNN_T3_ExpDecay_LR_' + str(initial_lr) + '_'     # Kennzeichnet Task 3 + ExponentialDecay + verwendete Start-Lernrate
#     + f'layers{n_cnn_layers}'                         # Gibt an, wie viele Convolution-Layer das Modell besitzt
#     + f'_p1_{n_cnn1planes}_p2_{n_cnn2planes}_p3_{n_cnn3planes}'   # Speichert Anzahl der Feature-Maps (Filter) pro Conv-Layer
#     + '_KERNEL' + str(n_cnn1kernel)                   # Gibt Kernelgröße der Convolution-Filter an
#     + '_Epochs' + str(n_epochs)                       # Speichert Anzahl der Trainings-Epochen
# )   # Ergebnis: Jede Ausgabedatei enthält alle wichtigen Hyperparameter im Dateinamen
#
#
# # Erstellt SGD-Optimizer mit zuvor definierten ExponentialDecay-Lernratenfunktion.
# optimizer = SGD(learning_rate=learning_rate)   # SGD aktualisiert Gewichte; die Lernrate ändert sich dynamisch während des Trainings.
#
#
# # Kompiliert das Modell: definiert Loss-Funktion, Metriken und Optimizer.
# model.compile(
#     loss='categorical_crossentropy',     # Crossentropy misst Abweichung der vorhergesagten Wahrscheinlichkeiten von den Labels.
#     metrics=['accuracy'],              # Accuracy zeigt, wie viele Bilder korrekt klassifiziert wurden.
#     optimizer=optimizer           # Verwendet SGD + ExponentialDecay
# )
#
#
# # Startet training vom Modell:
# history = model.fit(
#     X_train,                              # enthält Trainingsbilder
#     Y_train,                              # Enthält One-Hot-Labels
#     validation_split=0.1,                 # Reserviert 10% der Daten werden automatisch als Validierungsset verwendet.
#     batch_size=batch_size,                # gibt an wie viele Bilder gleichzeitig verarbeitet werden, bevor Gewichte aktualisiert werden
#     epochs=n_epochs,                      # Bestimmt Anzahl der Trainingsdurchläufe.
#     verbose=1                             # zeigt Trainingsfortschritt pro Epoche an
# )   # history speichert Verlauf von Training Loss und Validation Loss für jede Epoche
#
#
# # ------------------------------------------------------------
# # REQUIRED: analyse loss (train + validation)
# # ------------------------------------------------------------
# # Erstellt Grafik zur Analyse des Lernverhaltens des Modells-> zeigt, wie sich Fehler während des Trainings entwickelt.
# display_loss_function(
#     history=history,                       # enthält Trainings- und Validierungs-Loss.
#     figure_path=figure_path,               # gibt Speicherort des Plots an
#     figure_name=model_name + '_loss',      # setzt Dateiname des Plots
#     figure_format=figure_format,           # definiert Dateiformat (PNG)
#     onscreen=True                          # True zeigt Plot direkt im Fenster an
# )   #  Grafik zeigt Training Loss vs. Validation Loss
# #---------------------------------------------TASK 3 END ------------------------------------------------------------------------
#


#
# #
#
# #---------------------------------------------TASK 4 START ------------------------------------------------------------------------
# import os
#
# # ===== Matplotlib Backend (Fenster + PNG, kein Tk) =====
# import matplotlib
# matplotlib.use("Qt5Agg")   # MUSS vor pyplot stehen
#
# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec
#
# # === Keras / TensorFlow ===
# from keras.datasets import mnist
# from keras.models import Model, Sequential
# from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, BatchNormalization
# from keras.optimizers.schedules import ExponentialDecay
# from keras import callbacks
# from tensorflow.keras.optimizers import SGD, Adam
# from keras.utils import to_categorical
#
# # === Evaluation ===
# from sklearn.metrics import ConfusionMatrixDisplay
# from sklearn.metrics import classification_report, confusion_matrix
#
# # === Utils ===
# from datetime import datetime
# import numpy as np
#
#
#
# def display_classification_report(classification_report, figure_path, figure_name, onscreen=True):
#     f = open(os.path.join(figure_path, figure_name + '.txt'), 'w', encoding='utf-8')
#     f.write(classification_report)
#     f.close()
#
#     if onscreen:
#        print(classification_report)
#
# def display_confusion_matrix(confusion_matrix, labels, figure_path, figure_name, figure_format, onscreen=True):
#     fig, ax = plt.subplots()
#
#     disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=labels)
#     disp.plot(cmap=plt.cm.Greys, ax=ax)
#
#     fig.savefig(os.path.join(figure_path, figure_name + '.' + figure_format),
#                 format=figure_format)
#
#     if onscreen:
#         plt.show()
#
#     plt.close(fig)
#
#
# def display_activations(input, label, activations, layer_names, figure_path, figure_name, figure_format, onscreen=True):
#     fig = plt.figure(layout='constrained', figsize=(10, 8))
#     subfigs = fig.subfigures(1, len(activations)+1) # layers + input , figsize=(row_size*2.5,len(model.layers)*1.5))
#
#     subfigs[0].subplots(1, 1)
#     subfigs[0].suptitle('Label: {}'.format(label))
#     axs = subfigs[0].get_axes()
#     axs[0].set_xticks([])
#     axs[0].set_yticks([])
#     axs[0].imshow(input, cmap='gray_r')
#
#     for layer_index in range(0,len(activations)):
#         print("layer:" +str(layer_index))
#         print(activations[layer_index].shape[-1])
#         subfigs[layer_index+1].suptitle(layer_names[layer_index])
#         subfigs[layer_index+1].subplots(activations[layer_index].shape[-1],1)
#
#     for layer_index in range(0,len(activations)):
#         print(activations[layer_index].shape)
#         #range(0,activations.shape[-1]):
#         axs = subfigs[layer_index+1].get_axes()
#         for plane_index in range(0,activations[layer_index].shape[-1]):
#             plane = activations[layer_index][0,:, :, plane_index]
#             axs[plane_index].set_xticks([])
#             axs[plane_index].set_yticks([])
#             axs[plane_index].imshow(plane, cmap='gray_r')
#
#     fig.savefig(os.path.join(figure_path,figure_name+'.'+figure_format), format=figure_format)
#     if onscreen:
#         plt.show()
#
#     plt.close(fig)
#
#
# def display_weights_column(weights, layer_names,figure_path,figure_name,figure_format,onscreen=True):
#     n_layers_with_weights = 0
#     for layer_index in range(0, len(weights)):
#         layer_weights = weights[layer_index]
#         if len(layer_weights) > 0:
#             n_layers_with_weights += 1
#
#     fig = plt.figure(figsize=(30, 15), frameon=False)
#
#     plt.subplots_adjust(wspace=0.1, hspace=0.1, top=1.0, bottom=0.0, left=0.0, right=1.0)
#     subfigs = fig.subfigures(1, n_layers_with_weights)
#     # if there is only one layer with weights, subfigures() returns a single SubFigure
#     if n_layers_with_weights == 1:
#         subfigs = [subfigs]
#     layer_index_with_weights = 0
#     print("Number of layers: "+str(len(weights)))
#     for layer_index in range(0, len(weights)):
#         layer_weights = weights[layer_index]
#
#         print("layer:" +str(layer_index))
#         # only weights (0) no biases (1)
#         if len(layer_weights) > 0 and len(layer_weights[0].shape) > 1:
#             print(" weights shape ", layer_weights[0].shape)
#
#             #subfigs[layer_index_with_weights].suptitle(layer_names[layer_index])
#             #subfigs[layer_index_with_weights].tight_layout()
#             # squeeze=False squeezing at all is done: the returned Axes object is always a 2D array containing
#             axs = subfigs[layer_index_with_weights].subplots(layer_weights[0].shape[-2], layer_weights[0].shape[-1], squeeze=False)#, sharex=True, sharey=True)
#             subfigs[layer_index_with_weights].subplots_adjust(wspace=0.1, hspace=0.1, top=1.0, bottom=0.0, left=0.0, right=1.0)
#             print(axs.shape)
#             for i in range(0, layer_weights[0].shape[-2]):
#                 for j in range(0, layer_weights[0].shape[-1]):
#                     w = layer_weights[0]
#                     axs[i,j].imshow(w[:,:,i,j], cmap='gray_r', interpolation='nearest')
#                     axs[i,j].axis("off")
#
#             layer_index_with_weights += 1
#     fig.savefig(os.path.join(figure_path, figure_name + '.' + figure_format),
#                 format=figure_format)
#
#     if onscreen:
#         plt.show()
#
#     plt.close(fig)
#
#
# def display_loss_function(history, figure_path, figure_name, figure_format, onscreen=True):
#     loss = history.history['loss']
#     val_loss = history.history['val_loss']
#     epochs = range(1, len(loss) + 1)
#
#     fig = plt.figure()
#     plt.plot(epochs, loss, color='red', label='Training loss')
#     plt.plot(epochs, val_loss, color='green', label='Validation loss')
#     plt.title('Training loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#
#     fig.savefig(os.path.join(figure_path, figure_name + '.' + figure_format),
#                 format=figure_format)
#
#     if onscreen:
#         plt.show()
#
#     plt.close(fig)
#
#
# # loading the dataset
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
#
# # building the input vector from the 28x28 pixels
# X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
# X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
#
# print("X_train shape", X_train.shape)
# print("y_train shape", y_train.shape)
# print("X_test shape", X_test.shape)
# print("y_test shape", y_test.shape)
#
# # normalizing the data
# X_train /= 255
# X_test /= 255
#
# # one-hot encoding using keras' numpy-related utilities
# n_classes = 10
# print("Shape before one-hot encoding: ", y_train.shape)
# Y_train = to_categorical(y_train, n_classes)
# Y_test = to_categorical(y_test, n_classes)
# print("Shape after one-hot encoding: ", Y_train.shape)
#
# # TASK 1: topology parameters
#
# n_cnn_layers = 3
#
# n_cnn1planes = 10
# n_cnn2planes = 20
# n_cnn3planes = 30
#
#
# n_cnn1kernel = 3
# n_poolsize = 1
#
#
# # Stride defines the step size at which the filter moves across the input during convolution.
# # A larger stride results in a reduction of the spatial dimensions of the output feature map.
# # Stride can be adjusted to control the level of downsampling in the network.
# # Stride is a critical parameter for controlling the spatial resolution of the feature maps and influencing the receptive field of the network.
# n_strides = 1
# n_dense = 100
# dropout = 0.3
#
# # CHANGE FROM 1 TO MORE THAN 5
#
# n_epochs=4
#
#
# # für task 1, auskomenntieren, wenn du task 1 testen willst
# # model_name = (
# #     'CNN_T1_'
# #     + f'layers{n_cnn_layers}'
# #     + f'p1{n_cnn1planes}p2{n_cnn2planes}p3{n_cnn3planes}'
# #     + '_KERNEL' + str(n_cnn1kernel)
# #     + '_Epochs' + str(n_epochs)
# # )
#
#
#
# # ============================================================
# # TASK 4: OPTIMIZER – SGD WITH MOMENTUM
# # ============================================================
#
# # ----------------------------
# # EINSTELLUNG DER TRAININGSPARAMETER
# # ----------------------------
# # Legt Lernrate für diesen Trainingslauf fest:
# learning_rate = 0.005     # Bestimmt, wie stark Gewichte pro Schritt angepasst werden (wird in mehreren Runs geändert).
#
# # Legt Momentum-Wert fest:
# momentum = 0.9            # Momentum hilft, schneller in richtige Richtung zu lernen (fixer Wert laut Angabe)
#
#
# # ----------------------------
# # OUTPUT-ORDNER
# # ----------------------------
# figure_path = './task4/'   # Definiert Ordner, in dem alle Ergebnisgrafiken von Task 4 gespeichert werden:
#
#
# # Erstellt Ordner, falls er noch nicht existiert
# os.makedirs(figure_path, exist_ok=True)   # Verhindert Fehler, wenn Ordner bereits vorhanden ist.
#
# # Definiert Ausgabeformat von Grafik:
# figure_format = 'png'             # Grafiken werden als PNG-Dateien gespeichert
#
#
# # ----------------------------
# # MODELLNAME ZUR IDENTIFIKATION DES EXPERIMENTS
# # ----------------------------
# # Erstellt einen eindeutigen Modellnamen, um diesen Trainingslauf klar zu erkennen
# model_name = (
#     'CNN_T4_Momentum_'             # Kennzeichnet Task 4 + Verwendung von Momentum
#     + f'LR_{learning_rate}_'       # Speichert verwendete Lernrate im Namen
#     + f'M_{momentum}_'             # Speichert Momentum-Wert im Namen
#     + f'layers{n_cnn_layers}_'     # Gibt an, wie viele Convolution-Layer verwendet werden.
#     + f'p1_{n_cnn1planes}p2{n_cnn2planes}p3{n_cnn3planes}_'
#                                    # Speichert Anzahl der Filter pro Convolution-Layer.
#     + 'KERNEL' + str(n_cnn1kernel) # Gibt Kernelgröße der Convolution-Filter an.
#     + '_Epochs' + str(n_epochs)    # Speichert Anzahl der Trainings-Epochen
# )   # Ergebnis: Jede Grafik ist eindeutig einem bestimmten Experiment zugeordnet
#
#
#
# # ----------------------------
# # NEUES MODELL (frische Gewichte)
# # ----------------------------
# # Erstellt ein neues leeers neuronales Netz -> bekommt Schichten Schritt für Schritt
# model = Sequential()     # Sequential: Schichten werden von oben nach unten hinzugefügt
#
#
# # Fügt erste Bild-Erkennungs-Schicht hinzu.
# # Modell lernt hier einfache Merkmale wie Kanten und Linien zu erkennen.
# if n_cnn_layers >= 1:
#     model.add(Conv2D(
#         n_cnn1planes,                    # Wie viele Bildmerkmale gleichzeitig erkannt werden
#         (n_cnn1kernel, n_cnn1kernel),    # Größe des Suchfensters im Bild (z.B. 3x3 Pixel).
#         activation='relu',               # Negative Werte werden ignoriert, positive weitergegeben.
#         input_shape=(28, 28, 1)           # Eingabebild ist 28x28 Pixel + schwarz-weiß
#     ))
#     model.add(MaxPool2D())               # Verkleinert Bild + behält nur wichtigsten Merkmale
#
#
# # Fügt zweite Bild-Erkennungs-Schicht hinzu, falls aktiviert:
# # Modell erkennt hier komplexere Formen als in der ersten Schicht
# if n_cnn_layers >= 2:
#     model.add(Conv2D(
#         n_cnn2planes,                    # Erkennt mehr und komplexere Bildmerkmale
#         (n_cnn1kernel, n_cnn1kernel),    # Gleiche Fenstergröße wie zuvor
#         activation='relu'                # hilft dem Netz schneller zu lernen
#     ))
#     model.add(MaxPool2D())
#
#
# # Fügt dritte Bild-Erkennungs-Schicht hinzu, falls aktiviert:
# #  Modell erkennt hier komplexere Formen als in der ersten schicht.
# if n_cnn_layers >= 3:
#     model.add(Conv2D(
#         n_cnn3planes,                    # Erkennt sehr komplexe Formen
#         (n_cnn1kernel, n_cnn1kernel),    # Fenstergröße bleibt gleich
#         activation='relu'                # aktiviert nur sinnvolle Werte
#     ))
#     model.add(MaxPool2D())
#
#
# # Wandelt Bilddaten in eine einfache Zahlenliste um:
# model.add(Flatten())    # damit kann Netz Bildinfos weiterverarbeiten
#
# # Fügt Rechen-Schicht hinzu, die alle Bildmerkmale kombiniert:
# model.add(Dense(
#     n_dense,            # Anzahl der Recheneinheiten in dieser Schicht
#     activation='relu'   # erlaubt dem Netz, Zusammenhänge zu lernen
# ))
#
# # Fügt letzte Schicht hinzu, die die Zahl vorhersagt:
# model.add(Dense(
#     n_classes,          #  Ausgabe für jede Ziffer von 0 bis 9
#     activation='softmax' # Gibt an, welche Zahl am wahrscheinlichsten ist
# ))
#
#
#
# # ----------------------------
# # OPTIMIZER: SGD + MOMENTUM
# # ----------------------------
#
# # Erstellt Optimizer für das Training des Modells:
# optimizer = SGD(
#     learning_rate=learning_rate,   # gibt an, wie groß Lernschritte sind
#     momentum=momentum              # Momentum sorgt für gleichmäßigeres +schnelleres Lernen.
# )
#
#
# # Bereitet Modell für das Training vor:
# model.compile(
#     loss='categorical_crossentropy',  # Berechnet den Fehler zw Vorhersage + richtiger Klasse
#     metrics=['accuracy'],             # zeigt an, wie viele Bilder richtig erkannt wurden
#     optimizer=optimizer               # Verwendet den zuvor definierten SGD-Optimizer mit Momentum
# )
#
#
#
# # ----------------------------
# # TRAINING DES MODELLS
# # ----------------------------
#
# # Startet Training des neuronalen Netzes:
# history = model.fit(
#     X_train,               # trainingsbilder, aus denen das Modell lernt
#     Y_train,                # Richtige Klassen der Trainingsbilder (One-Hot-kodiert)
#     validation_split=0.1,   # 10 % der Daten werden zur Kontrolle während des Trainings verwendet.
#     batch_size=128,         # Anzahl der Bilder, die gleichzeitig verarbeitet werden
#     epochs=n_epochs,        # Gibt an, wie oft alle Trainingsdaten durchlaufen werden
#     verbose=1               # zeigt den Trainingsfortschritt nach jeder Epoche an
# )   # history speichert Loss-Werte für Training und Validierung
#
#
# # ----------------------------
# # AUSWERTUNG: LOSS-GRAFIK
# # ----------------------------
#
# # Erstellt eine Grafik zur Auswertung des Trainings.
# display_loss_function(
#     history,                     # enthält gespeicherten Loss-Werte aus dem Training
#     figure_path,                 # Ordner, in dem Grafik gespeichert wird
#     model_name + '_loss',         # Name der Grafikdatei
#     figure_format,               # definiert Ausgabeformat (PNG)
#     onscreen=True                # zeigt Grafik extra direkt am Bildschirm an
# )
#
# #---------------------------------------------TASK 4 END ------------------------------------------------------------------------
#
#


#
# # ---------------------------------------------TASK 5+6 START ------------------------------------------------------------------------
# import os
#
# # ===== Matplotlib Backend (Fenster + PNG, kein Tk) =====
# import matplotlib
# matplotlib.use("Agg")   # MUSS vor pyplot stehen
#
# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec
#
# # === Keras / TensorFlow ===
# from keras.datasets import mnist
# from keras.models import Model, Sequential
# from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, BatchNormalization
# from keras.optimizers.schedules import ExponentialDecay
# from keras import callbacks
# from tensorflow.keras.optimizers import SGD, Adam
# from keras.utils import to_categorical
#
# # === Evaluation ===
# from sklearn.metrics import ConfusionMatrixDisplay
# from sklearn.metrics import classification_report, confusion_matrix
#
# # === Utils ===
# from datetime import datetime
# import numpy as np
#
#
#
#
# def display_classification_report(classification_report, figure_path, figure_name, onscreen=True):
#     f = open(os.path.join(figure_path, figure_name + '.txt'), 'w', encoding='utf-8')
#     f.write(classification_report)
#     f.close()
#
#     if onscreen:
#        print(classification_report)
#
# def display_confusion_matrix(confusion_matrix, labels, figure_path, figure_name, figure_format, onscreen=True):
#     fig, ax = plt.subplots()
#
#     disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=labels)
#     disp.plot(cmap=plt.cm.Greys, ax=ax)
#
#     fig.savefig(os.path.join(figure_path, figure_name + '.' + figure_format),
#                 format=figure_format)
#
#     if onscreen:
#         plt.show()
#
#     plt.close(fig)
#
#
# def display_activations(input, label, activations, layer_names, figure_path, figure_name, figure_format, onscreen=True):
#     fig = plt.figure(layout='constrained', figsize=(10, 8))
#     subfigs = fig.subfigures(1, len(activations)+1) # layers + input , figsize=(row_size*2.5,len(model.layers)*1.5))
#
#     subfigs[0].subplots(1, 1)
#     subfigs[0].suptitle('Label: {}'.format(label))
#     axs = subfigs[0].get_axes()
#     axs[0].set_xticks([])
#     axs[0].set_yticks([])
#     axs[0].imshow(input, cmap='gray_r')
#
#     for layer_index in range(0,len(activations)):
#         print("layer:" +str(layer_index))
#         print(activations[layer_index].shape[-1])
#         subfigs[layer_index+1].suptitle(layer_names[layer_index])
#         subfigs[layer_index+1].subplots(activations[layer_index].shape[-1],1)
#
#     for layer_index in range(0,len(activations)):
#         print(activations[layer_index].shape)
#         #range(0,activations.shape[-1]):
#         axs = subfigs[layer_index+1].get_axes()
#         for plane_index in range(0,activations[layer_index].shape[-1]):
#             plane = activations[layer_index][0,:, :, plane_index]
#             axs[plane_index].set_xticks([])
#             axs[plane_index].set_yticks([])
#             axs[plane_index].imshow(plane, cmap='gray_r')
#
#     fig.savefig(os.path.join(figure_path,figure_name+'.'+figure_format), format=figure_format)
#     if onscreen:
#         plt.show()
#
#     plt.close(fig)
#
#
# def display_weights_column(weights, layer_names,figure_path,figure_name,figure_format,onscreen=True):
#     n_layers_with_weights = 0
#     for layer_index in range(0, len(weights)):
#         layer_weights = weights[layer_index]
#         if len(layer_weights) > 0:
#             n_layers_with_weights += 1
#
#     fig = plt.figure(figsize=(30, 15), frameon=False)
#
#     plt.subplots_adjust(wspace=0.1, hspace=0.1, top=1.0, bottom=0.0, left=0.0, right=1.0)
#     subfigs = fig.subfigures(1, n_layers_with_weights)
#     # if there is only one layer with weights, subfigures() returns a single SubFigure
#     if n_layers_with_weights == 1:
#         subfigs = [subfigs]
#     layer_index_with_weights = 0
#     print("Number of layers: "+str(len(weights)))
#     for layer_index in range(0, len(weights)):
#         layer_weights = weights[layer_index]
#
#         print("layer:" +str(layer_index))
#         # only weights (0) no biases (1)
#         if len(layer_weights) > 0 and len(layer_weights[0].shape) > 1:
#             print(" weights shape ", layer_weights[0].shape)
#
#             #subfigs[layer_index_with_weights].suptitle(layer_names[layer_index])
#             #subfigs[layer_index_with_weights].tight_layout()
#             # squeeze=False squeezing at all is done: the returned Axes object is always a 2D array containing
#             axs = subfigs[layer_index_with_weights].subplots(layer_weights[0].shape[-2], layer_weights[0].shape[-1], squeeze=False)#, sharex=True, sharey=True)
#             subfigs[layer_index_with_weights].subplots_adjust(wspace=0.1, hspace=0.1, top=1.0, bottom=0.0, left=0.0, right=1.0)
#             print(axs.shape)
#             for i in range(0, layer_weights[0].shape[-2]):
#                 for j in range(0, layer_weights[0].shape[-1]):
#                     w = layer_weights[0]
#                     axs[i,j].imshow(w[:,:,i,j], cmap='gray_r', interpolation='nearest')
#                     axs[i,j].axis("off")
#
#             layer_index_with_weights += 1
#     fig.savefig(os.path.join(figure_path, figure_name + '.' + figure_format),
#                 format=figure_format)
#
#     if onscreen:
#         plt.show()
#
#     plt.close(fig)
#
#
# def display_loss_function(history, figure_path, figure_name, figure_format, onscreen=True):
#     loss = history.history['loss']
#     val_loss = history.history['val_loss']
#     epochs = range(1, len(loss) + 1)
#
#     fig = plt.figure()
#     plt.plot(epochs, loss, color='red', label='Training loss')
#     plt.plot(epochs, val_loss, color='green', label='Validation loss')
#     plt.title('Training loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#
#     fig.savefig(os.path.join(figure_path, figure_name + '.' + figure_format),
#                 format=figure_format)
#
#     if onscreen:
#         plt.show()
#
#     plt.close(fig)
#
#
# # loading the dataset
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
#
# # building the input vector from the 28x28 pixels
# X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
# X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
#
# print("X_train shape", X_train.shape)
# print("y_train shape", y_train.shape)
# print("X_test shape", X_test.shape)
# print("y_test shape", y_test.shape)
#
# # normalizing the data
# X_train /= 255
# X_test /= 255
#
# # one-hot encoding using keras' numpy-related utilities
# n_classes = 10
# print("Shape before one-hot encoding: ", y_train.shape)
# Y_train = to_categorical(y_train, n_classes)
# Y_test = to_categorical(y_test, n_classes)
# print("Shape after one-hot encoding: ", Y_train.shape)
#
# # TASK 1: topology parameters
#
# n_cnn_layers = 3
#
# n_cnn1planes = 10
# n_cnn2planes = 20
# n_cnn3planes = 30
#
#
# n_cnn1kernel = 3
# n_poolsize = 1
#
#
# # Stride defines the step size at which the filter moves across the input during convolution.
# # A larger stride results in a reduction of the spatial dimensions of the output feature map.
# # Stride can be adjusted to control the level of downsampling in the network.
# # Stride is a critical parameter for controlling the spatial resolution of the feature maps and influencing the receptive field of the network.
# n_strides = 1
# n_dense = 100
# dropout = 0.3
#
# # CHANGE FROM 1 TO MORE THAN 5
#
# n_epochs=2
#
#
# # für task 1, auskomenntieren, wenn du task 1 testen willst
# # model_name = (
# #     'CNN_T1_'
# #     + f'layers{n_cnn_layers}'
# #     + f'p1{n_cnn1planes}p2{n_cnn2planes}p3{n_cnn3planes}'
# #     + '_KERNEL' + str(n_cnn1kernel)
# #     + '_Epochs' + str(n_epochs)
# # )
#
# # ============================================================
# # TASK 2: LEARNING RATE
# # ============================================================
# # Task 2.1: Learning rate controls the step size of weight updates
# # Task 2.2: Use SGD as optimizer
# # Task 2.3: Learning rate will be varied between [0.001, 0.01]
# # Task 2.4: model_name includes learning rate for identification
#
# learning_rate = 0.01   # <- diesen Wert änderst du für die 4 Runs, 0.001, 0.003, 0.005, 0.01
#
# model_name = (
#     'CNN_T2_LR_' + str(learning_rate) + '_'
#     + f'layers{n_cnn_layers}'
#     + f'p1{n_cnn1planes}p2{n_cnn2planes}p3{n_cnn3planes}'
#     + '_KERNEL' + str(n_cnn1kernel)
#     + '_Epochs' + str(n_epochs)
# )
#
# optimizer = SGD(learning_rate=learning_rate)
#
#
#
#
#
# # Now every run will produce CNN_T1_layers2_p1_10_p2_20_p3_30_..._loss.png, so you know which is which.
#
# #model_name = 'CNN_Handwritten_OCR_CNN'+str(n_cnn1planes)+'_KERNEL'+str(n_cnn1kernel)+'_Epochs' + str(n_epochs)
# #figure_format='svg'
# figure_format='png'
# figure_path='./'
# log_path='./log'
#
# # building a linear stack of layers with the sequential model
# model = Sequential()
# # ----- first conv + pooling block -----
# if n_cnn_layers >= 1:
#     cnn1 = Conv2D(
#         n_cnn1planes,
#         kernel_size=(n_cnn1kernel, n_cnn1kernel),
#         strides=(n_strides, n_strides),
#         padding='valid',
#         activation='relu',
#         input_shape=(28, 28, 1)
#     )
#     model.add(cnn1)
#     model.add(MaxPool2D(pool_size=(n_poolsize, n_poolsize)))
#     # model.add(Dropout(dropout))
#
# # ----- second conv + pooling block -----
# if n_cnn_layers >= 2:
#     cnn2 = Conv2D(
#         n_cnn2planes,
#         kernel_size=(n_cnn1kernel, n_cnn1kernel),
#         strides=(n_strides, n_strides),
#         padding='valid',
#         activation='relu'
#     )
#     model.add(cnn2)
#     model.add(MaxPool2D(pool_size=(n_poolsize, n_poolsize)))
#     # model.add(Dropout(dropout))
#
# # ----- third conv + pooling block -----
# if n_cnn_layers >= 3:
#     cnn3 = Conv2D(
#         n_cnn3planes,
#         kernel_size=(n_cnn1kernel, n_cnn1kernel),
#         strides=(n_strides, n_strides),
#         padding='valid',
#         activation='relu'
#     )
#     model.add(cnn3)
#     model.add(MaxPool2D(pool_size=(n_poolsize, n_poolsize)))
#     # model.add(Dropout(dropout))
# # flatten output of conv
# model.add(Flatten())
#
#
# model.add(Dropout(dropout))
#
# # hidden layer
# model.add(Dense(n_dense, activation='relu'))
# # output layer
# model.add(Dense(n_classes, activation='softmax'))
#
# # compiling the sequential model
#
# model_name += 'Optimzer' + 'SGD'
#
# # vary the constant learning rate
# #task 1
# #model_name += 'LearningRate' + 'Constant'
# #learning_rate = 0.001
#
#
# # OR use a learning rate scheduler that adapts the learning rate over the epochs of the training process
# # https://keras.io/2.15/api/optimizers/learning_rate_schedules/
#
# #model_name += 'LearningRate' + 'ExponentialDecay'
# #learning_rate = ExponentialDecay(initial_learning_rate=1e-2, decay_steps=n_epochs, decay_rate=0.9)
#
# #learning_rate=0.01
# #task 1
# #momentum = 0.9
# #optimizer=SGD(learning_rate = learning_rate, momentum = momentum)
#
# #optimizer=Adam(learning_rate = learning_rate)
#
# # vary the constant learning rate
# #learning_rate = 0.01
# #optimizer=SGD(learning_rate=learning_rate)
#
# #task 1
# #model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)
#
# #task 2
# model.compile(
#     loss='categorical_crossentropy',
#     metrics=['accuracy'],
#     optimizer=optimizer
# )
#
# layer_names = [layer.name for layer in model.layers[:8]]
#
# weights = [layer.get_weights() for layer in model.layers[:4]]
# figure_name=model_name + '_initial_weights'
# display_weights_column(weights, layer_names, './', figure_name, figure_format, False )
#
# # training the model for n_epochs, use 10% of the training data as validation data
# history = model.fit(X_train, Y_train, validation_split = 0.1, batch_size=128, epochs=n_epochs )
#
# figure_name=model_name + '_loss'
# display_loss_function(history,'./',figure_name,figure_format)
#
#
# weights = [layer.get_weights() for layer in model.layers[:4]]
# figure_name=model_name + '_weights'
# display_weights_column(weights, layer_names, './', figure_name, figure_format, False )
#
# X_test_images = X_test[:2]
# for i in range(X_test_images.shape[0]):
#     Y_test_pred = model.predict(np.expand_dims(X_test[i], axis=0))
#
#     activation_model = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])
#     activations = activation_model.predict(np.expand_dims(X_test[i], axis=0))
#
#     figure_name=model_name + 'activations' + 'test_image_' + str(i)
#     display_activations(X_test[i],np.argmax(Y_test[i]),activations[:4], layer_names[:4], figure_path, figure_name, figure_format)
#
# y_test_pred = model.predict(X_test)
# cm = confusion_matrix(y_test, np.argmax(y_test_pred,axis=1), labels=range(0,n_classes))
#
# figure_name=model_name + '_confusion_matrix'
# display_confusion_matrix(cm, range(0,n_classes), figure_path,figure_name,figure_format)
#
# figure_name=model_name + '_classification_report'
# display_classification_report(classification_report(y_test, np.argmax(y_test_pred,axis=1), target_names=[str(c) for c in range(0,n_classes)], digits=4), figure_path, figure_name)
#
# figure_name=model_name + '_model_summary'
# stringlist = []
# model.summary(print_fn=lambda x: stringlist.append(x))
# model_summary = "\n".join(stringlist)
# display_classification_report(model_summary, figure_path, figure_name)
#
#
#
#
#
# # ============================================================
# # TASK 3: LEARNING RATE SCHEDULES (SGD + ExponentialDecay)
# # ============================================================
#
# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
# from keras.optimizers import SGD
# from keras.optimizers.schedules import ExponentialDecay
# import os
#
# # --- eigener Output-Ordner nur für Task 3 ---
# figure_path = './task3/'
# os.makedirs(figure_path, exist_ok=True)
# figure_format = 'png'
#
# # ------------------------------------------------------------
# # MANUELL: pro Run ändern (mind. 4 Werte insgesamt testen)
# # ------------------------------------------------------------
# initial_lr = 0.1   # 0.001 | 0.01 | 0.05 | 0.1
#
# batch_size = 128
#
# # IMPORTANT: decay_steps zählt "optimizer steps" (Batches), nicht Epochen!
# steps_per_epoch = X_train.shape[0] // batch_size
#
# learning_rate = ExponentialDecay(
#     initial_learning_rate=initial_lr,
#     decay_steps=steps_per_epoch,   # => ungefähr pro Epoche decay
#     decay_rate=0.9
# )
#
# # ------------------------------------------------------------
# # NEUES MODELL (frische Gewichte für fairen Vergleich!)
# # (Architektur ident zu Task 2)
# # ------------------------------------------------------------
# model = Sequential()
#
# if n_cnn_layers >= 1:
#     model.add(Conv2D(n_cnn1planes, (n_cnn1kernel, n_cnn1kernel),
#                      activation='relu', input_shape=(28, 28, 1)))
#     model.add(MaxPool2D())
#
# if n_cnn_layers >= 2:
#     model.add(Conv2D(n_cnn2planes, (n_cnn1kernel, n_cnn1kernel),
#                      activation='relu'))
#     model.add(MaxPool2D())
#
# if n_cnn_layers >= 3:
#     model.add(Conv2D(n_cnn3planes, (n_cnn1kernel, n_cnn1kernel),
#                      activation='relu'))
#     model.add(MaxPool2D())
#
# model.add(Flatten())
# model.add(Dense(n_dense, activation='relu'))
# model.add(Dense(n_classes, activation='softmax'))
#
# # ------------------------------------------------------------
# # Compile + Train
# # ------------------------------------------------------------
# model_name = (
#     'CNN_T3_ExpDecay_LR_' + str(initial_lr) + '_'
#     + f'layers{n_cnn_layers}'
#     + f'p1{n_cnn1planes}p2{n_cnn2planes}p3{n_cnn3planes}'
#     + '_KERNEL' + str(n_cnn1kernel)
#     + '_Epochs' + str(n_epochs)
# )
#
# optimizer = SGD(learning_rate=learning_rate)
#
# model.compile(
#     loss='categorical_crossentropy',
#     metrics=['accuracy'],
#     optimizer=optimizer
# )
#
# history = model.fit(
#     X_train,
#     Y_train,
#     validation_split=0.1,
#     batch_size=batch_size,
#     epochs=n_epochs,
#     verbose=1
# )
#
# # ------------------------------------------------------------
# # REQUIRED: analyse loss (train + validation)
# # ------------------------------------------------------------
# display_loss_function(
#     history=history,
#     figure_path=figure_path,
#     figure_name=model_name + '_loss',
#     figure_format=figure_format,
#     onscreen=True
# )
#
# # ============================================================
# # TASK 4: OPTIMIZER – SGD WITH MOMENTUM
# # ============================================================
#
# # ----------------------------
# # MANUELLE PARAMETER
# # ----------------------------
# learning_rate = 0.005     # später ändern: 0.001 | 0.003 | 0.005 | 0.01
# momentum = 0.9           # fixer Wert laut Angabe
#
# # ----------------------------
# # OUTPUT-ORDNER
# # ----------------------------
# figure_path = './task4/'
# os.makedirs(figure_path, exist_ok=True)
# figure_format = 'png'
#
# # ----------------------------
# # MODELLNAME (wie Task 2)
# # ----------------------------
# model_name = (
#     'CNN_T4_Momentum_'
#     + f'LR_{learning_rate}_'
#     + f'M_{momentum}_'
#     + f'layers{n_cnn_layers}_'
#     + f'p1_{n_cnn1planes}p2{n_cnn2planes}p3{n_cnn3planes}_'
#     + 'KERNEL' + str(n_cnn1kernel)
#     + '_Epochs' + str(n_epochs)
# )
#
# # ----------------------------
# # NEUES MODELL (frische Gewichte!)
# # ----------------------------
# model = Sequential()
#
# if n_cnn_layers >= 1:
#     model.add(Conv2D(n_cnn1planes, (n_cnn1kernel, n_cnn1kernel),
#                      activation='relu', input_shape=(28, 28, 1)))
#     model.add(MaxPool2D())
#
# if n_cnn_layers >= 2:
#     model.add(Conv2D(n_cnn2planes, (n_cnn1kernel, n_cnn1kernel),
#                      activation='relu'))
#     model.add(MaxPool2D())
#
# if n_cnn_layers >= 3:
#     model.add(Conv2D(n_cnn3planes, (n_cnn1kernel, n_cnn1kernel),
#                      activation='relu'))
#     model.add(MaxPool2D())
#
# model.add(Flatten())
# model.add(Dense(n_dense, activation='relu'))
# model.add(Dense(n_classes, activation='softmax'))
#
# # ----------------------------
# # OPTIMIZER: SGD + MOMENTUM
# # ----------------------------
# optimizer = SGD(
#     learning_rate=learning_rate,
#     momentum=momentum
# )
#
# model.compile(
#     loss='categorical_crossentropy',
#     metrics=['accuracy'],
#     optimizer=optimizer
# )
#
# # ----------------------------
# # TRAINING
# # ----------------------------
# history = model.fit(
#     X_train,
#     Y_train,
#     validation_split=0.1,
#     batch_size=128,
#     epochs=n_epochs,
#     verbose=1
# )
#
# # ----------------------------
# # REQUIRED OUTPUT: LOSS PLOT
# # ----------------------------
# display_loss_function(
#     history,
#     figure_path,
#     model_name + '_loss',
#     figure_format,
#     onscreen=True
# )
#
# # ============================================================
# # TASK 5: DROPOUT LAYER (REGULARIZATION)
# # Dropout deaktiviert während des Trainings zufällig Neuronen,
# # um Overfitting zu reduzieren und die Generalisierung zu
# # verbessern
# # ============================================================
#
# # zu testende Dropout-Raten
# dropout_rates = [0.2, 0.3, 0.4, 0.5]
#
# # Ausgabeordner für Task-5-Ergebnisse
# figure_path = './task5/'
# os.makedirs(figure_path, exist_ok=True)
# figure_format = 'png'
#
# #Variablen zur Speicherung des besten Modells
# best_val_accuracy = 0.0
# best_model = None
# best_model_name = None
#
#
# # Für jede Dropout-Konfiguration wid ein neues Modell trainiert
# for dropout_rate in dropout_rates:
#
#     # Modellname enthält die Dropout-Rate zur eindeutigen Identifikation
#     model_name = (
#         'CNN_T5_Dropout_'
#         + f'DR_{dropout_rate}_'
#         + f'layers{n_cnn_layers}_'
#         + f'p1_{n_cnn1planes}p2{n_cnn2planes}p3{n_cnn3planes}_'
#         + 'KERNEL' + str(n_cnn1kernel)
#         + '_Epochs' + str(n_epochs)
#     )
#
#     #neues Modell mit frischen Gewichten
#     model = Sequential()
#
#     # ----- Conv Block 1 -----
#     if n_cnn_layers >= 1:
#         model.add(Conv2D(
#             n_cnn1planes,
#             (n_cnn1kernel, n_cnn1kernel),
#             activation='relu',
#             input_shape=(28, 28, 1)
#         ))
#         model.add(MaxPool2D())
#         model.add(Dropout(dropout_rate))
#
#     # ----- Conv Block 2 -----
#     if n_cnn_layers >= 2:
#         model.add(Conv2D(
#             n_cnn2planes,
#             (n_cnn1kernel, n_cnn1kernel),
#             activation='relu'
#         ))
#         model.add(MaxPool2D())
#         # Dropout nach dem Convulation-Block
#         model.add(Dropout(dropout_rate))
#
#     # ----- Conv Block 3 -----
#     if n_cnn_layers >= 3:
#         model.add(Conv2D(
#             n_cnn3planes,
#             (n_cnn1kernel, n_cnn1kernel),
#             activation='relu'
#         ))
#         model.add(MaxPool2D())
#         model.add(Dropout(dropout_rate))
#
#     # Übergang von Feature-Maps zu Vektorform
#     model.add(Flatten())
#
#     # Voll verbundene Klassifikationsschichten
#     model.add(Dense(n_dense, activation='relu'))
#     model.add(Dropout(dropout_rate))
#     model.add(Dense(n_classes, activation='softmax'))
#
#     # Kompilieren des Modells mit SGD-Optimizer
#     optimizer = SGD(learning_rate=learning_rate)
#
#     model.compile(
#         loss='categorical_crossentropy',
#         metrics=['accuracy'],
#         optimizer=optimizer
#     )
#
#     # Training des Modells mit Validierungsanteil
#     history = model.fit(
#         X_train,
#         Y_train,
#         validation_split=0.1,
#         batch_size=128,
#         epochs=n_epochs,
#         verbose=1
#     )
#
#     # ----- Loss plot -----
#     # Darstellung der Trainings- und Validierungs-Loss-Funktion
#     display_loss_function(
#         history,
#         figure_path,
#         model_name + '_loss',
#         figure_format,
#         onscreen=True
#     )
#
#     # ----- Track best model (by validation accuracy) -----
#     # Auswahl des besten Modells anhand der Validierungsgenauigkeit
#     max_val_acc = max(history.history['val_accuracy'])
#     if max_val_acc > best_val_accuracy:
#         best_val_accuracy = max_val_acc
#         best_model = model
#         best_model_name = model_name
#
#
# # ============================================================
# # TASK 6: FINAL ACCURACY EVALUATION
# # Die Evalution erfolgt auf bisher ungesehenen Testdaten.
# # ============================================================
#
# figure_path = './task6/'
# os.makedirs(figure_path, exist_ok=True)
#
# print("Best model selected:", best_model_name)
# print("Best validation accuracy:", best_val_accuracy)
#
# # Predict on test data
# y_test_pred = best_model.predict(X_test)
#
# # Confusion Matrix
# # zeigt korrekte und falsche Klassifikationen pro Klasee
# cm = confusion_matrix(
#     y_test,
#     np.argmax(y_test_pred, axis=1),
#     labels=range(0, n_classes)
# )
#
# figure_name = best_model_name + '_confusion_matrix'
# display_confusion_matrix(
#     cm,
#     range(0, n_classes),
#     figure_path,
#     figure_name,
#     figure_format,
#     onscreen=True
# )
#
# # Classification Report
# # enthält Precision, Recall, F1-Score und Gesamtgenauigkeit
# report = classification_report(
#     y_test,
#     np.argmax(y_test_pred, axis=1),
#     target_names=[str(c) for c in range(0, n_classes)],
#     digits=4
# )
#
# figure_name = best_model_name + '_classification_report'
# display_classification_report(
#     report,
#     figure_path,
#     figure_name,
#     onscreen=True
# )
# # ---------------------------------------------TASK 5+6 END ------------------------------------------------------------------------
