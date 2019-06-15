import os
import matplotlib.pyplot as plt
from keras.optimizers import SGD, Adam
from sklearn.metrics import confusion_matrix
from train_model.cnn import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from preprocess import constants

current_directory = os.path.dirname(__file__)
SPECTROGRAMS_DIRECTORY = constants.SPECTROGRAMS_DIRECTORY

LOAD_MODEL = True
SAVE_MODEL = False
MODEL_NAME = "my-model.h5"

TRAIN_SUBDIR = constants.TRAIN_SUBDIR
TEST_SUBDIR = constants.TEST_SUBDIR
VALIDATION_SUBDIR = constants.VALIDATION_SUBDIR

img_rows = 120
img_cols = 160
batch_size = 32
epochs = 30
channels = 3

nb_train_samples = 7686
nb_validation_samples = 369
nb_test_samples = 464

print("Loading the training dataset...")
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    SPECTROGRAMS_DIRECTORY + TRAIN_SUBDIR,
            shuffle=True,
            target_size=(img_cols, img_rows),
            batch_size=batch_size,
            class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
    SPECTROGRAMS_DIRECTORY + VALIDATION_SUBDIR,
                    shuffle=True,
                    target_size=(img_cols, img_rows),
                    batch_size=batch_size,
                    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    SPECTROGRAMS_DIRECTORY + TEST_SUBDIR,
                target_size=(img_cols, img_rows),
                batch_size=1,
                class_mode='binary',
                shuffle=False)

model = None
if LOAD_MODEL:
    model = load_model(MODEL_NAME)
    print("Loaded model from disk")
else:
    input_shape=(img_cols, img_rows, channels)
    model = Model.build(input_shape)
    opt = Adam(lr=0.0001)
    model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        verbose=2,
        class_weight=None)

    print("EVALUATE THE MODEL...")
    score = model.evaluate_generator(generator=validation_generator, steps=nb_validation_samples // batch_size)

    print("Accuracy = " + str(score[1]))
    print("Loss = " + str(score[0]))

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['acc', 'val_acc'], loc='upper right')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.show()

print("Model summary:")
print(model.summary())

print("MAKE PREDICTIONS...")
test_generator.reset()
pred = model.predict_generator(test_generator,
                               steps=nb_test_samples // 1,
                               verbose=2)


file_names=test_generator.filenames

ground_truth = test_generator.classes
label2index = test_generator.class_indices
print("Label to index")
print(label2index)

predicted_classes = []
for k in pred:
    if k < 0.5:
        predicted_classes.append(0)
    else:
        predicted_classes.append(1)


cm = confusion_matrix(ground_truth, predicted_classes)
print("Confusion matrix:")
print(cm)
if SAVE_MODEL:
    model.save(MODEL_NAME)


text_file = open("results-pred-155.txt", "w")
k=0
for i in predicted_classes:
    text_file.write("Filename: " + file_names[k])
    text_file.write("\n")
    text_file.write("          Predicted: " + str(i))
    text_file.write("\n")
    text_file.write("          Actual: " + str(ground_truth[k]))
    text_file.write("\n")
    k += 1
text_file.close()


