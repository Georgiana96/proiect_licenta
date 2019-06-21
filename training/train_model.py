import os
import matplotlib.pyplot as plt
from keras.optimizers import SGD, Adam
from sklearn.metrics import confusion_matrix
from training import cnn
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from preprocess import constants
from sklearn.metrics import classification_report

current_directory = os.path.dirname(__file__)
SPECTROGRAMS_DIRECTORY = constants.SPECTROGRAMS_DIRECTORY.split('/')[1] + '/'

LOAD_MODEL = True
SAVE_MODEL = False
MODEL_NAME = "my_model-154.h5"

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
nb_test_samples = 367

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
    model = cnn.Model.build(input_shape)
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


text_file = open("results-pred-154.txt", "w")
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

print(classification_report(ground_truth, predicted_classes))

#################################3
from keras.preprocessing import image
import numpy as np
import glob
from keras.models import Model
model_feat = Model(inputs=model.input,outputs=model.get_layer('flatten_2').output)

def get_features(img_path):
    img = image.load_img(img_path, target_size=(160, 120))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)
    flatten = model_feat.predict(x)
    return list(flatten[0])

def get_dataset(path, label):
    X = []
    y = []
    for relative_path in glob.glob(path + '*.png'):
        path = relative_path.split("\\")[0]
        name = relative_path.split("\\")[1]
        image_path = path + "/" + name
        X.append(get_features(image_path))
        y.append(label)
    return X,y


trainX_0, trainY_0 = get_dataset(current_directory + '/generated_spectrograms/train/alte_sentimente/',0)
testX_0, testY_0 = get_dataset(current_directory + '/generated_spectrograms/test/alte_sentimente/',0)

trainX_1, trainY_1 = get_dataset(current_directory + '/generated_spectrograms/train/furie/',1)
testX_1, testY_1 = get_dataset(current_directory + '/generated_spectrograms/test/furie/',1)

X = trainX_0 + testX_0 + trainX_1 + testX_1
y = trainY_0 + testY_0 + trainY_1 + testY_1

X_train = trainX_0 + trainX_1
y_train = trainY_0 + trainY_1

X_test = testX_0 + testX_1
y_test = testY_0 + testY_1
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

clf = SVC(kernel='poly', C=0.5)
print("Fitting...")
clf.fit(X_train, y_train)

print("Making predictions...")
predicted = clf.predict(X_test)

# get the accuracy
print (accuracy_score(y_test, predicted))

cm = confusion_matrix(y_test, predicted)
print("Confusion matrix:")
print(cm)

print(classification_report(y_test, predicted))
