import keras.utils as image
import tensorflow as tf
import numpy as np


def load_tflite_model(model_path):
    # Load the TFLite model and allocate tensors.
    tflite_interpreter = tf.lite.Interpreter(model_path=model_path)
    tflite_interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()

    return tflite_interpreter, input_details, output_details


def load_labels(label_path):
    with open(label_path, 'r') as f:
        labels = list(map(str.strip, f.readlines()))
    return labels


def load_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img = tf.expand_dims(img_array, 0)  # Create a batch
    img = np.array(img, dtype=np.uint8)
    return img


def pred( img_path, model_path = 'model.tflite', label_path ='labels.txt'):
    tflite_interpreter, input_details, output_details = load_tflite_model(model_path)
    labels = load_labels(label_path)
    img = load_image(img_path)
    top_k_results = len(labels)  # number of labels or classes

    tflite_interpreter.set_tensor(input_details[0]['index'], img)

    # Run inference
    tflite_interpreter.invoke()

    # Get prediction results
    predictions = tflite_interpreter.get_tensor(output_details[0]['index'])[0]
    # print("Prediction results shape:", predictions)
    top_k_indices = np.argsort(predictions)[::-1][:top_k_results]

    pred_max = predictions[top_k_indices[0]] / 255.0
    lbl_max = labels[top_k_indices[0]]
    return lbl_max

print(pred( 'photo.jpg'))
