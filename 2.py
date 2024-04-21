import numpy as np
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image


model = ResNet50(weights='imagenet')


def classify_image(image_path, prompt=None):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    print("Predicted:", decode_predictions(preds, top=3)[0])

    if prompt:
        prompt_preds = model.predict(x, steps=1, verbose=0)
        print("Prompt Prediction:", decode_predictions(prompt_preds, top=3)[0])


test1 = 'pic/fb1.jpeg'
classify_image(test1)
