import tensorflow as tf
import pickle
import string
import numpy as np  







# defining a function which removes the punctuations from a string
def clean_text(s):
  out = s.translate(str.maketrans('', '', string.punctuation))
  return out


# function for prediction
def predict(test_review,test_aspect):

    # defining a label map
    label_map = {
                0:'Negative',
                1: 'Neutral',
                2:'Positive'
                }
    # preprocessing the test review
    cleaned_test_review=clean_text(test_review)
    cleaned_test_review=cleaned_test_review.lower()

    # preprocessing the test aspect
    cleaned_test_aspect=clean_text(test_aspect)
    cleaned_test_aspect=cleaned_test_aspect.lower()

    # loading the tokenzier
    with open('pb_format/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # converting the test reviews to review sequences
    test_review_sequence=tokenizer.texts_to_sequences([cleaned_test_review])
    # padding the review sequences
    test_reviews_sequence_padded=tf.keras.preprocessing.sequence.pad_sequences(test_review_sequence,maxlen=20)

    # converting the test aspect to aspect sequence
    test_aspect_sequence=tokenizer.texts_to_sequences([cleaned_test_aspect])
    # padding the review sequences
    test_aspect_sequence_padded=tf.keras.preprocessing.sequence.pad_sequences(test_aspect_sequence,maxlen=2)

    # Load the saved model
    model = tf.saved_model.load("pb_format")

    # Convert input data to tensors
    test_review_tensor = tf.constant(test_reviews_sequence_padded, dtype=tf.float64)
    test_aspect_tensor = tf.constant(test_aspect_sequence_padded, dtype=tf.float64)

    # Perform inference using the appropriate signature
    infer = model.signatures["serving_default"]
    output = infer(aspect_input_text=test_aspect_tensor, review_input_text=test_review_tensor)
    


    # Extract predictions from the output
    predictions = output["dense_56"].numpy()

    # Get the predicted label using argmax
    predicted_label = label_map[np.argmax(predictions)]

    

    return predicted_label























"""  # prediction
    test_review_input= tf.keras.Input(shape=(20,),dtype="float64")
    test_aspect_input=tf.keras.Input(shape=(2,),dtype="float64")

    from keras.models import load_model

    # Load the model
    model = load_model("h5_format/absa.h5")

    # Perform inference
    predictions = model.predict([test_reviews_sequence_padded, test_aspect_sequence_padded])
    prediction = label_map[np.argmax(predictions)]

    return prediction """