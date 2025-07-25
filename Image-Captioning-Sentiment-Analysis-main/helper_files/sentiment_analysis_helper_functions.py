import numpy as np
import tensorflow as tf
from tensorflow.keras import activations




def construct_encodings(X, tokenizer, max_len, truncation=True, padding=True):
  return tokenizer(X, max_length=max_len, truncation=truncation, padding = padding)

def construct_tfdataset(encodings, y=None):
    if y is not None:
        return tf.data.Dataset.from_tensor_slices((dict(encodings),y))
    else:
        return tf.data.Dataset.from_tensor_slices(dict(encodings))
    
def create_predictor(model, tokenizer, model_name, max_len):
  def predict_proba(text):
      x = [text]

      encodings = construct_encodings(x, tokenizer, max_len=max_len)
      tfdataset = construct_tfdataset(encodings)
      tfdataset = tfdataset.batch(1)

      preds = model.predict(tfdataset).logits
      preds = activations.softmax(tf.convert_to_tensor(preds)).numpy()
      return np.argmax(preds)

  return predict_proba