"""# For Production"""

class CaptionModel(tf.keras.models.Model):
  def __init__(self):
    super(CaptionModel,self).__init__()
    self.encoder = get_encoder()
    self.word_embed = L.Embedding(top_k, WORD_EMBED_SIZE)
    self.img_embed_to_bottleneck = L.Dense(IMG_EMBED_BOTTLENECK, 
                                      input_shape=(None, IMG_EMBED_SIZE), 
                                      activation='elu')
    self.img_embed_bottleneck_to_h0 = L.Dense(LSTM_UNITS,
                                         input_shape=(None, IMG_EMBED_BOTTLENECK),
                                         activation='elu')
    self.lstm = L.LSTM(LSTM_UNITS, return_sequences=True)
    self.flat_hidden_states = L.Reshape((-1,LSTM_UNITS))
    self.token_logits_bottleneck = L.Dense(LOGIT_BOTTLENECK, 
                                      input_shape=(None, LSTM_UNITS),
                                      activation="elu")
    self.token_logits = L.Dense(top_k,
                           input_shape=(None, LOGIT_BOTTLENECK))
  
  def call(self,images,y):
    x = self.encoder(images)
    x = self.img_embed_to_bottleneck(x)
    x = self.img_embed_bottleneck_to_h0(x)
    y = self.word_embed(y)
    out = self.lstm(y,initial_state=[x,x])
    out = self.flat_hidden_states(out)
    out = self.token_logits_bottleneck(out)
    out = self.token_logits(out)
    return out

  
caption_model = CaptionModel()
caption_model.compile(loss=loss_function,optimizer=optimizer)
image = tf.keras.applications.inception_v3.preprocess_input(np.asarray(Image.open("/content/sample_data/IMG_20230411_195227.jpg").resize((256,256))))
image = np.stack([image])
caption = np.stack([[3 for i in range(47)]])
a = caption_model(image,caption[:,1:])
caption_model.load_weights('finmodel.hdf5')

def preprocess(image_path):
  image = np.asarray(Image.open(image_path).resize((256,256)))
  image = tf.keras.applications.inception_v3.preprocess_input(image)
  return image

def evaluate(image):
  result = [tokenizer.word_index['<start>']]
  dec_input = tf.expand_dims(result, 0)
  images = np.stack([image])

  for i in range(captions_vector.shape[1]):
    out = caption_model(images,dec_input)
    pred_id = np.argmax(tf.nn.softmax(out,axis=2)[0,-1,:].numpy())
    if tokenizer.index_word[pred_id]!="<end>":
      result.append(pred_id)
    else:
      break
    dec_input = tf.expand_dims(result, 0)
  predicted = [tokenizer.index_word[i] for i in result]
  return " ".join(predicted[1:])
