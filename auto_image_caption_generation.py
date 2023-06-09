# -*- coding: utf-8 -*-
"""Auto_Image Caption Generation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VCkUv5NdkQbWrLMP2eySuFqMV6JeDKF6
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import collections
import random
import numpy as np
import os
import time
import json
from PIL import Image
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
from tqdm.notebook import tqdm

"""# New Section"""

# Download caption annotation files
annotation_folder = '/annotations/'
if not os.path.exists(os.path.abspath('.') + annotation_folder):
  annotation_zip = tf.keras.utils.get_file('captions.zip',
                                           cache_subdir=os.path.abspath('.'),
                                           origin='http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                           extract=True)
  annotation_file = os.path.dirname(annotation_zip)+'/annotations/captions_train2014.json'
  os.remove(annotation_zip)

# Download image files
image_folder = '/train2014/'
if not os.path.exists(os.path.abspath('.') + image_folder):
  image_zip = tf.keras.utils.get_file('train2014.zip',
                                      cache_subdir=os.path.abspath('.'),
                                      origin='http://images.cocodataset.org/zips/train2014.zip',
                                      extract=True)
  PATH = os.path.dirname(image_zip) + image_folder
  os.remove(image_zip)
else:
  PATH = os.path.abspath('.') + image_folder

annotation_file = '/content/annotations/captions_train2014.json'
PATH = '/content/train2014/'

with open(annotation_file,'r') as f:
  annotations = json.load(f)

image_path_to_caption = collections.defaultdict(list)
for val in annotations['annotations']:
  caption = f"<start> {val['caption']} <end>"
  image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (val['image_id'])
  image_path_to_caption[image_path].append(caption)

image_paths = list(image_path_to_caption.keys())
random.shuffle(image_paths)

train_image_paths = image_paths[:10000]
train_captions = []
img_name_vector = []

for image_path in tqdm(train_image_paths):
  if np.asarray(Image.open(image_path)).ndim==3:
    caption_list = image_path_to_caption[image_path]
    train_captions.extend(caption_list)
    img_name_vector.extend([image_path] * len(caption_list))

def get_encoder():
  model = tf.keras.applications.InceptionV3(include_top=False,input_shape=(256,256,3))
  model.trainable = False
  model = tf.keras.models.Model(model.inputs, L.GlobalAveragePooling2D()(model.output))

  return model

def apply_model(model,img_name_vector):
  images = []
  for path in img_name_vector:
    img = Image.open(path)
    images.append(tf.keras.applications.inception_v3.preprocess_input(np.asarray(img.resize((256,256)))))
  images = np.stack(images)
  embeddings = model.predict(images)
  return embeddings

top_k = 5000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                  oov_token="<unk>",
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(train_captions)
tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'

train_seqs = tokenizer.texts_to_sequences(train_captions)
captions_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

captions_vector.shape

class DataLoader:
  def __init__(self,img_name_vector,captions,embedding_model):
    self.files = img_name_vector
    self.captions = captions
    self.model = embedding_model

  def image_reader(self,files,captions):
    for i in range(len(files)):
      yield files[i],captions[i]

  def batch_generator(self,items,batch_size):
    a=[]
    i=0
    for item in items:
      a.append(item)
      i+=1

      if i%batch_size==0:
        yield a
        a=[]
    if len(a) is not 0:
      yield a
  
  def flow(self,batch_size):
    """
    flow from given directory in batches
    ==========================================
    batch_size: size of the batch
    """
    while True:
      for bat in self.batch_generator(self.image_reader(self.files,self.captions),batch_size):
        batch_images = []
        batch_labels = []
        for im,im_label in bat:
          batch_images.append(im)
          batch_labels.append(im_label)
        batch_labels =  np.stack(batch_labels,axis=0)
        batch_embeddings = np.stack(apply_model(self.model,batch_images),axis=0)
        yield (batch_embeddings,batch_labels[:,:-1]),batch_labels[:,1:]

K.clear_session()
embedding_model = get_encoder()
loader = DataLoader(img_name_vector,captions_vector,embedding_model)
data = loader.flow(batch_size = 32)

IMG_EMBED_SIZE = 2048
IMG_EMBED_BOTTLENECK = 160
WORD_EMBED_SIZE = 100
LSTM_UNITS = 300
LOGIT_BOTTLENECK = 120
top_k=5000

class Decoder(tf.keras.models.Model):
  def __init__(self):
    super(Decoder,self).__init__()
    self.img_embed_to_bottleneck = L.Dense(IMG_EMBED_BOTTLENECK, 
                                      input_shape=(None, IMG_EMBED_SIZE), 
                                      activation='elu')
    self.img_embed_bottleneck_to_h0 = L.Dense(LSTM_UNITS,
                                         input_shape=(None, IMG_EMBED_BOTTLENECK),
                                         activation='elu')

    self.lstm = L.LSTM(LSTM_UNITS, return_sequences=True)
    self.word_embed = L.Embedding(top_k, WORD_EMBED_SIZE)
    self.flat_hidden_states = L.Reshape((-1,LSTM_UNITS))
  
    self.token_logits_bottleneck = L.Dense(LOGIT_BOTTLENECK, 
                                      input_shape=(None, LSTM_UNITS),
                                      activation="elu")
    self.token_logits = L.Dense(top_k,
                           input_shape=(None, LOGIT_BOTTLENECK))
    
  def call(self,data):
    x, captions = data
    captions = self.word_embed(captions)

    x = self.img_embed_to_bottleneck(x)
    x = self.img_embed_bottleneck_to_h0(x)
    out = self.lstm(captions,initial_state=[x,x])
    out = self.flat_hidden_states(out)
    out = self.token_logits_bottleneck(out)
    out = self.token_logits(out)
    return out

loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,reduction='none')
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

def loss_function(real,pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_obj(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

decoder = Decoder()
decoder.compile(loss=loss_function,optimizer=optimizer)

ckpt = tf.keras.callbacks.ModelCheckpoint('model.hdf5',monitor='loss',mode='min',save_best_only=True,save_weights_only=True)

decoder.fit(data,
            epochs=5,
            steps_per_epoch=captions_vector.shape[0]//32,
            callbacks=[ckpt])

decoder.load_weights('model.hdf5')

class FinalModel(tf.keras.models.Model):
  def __init__(self):
    super(FinalModel,self).__init__()
    self.encoder = embedding_model
    self.word_embed = decoder.word_embed
    self.img_embed_to_bottleneck = decoder.img_embed_to_bottleneck
    self.img_embed_bottleneck_to_h0 = decoder.img_embed_bottleneck_to_h0
    self.lstm = decoder.lstm
    self.flat_hidden_states = decoder.flat_hidden_states
    self.token_logits_bottleneck = decoder.token_logits_bottleneck
    self.token_logits = decoder.token_logits
  
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

final_model = FinalModel()

final_model.save_weights('finmodel.hdf5')

def preprocess(image_path):
  image = np.asarray(Image.open(image_path).resize((256,256)))
  image = tf.keras.applications.inception_v3.preprocess_input(image)
  return image

def display(image_path):
  plt.figure(figsize=(10,10));
  plt.imshow(np.asarray(Image.open(image_path)));
  plt.title(evaluate(preprocess(image_path)));

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
image = tf.keras.applications.inception_v3.preprocess_input(np.asarray(Image.open("").resize((256,256))))
image = np.stack([image])
caption = np.stack([[3 for i in range(47)]])
a = caption_model(image,caption[:,1:])
caption_model.load_weights('finmodel.hdf5')

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

!wget "https://i.pinimg.com/474x/6f/ed/23/6fed238155f97758fcb791652c07175e--funny-cat-pictures-funny-pics.jpg"

display('6fed238155f97758fcb791652c07175e--funny-cat-pictures-funny-pics.jpg')

!wget "https://c1.peakpx.com/wallpaper/189/749/296/baseball-player-batter-game-ball-wallpaper-preview.jpg"

display("baseball-player-batter-game-ball-wallpaper-preview.jpg")

!wget "https://st.depositphotos.com/3332767/4586/i/950/depositphotos_45861295-stock-photo-man-riding-motorcycle-on-road.jpg"

display("depositphotos_45861295-stock-photo-man-riding-motorcycle-on-road.jpg")

with open('tokens.txt','w') as f:
  json.dump(tokenizer.index_word,f)

with open('words.txt','w') as f:
  json.dump(tokenizer.word_index,f)

