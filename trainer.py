from __future__ import division, print_function

import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import resampy
import soundfile as sf
import tensorflow as tf
from tensorflow.keras import Model, layers, optimizers


import params as yamnet_params
import yamnet as yamnet_model
import features as features_lib
from random import shuffle

import librosa
import wave
import json
import glob

from tqdm import tqdm

import gc

intense = {"약함" : 0.,"보통":1.,"강함":2.}
emotion_enc = {"기쁨" : 0, "사랑스러움" : 1, "두려움" : 2, "화남" : 3, "슬픔" : 4, "놀라움" : 5, "없음" : 6}
emotion_dec = { 0 : "기쁨", 1 : "사랑스러움", 2 : "두려움", 3 : "화남", 4 : "슬픔", 5 : "놀라움", 6 : "없음"}

def load_split_data_from_json(root_path, sr = 16000):
    label_path = os.path.join(root_path, "label")
    data_path = os.path.join(root_path, "data")
    json_list = glob.glob(os.path.join(label_path,'*.json'))
    #shuffle(json_list)
    gc.disable()
    for i, path in enumerate(json_list):
        #gc.collect()
        with open(path, 'r') as f:
            data = json.load(f)
        
        #if i ^ ((1 << 10) - 1) == 0:
        #    gc.enable()
        #    gc.collect()
        #    gc.disable()
        wav_name = os.path.join(data_path, os.path.basename(path).split(".")[0] + ".wav")
        audio, _ = librosa.load(wav_name, sr=sr)
        audio = np.array(audio[:int(2 * 0.96 * sr)])
        # with wave.open(wav_name, 'rb') as wav_file:
        #     frame_rate = sr
        #     start_time = 0  # 시작 시간(초)
        #     duration = 2 * 0.96    # 읽을 시간 길이(초)
        #     num_frames = int(frame_rate * duration)

        #     frames = wav_file.readframes(num_frames)
        #     audio = np.frombuffer(frames, dtype=np.int16)

        text = data['text']
        emotion_category = float(data["emotion_category"])
        emotion_intense = float(data["emotion_intense"])
        
        yield audio, emotion_category, emotion_intense


def preprocess(audio_segment, emotion_category, emotion_intensity):
    labels = {
        'emotion_class_output': emotion_category,
        'emotion_intensity_output': emotion_intensity
    }
    return audio_segment, labels

def prediction_model(input_shape, num_classes):
    """Defines the prediction model for emotion classification and intensity regression."""
    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dropout(0.5)(x)
    emotion_class_output = layers.Dense(num_classes, activation='softmax', name='emotion_class_output')(x)
    emotion_intensity_output = layers.Dense(1, activation='linear', name='emotion_intensity_output')(x)
    model = Model(inputs=inputs,  outputs= [emotion_class_output, emotion_intensity_output], name='emotion_recognition_model')
    return model

class SaveModelAtEpochEnd(tf.keras.callbacks.Callback):
    def __init__(self, save_dir):
        super(SaveModelAtEpochEnd, self).__init__()
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def on_epoch_end(self, epoch, logs=None):
        epoch_num = epoch + 1
        save_path = os.path.join(self.save_dir, f'epoch_{epoch_num:02d}')
        os.makedirs(save_path, exist_ok=True)
        tf.saved_model.save(self.model, save_path)
        print(f'\nSaved model at epoch {epoch_num} to {save_path}')

if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    #print(gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    train_path = "C:/Users/jspark/Desktop/train"

    dataset = tf.data.Dataset.from_generator(
        lambda: load_split_data_from_json(train_path),
        output_signature=(
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
    )
    #dataset = dataset.cache()
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    #dataset = dataset.cache()

    dataset = dataset.shuffle(buffer_size=1000)

    #dataset = dataset.shuffle(buffer_size=5000)
    batch_size = 64
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=([None, ], {'emotion_class_output': [], 'emotion_intensity_output': []}),
        padding_values=(0.0, {'emotion_class_output': 0, 'emotion_intensity_output': 0.0})
    )
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    params = yamnet_params.Params()

    embedding_model = yamnet_model.yamnet_embedding_model(params, "my_yamnet.h5")

    for layer in embedding_model.layers:
        layer.trainable = True

    num_classes = len(emotion_enc)
    prediction_net = prediction_model(input_shape=(1024,), num_classes=num_classes)

    waveform_input = embedding_model.inputs[0]
    embeddings = embedding_model.outputs[0]
    #waveform_input = layers.Input(shape=(None,), dtype=tf.float32, name='waveform_input')
    #embeddings = embedding_model(waveform_input)
    predictions = prediction_net(embeddings)

    model = Model(inputs=waveform_input, outputs={"emotion_class_output": predictions[0],
                "emotion_intensity_output": predictions[1]}, name='emotion_recognition_model')
    

    epoch = 10

    lr_schedule = optimizers.schedules.CosineDecay(
        initial_learning_rate=0.001,
        decay_steps=100
    )

    optimizer = optimizers.Adam(lr_schedule)
    save_dir = f'results_pretrain_emb_{batch_size}'
    save_callback = SaveModelAtEpochEnd(save_dir=save_dir)
    model.compile(
        optimizer= optimizer,
        loss={
            'emotion_class_output': 'sparse_categorical_crossentropy',
            'emotion_intensity_output': 'mean_squared_error'
        },
        metrics={
            'emotion_class_output': 'accuracy',
            'emotion_intensity_output': 'mae'
        },
        #run_eagerly=True
    )

    model.fit(dataset, epochs=epoch, verbose=1,callbacks=[save_callback])
