import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import resampy
import soundfile as sf
import tensorflow as tf
from tensorflow.keras import Model, layers, optimizers

import librosa
import json
import glob

from tqdm import tqdm

def load_split_data_from_json(root_path, sr = 16000):
    label_path = os.path.join(root_path, "label")
    data_path = os.path.join(root_path, "data")
    json_list = glob.glob(os.path.join(label_path,'*.json'))
    for i, path in enumerate(json_list):
        with open(path, 'r') as f:
            data = json.load(f)
        wav_name = os.path.join(data_path, os.path.basename(path).split(".")[0] + ".wav")
        audio, _ = librosa.load(wav_name, sr=sr)
        #audio = np.array(audio)
        audio = np.array(audio[:int(2 * 0.96 * sr)])

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

if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    print(gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    model_path = "results_pretrain_emb_64/epoch_03"
    test_data_path = "C:/Users/jspark/Desktop/valid"

    test_data_path = os.path.join(test_data_path, "indoor")
    #test_data_path = os.path.join(test_data_path, "outdoor")


    model = tf.saved_model.load(model_path)
    loss_fn_class = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    loss_fn_intensity = tf.keras.losses.MeanSquaredError()
    mean_loss_class = tf.keras.metrics.Mean()
    mean_loss_intensity = tf.keras.metrics.Mean()
    
    accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    mae_metric = tf.keras.metrics.MeanAbsoluteError()

    dataset = tf.data.Dataset.from_generator(
        lambda: load_split_data_from_json(test_data_path),
        output_signature=(
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
    )

    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    cnt = 84160

    pbar =  tqdm(dataset, total=cnt)

    heat = np.zeros((7,7))

    for audio, labels in pbar:
        predictions = model(audio, training=False)

        emotion_class_preds = predictions['emotion_class_output']
        emotion_intensity_preds = predictions['emotion_intensity_output']
        
        emotion_class_true = labels['emotion_class_output']
        emotion_intensity_true = labels['emotion_intensity_output']
        
        heat[int(emotion_intensity_true)] += emotion_class_preds

        loss_class = loss_fn_class(emotion_class_true, emotion_class_preds)
        loss_intensity = loss_fn_intensity(emotion_intensity_true, emotion_intensity_preds)

        mean_loss_class.update_state(loss_class)
        mean_loss_intensity.update_state(loss_intensity)
        accuracy_metric.update_state(emotion_class_true, emotion_class_preds)
        mae_metric.update_state(emotion_intensity_true, emotion_intensity_preds)
        pbar.set_postfix({
            'Loss': mean_loss_class.result().numpy(),
            'Accuracy': accuracy_metric.result().numpy()
        })
    
    print(heat)

    heat /= cnt
    
    print(f"Test Emotion Class Loss: {mean_loss_class.result().numpy()}")
    print(f"Test Emotion Intensity Loss: {mean_loss_intensity.result().numpy()}")
    print(f"Test Emotion Class Accuracy: {accuracy_metric.result().numpy()}")
    print(f"Test Emotion Intensity MAE: {mae_metric.result().numpy()}")

    print(heat)