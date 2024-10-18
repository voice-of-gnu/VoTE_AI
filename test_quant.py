import numpy as np
import tensorflow as tf
import os
from tqdm import tqdm
import librosa
import json
import glob

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

# TFLite 모델 파일 경로
tflite_model_path = 'quantized_model/yamnet_fp16_quant.tflite'
accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()

# TFLite 인터프리터 로드 및 텐서 할당
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# 입력 및 출력 세부사항 가져오기
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 입력 텐서 정보 (데이터 타입, 크기)
input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']

#print(input_details)
#print(output_details)

print("Input shape:", input_shape)
print("Input dtype:", input_dtype)


test_data_path = "valid/valid"

test_data_path = os.path.join(test_data_path, "indoor")

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

# 5. 여러 테스트 샘플에 대한 추론 수행
correct_predictions = 0
total_predictions = 10  # 테스트할 데이터 샘플 수

cnt = 84160

pbar =  tqdm(dataset, total=cnt)

for audio, labels in pbar:
    # input_data = audio.reshape(input_shape)
    input_data = audio
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[1]['index'])
    predicted_label = np.argmax(output_data)
    
    emotion_class_preds = predicted_label
    emotion_class_true = labels['emotion_class_output']
    
    accuracy_metric.update_state(emotion_class_true, emotion_class_preds)
    

print(f"Test Emotion Class Accuracy: {accuracy_metric.result().numpy()}")
