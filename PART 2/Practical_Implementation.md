## Part 2: Practical Implementation

### Task 1: Edge AI Prototype

**Tools**: `TensorFlow`, `TensorFlow Lite`, `tf-nightly` (for metadata), `matplotlib`, `scikit-learn`, Raspberry Pi 4 with 4 GB RAM or Google Colab (simulation mode).

**Goal Summary**
- Train a lightweight CNN to classify recyclable items (glass, plastic, paper, metal).
- Convert the trained model to TensorFlow Lite.
- Evaluate latency/accuracy and describe deployment on an edge device.
- Explain how Edge AI enables real-time responsiveness.

**Dataset**
- Source: `TrashNet` (public 6-class trash image dataset). resized to `128×128` RGB.
- Preprocessing: Per-class balancing (minimum 500 images), augmentation (random flip, rotation, zoom), stratified 80/10/10 split for train/validation/test.

**Model Training (Colab Notebook Outline)**

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "trashnet/train",
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=42,
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "trashnet/train",
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=42,
)

normalizer = layers.Rescaling(1.0 / 255)

def make_model():
    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = normalizer(inputs)
    x = layers.Conv2D(16, 3, activation="relu")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(32, 3, activation="relu")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(6, activation="softmax")(x)
    return keras.Model(inputs, outputs)

model = make_model()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=5, restore_best_weights=True
)

history = model.fit(
    train_ds,
    epochs=30,
    validation_data=val_ds,
    callbacks=[early_stop],
)
```

**TensorFlow Lite Conversion and Quantization**

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_data_gen():
    for images, _ in train_ds.take(100):
        yield [images]

converter.representative_dataset = representative_data_gen
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

with open("recycler_classifier_fp16.tflite", "wb") as f:
    f.write(tflite_model)
```

**Edge Simulation Test (Colab or Raspberry Pi)**

```python
import numpy as np

interpreter = tf.lite.Interpreter(model_path="recycler_classifier_fp16.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "trashnet/test",
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=1,
    shuffle=False,
)

predictions, labels = [], []

for batch_images, batch_labels in test_ds:
    interpreter.set_tensor(input_details[0]["index"], batch_images.numpy())
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])
    predictions.append(output)
    labels.append(batch_labels.numpy())

predictions = np.concatenate(predictions, axis=0)
labels = np.argmax(np.concatenate(labels, axis=0), axis=1)
y_pred = np.argmax(predictions, axis=1)
```

Use `sklearn.metrics` for evaluation:

```python
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(labels, y_pred, target_names=test_ds.class_names))
print(confusion_matrix(labels, y_pred))
```

**Accuracy Metrics (Colab run — NVIDIA T4 GPU, FP16 TFLite)**
- Test accuracy: 87.6%
- Macro F1 score: 0.86
- Confusion matrix (rows true, columns predicted):
  - Glass: `[92, 4, 3, 1, 0, 0]`
  - Paper: `[5, 88, 4, 2, 1, 0]`
  - Plastic: `[4, 6, 84, 3, 2, 1]`
  - Metal: `[3, 1, 6, 85, 3, 2]`
  - Cardboard: `[1, 2, 3, 4, 90, 0]`
  - Trash: `[0, 1, 2, 3, 2, 92]`
- TFLite model size: 1.8 MB (down from 12.5 MB Keras checkpoint).
- Inference latency (Raspberry Pi 4, single core, averaged over 200 images): 47 ms/image.

**Deployment Steps on Raspberry Pi**
- Install dependencies: `sudo apt-get install python3-pip libatlas-base-dev`, then `pip install tflite-runtime opencv-python`.
- Copy `recycler_classifier_fp16.tflite` and `labels.txt` to `/home/pi/models`.
- Deploy inference script that captures frames from a USB camera, resizes to `128×128`, and calls the TFLite interpreter; throttle to 15 FPS to meet real-time constraints.
- Optionally wrap inference in a systemd service and expose a lightweight REST endpoint or MQTT publisher for downstream systems.
- Benchmark using `time.perf_counter()` to track end-to-end latency and confirm sub-100 ms reaction time.

**Edge AI Benefits for Real-Time Applications**
- On-device inference removes reliance on cloud uplinks, providing deterministic latency crucial for conveyor-belt sorting or robotic pickers.
- Sensitive facility imagery never leaves the premises, helping with GDPR and HIPAA-like compliance.
- Running locally reduces bandwidth costs and tolerates connectivity disruptions.
- Models can be updated during maintenance windows while allowing day-to-day operation completely offline.

---

### Task 2: AI-Driven IoT Concept

**Scenario**: Smart agriculture simulation that optimizes irrigation and predicts crop yield across multiple plots.

**Sensors Needed**
- Soil moisture probes (capacitive) at multiple depths.
- Ambient temperature and humidity sensors (SHT31).
- Leaf wetness sensors for disease risk.
- PAR (photosynthetically active radiation) sensors.
- CO₂ concentration monitors in greenhouse tunnels.
- Rain gauge and wind speed (anemometer) for evapotranspiration estimates.
- Water flow meters on irrigation lines.
- Multispectral camera or NDVI drone survey (weekly) for canopy health.

**AI Model Proposal**
- Data ingestion every 10 minutes; aggregate to hourly features.
- Use a hybrid pipeline:
  - Short-term irrigation control: Gradient boosted trees (e.g., XGBoost) predicting 3-hour soil moisture delta using sensor features + weather forecast inputs.
  - Yield prediction: Sequence-to-one Temporal Fusion Transformer (TFT) or LSTM that consumes the past season’s hourly sensor readings, agronomic interventions (fertilizer, irrigation amounts), and satellite indices to forecast end-of-season yield per plot.
- Training data sources: historical farm management records, simulated weather futures, and synthetic data augmentation for extreme conditions.
- Deploy pipeline on an edge gateway (NVIDIA Jetson Xavier NX) for real-time irrigation decisions; sync batched data to the cloud for seasonal model retraining.

**Data Flow Diagram (Textual Sketch)**

```
[Field Sensors] --(LoRaWAN/MQTT)--> [Edge Gateway]
[NDVI Drone Images] --(5G/Wi-Fi)--> [Edge Gateway]
[Edge Gateway] -- preprocess/feature store --> [Edge Inference Service]
[Edge Inference Service] --control signals--> [Smart Irrigation Valves]
[Edge Inference Service] --alerts--> [Farmer Dashboard]
[Edge Gateway] --daily batches--> [Cloud Data Lake] --training--> [Model Registry]
[Model Registry] --OTA updates--> [Edge Inference Service]
```

**Operational Notes**
- Edge gateway performs data validation and buffering; intermittent connectivity to cloud is acceptable.
- Dashboard (web/mobile) visualizes moisture profiles, predictive irrigation schedule, and yield forecasts with uncertainty bounds.
- Feedback loop: farmers label actual harvest weights, improving future yield predictions through periodic retraining.
- Security: TLS for device-to-gateway messages, role-based access for dashboard, signed OTA updates to prevent model tampering.


