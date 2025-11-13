import os
import pathlib
import time
from typing import Tuple

import numpy as np
import tensorflow as tf


"""
Edge AI Prototype (Task 1)

Tools: TensorFlow/TensorFlow Lite (Colab or Raspberry Pi-friendly)
Goal:
- Train a lightweight image classifier (uses tf_flowers as a stand-in dataset)
- Convert to TensorFlow Lite and test it on a sample set
- Print accuracy metrics and save a TFLite model for edge deployment

Run:
- In Google Colab: Runtime > Change runtime type > GPU (optional), then run
- Locally with Python 3.9+ and TensorFlow installed

Note:
- We use 'tf_flowers' (5 classes) to simulate "recyclable item recognition".
- Replace the dataset with your recyclables dataset directory when available.
"""


AUTOTUNE = tf.data.AUTOTUNE
IMG_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 3  # keep small for prototype speed


def load_dataset() -> Tuple[tf.data.Dataset, tf.data.Dataset, int]:
	"""Load and preprocess the tf_flowers dataset with a train/val split."""
	(train_ds, val_ds), ds_info = tf.keras.utils.image_dataset_from_directory(
		# If you have your own dataset dir, set 'directory="data/recyclables"' and ensure subfolders per class
		directory=None,  # None triggers fallback loader below
		# This call is intentionally wrapped below; see fallback implementation.
	)


def _fallback_flowers_loader() -> Tuple[tf.data.Dataset, tf.data.Dataset, int]:
	"""Fallback loader using tf_flowers via tf.data (robust for Colab)."""
	import tensorflow_datasets as tfds
	(raw_train, raw_val), ds_info = tfds.load(
		"tf_flowers",
		split=["train[:80%]", "train[80%:]"],
		with_info=True,
		as_supervised=True,
	)
	num_classes = ds_info.features["label"].num_classes

	def preprocess(image, label):
		image = tf.image.resize(image, IMG_SIZE)
		image = tf.cast(image, tf.float32) / 255.0
		return image, label

	train_ds = (
		raw_train
		.map(preprocess, num_parallel_calls=AUTOTUNE)
		.cache()
		.shuffle(1000)
		.batch(BATCH_SIZE)
		.prefetch(AUTOTUNE)
	)
	val_ds = (
		raw_val
		.map(preprocess, num_parallel_calls=AUTOTUNE)
		.cache()
		.batch(BATCH_SIZE)
		.prefetch(AUTOTUNE)
	)
	return train_ds, val_ds, num_classes


def build_model(num_classes: int) -> tf.keras.Model:
	"""Create a lightweight transfer-learning model using MobileNetV2."""
	base = tf.keras.applications.MobileNetV2(
		input_shape=(*IMG_SIZE, 3),
		include_top=False,
		weights="imagenet",
	)
	base.trainable = False  # freeze for fast prototype training

	inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
	x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
	x = base(x, training=False)
	x = tf.keras.layers.GlobalAveragePooling2D()(x)
	x = tf.keras.layers.Dropout(0.2)(x)
	outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
	model = tf.keras.Model(inputs, outputs)
	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
		loss="sparse_categorical_crossentropy",
		metrics=["accuracy"],
	)
	return model


def convert_to_tflite(keras_model: tf.keras.Model, optimize: bool = True) -> bytes:
	"""Convert the Keras model to TFLite flatbuffer bytes."""
	converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
	if optimize:
		converter.optimizations = [tf.lite.Optimize.DEFAULT]
	tflite_model = converter.convert()
	return tflite_model


def evaluate_tflite_model(tflite_model: bytes, dataset: tf.data.Dataset) -> float:
	"""Evaluate TFLite model accuracy by running inference over a dataset."""
	interpreter = tf.lite.Interpreter(model_content=tflite_model)
	interpreter.allocate_tensors()

	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()
	input_index = input_details[0]["index"]
	output_index = output_details[0]["index"]

	correct = 0
	total = 0
	for batch_images, batch_labels in dataset:
		# Convert to numpy for TFLite interpreter
		images_np = batch_images.numpy()
		labels_np = batch_labels.numpy()
		for img, label in zip(images_np, labels_np):
			# Add batch dimension
			interpreter.set_tensor(input_index, np.expand_dims(img, axis=0).astype(np.float32))
			interpreter.invoke()
			preds = interpreter.get_tensor(output_index)
			pred_label = int(np.argmax(preds[0]))
			correct += int(pred_label == int(label))
			total += 1
	return correct / max(total, 1)


def main():
	print("Loading dataset...")
	try:
		train_ds, val_ds, num_classes = _fallback_flowers_loader()
	except Exception as e:
		print("Failed loading tf_flowers via TFDS:", e)
		print("Exiting.")
		return

	print(f"Classes: {num_classes}")

	print("Building model...")
	model = build_model(num_classes)
	model.summary()

	print("Training...")
	start_time = time.time()
	history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)
	train_time_s = time.time() - start_time

	val_loss, val_acc = model.evaluate(val_ds, verbose=0)
	print(f"Validation accuracy (Keras): {val_acc:.4f}")

	print("Converting to TensorFlow Lite...")
	tflite_bytes = convert_to_tflite(model, optimize=True)

	# Ensure output directory
	out_dir = pathlib.Path("Part 2") / "artifacts"
	out_dir.mkdir(parents=True, exist_ok=True)
	tflite_path = out_dir / "flowers_mobilenet_v2.tflite"
	with open(tflite_path, "wb") as f:
		f.write(tflite_bytes)
	print(f"Saved TFLite model to: {tflite_path}")

	print("Evaluating TFLite model...")
	tflite_acc = evaluate_tflite_model(tflite_bytes, val_ds)
	print(f"Validation accuracy (TFLite): {tflite_acc:.4f}")

	# Summarize metrics
	report = [
		"Task 1 Metrics",
		f"- Epochs: {EPOCHS}",
		f"- Train time (s): {train_time_s:.2f}",
		f"- Keras val accuracy: {val_acc:.4f}",
		f"- TFLite val accuracy: {tflite_acc:.4f}",
	]
	report_path = out_dir / "task1_metrics.txt"
	with open(report_path, "w", encoding="utf-8") as f:
		f.write("\n".join(report))
	print(f"Wrote metrics to: {report_path}")

	print("Done.")


if __name__ == "__main__":
	main()


