from pathlib import Path
import argparse

import tensorflow as tf
from keras import layers, models, callbacks

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "seeds"
DEFAULT_MODEL_PATH = ROOT / "model1.h5"
DEFAULT_CLASSES_PATH = ROOT / "classes.txt"
DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_BATCH_SIZE = 16
DEFAULT_VALIDATION_SPLIT = 0.20
DEFAULT_SEED = 42
DEFAULT_EPOCHS = 15
AUTOTUNE = tf.data.AUTOTUNE


def build_dataset(data_dir: Path, image_size, batch_size, validation_split, seed):
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
    )

    class_names = train_ds.class_names
    normalization = layers.Rescaling(1.0 / 255.0)
    train_ds = train_ds.map(lambda x, y: (normalization(x), y), num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (normalization(x), y), num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, class_names


def save_class_names(class_names, classes_path: Path):
    classes_path.write_text("\n".join(class_names) + "\n", encoding="utf-8")


def build_model(num_classes, image_size):
    model = models.Sequential([
        layers.Input(shape=(*image_size, 3)),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train model1 on the seeds dataset.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Root directory for seed class folders")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH,
                        help="Path to save the trained model")
    parser.add_argument("--classes-path", type=Path, default=DEFAULT_CLASSES_PATH,
                        help="Path to save the class names")
    parser.add_argument("--image-size", type=int, nargs=2, default=list(DEFAULT_IMAGE_SIZE),
                        help="Image size for training, e.g. --image-size 224 224")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--validation-split", type=float, default=DEFAULT_VALIDATION_SPLIT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    return parser.parse_args()


def main():
    args = parse_args()
    train_ds, val_ds, class_names = build_dataset(
        args.data_dir,
        tuple(args.image_size),
        args.batch_size,
        args.validation_split,
        args.seed,
    )

    save_class_names(class_names, args.classes_path)

    model = build_model(len(class_names), tuple(args.image_size))
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    checkpoint = callbacks.ModelCheckpoint(
        filepath=str(args.model_path),
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[checkpoint, early_stop],
    )

    model.save(args.model_path)
    print(f"Saved trained model to {args.model_path}")


if __name__ == "__main__":
    main()
