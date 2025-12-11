import argparse
import math
import os
from pathlib import Path
import warnings
import tensorflow as tf
from sklearn.model_selection import train_test_split
from preprocessing import load_driving_log, balance_steering, batch_generator, plot_steering_histogram, clip_steering
from model import nvidia_model
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_history(history: tf.keras.callbacks.History, out_path: str) -> None:
    hist = history.history
    plt.figure(figsize=(8, 4))
    plt.plot(hist["loss"], label="train_loss")
    plt.plot(hist["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", help="Folder containing driving_log.csv and IMG/")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--balance", action="store_true", help="Enable steering angle balancing")
    parser.add_argument("--clip-angle", type=float, default=0.9, help="Clip steering angles to +/- this value before training")
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--output", default="model.keras")
    parser.add_argument("--plots-dir", default="artifacts")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    csv_path = data_dir / "driving_log.csv"
    output_path = Path(args.output)
    if output_path.suffix.lower() in {".h5", ".hdf5"}:
        adjusted = output_path.with_suffix(".keras")
        warnings.warn(f"Keras 3 requires .keras files; saving to {adjusted} instead of {output_path}")
        output_path = adjusted

    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = load_driving_log(str(csv_path), str(data_dir))
    df = clip_steering(df, args.clip_angle)
    plot_steering_histogram(df, str(plots_dir / "steering_hist_raw.png"))
    if args.balance:
        df = balance_steering(df)
        plot_steering_histogram(df, str(plots_dir / "steering_hist_balanced.png"))

    train_df, val_df = train_test_split(df, test_size=args.val_split, shuffle=True, random_state=42)
    train_gen = batch_generator(train_df, args.batch_size, is_training=True)
    val_gen = batch_generator(val_df, args.batch_size, is_training=False)

    steps_per_epoch = math.ceil(len(train_df) / args.batch_size)
    val_steps = math.ceil(len(val_df) / args.batch_size)

    model = nvidia_model(args.l2)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr), loss="mse", metrics=["mae"])

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(output_path), monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)
    ]

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=val_steps,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    plot_history(history, str(plots_dir / "training_curves.png"))
    print(f"Saved best model to {output_path}")

if __name__ == "__main__":
    main()
