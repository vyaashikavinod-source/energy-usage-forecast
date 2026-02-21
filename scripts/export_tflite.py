import argparse
from pathlib import Path
import tensorflow as tf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saved_model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sm = Path(args.saved_model)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    converter = tf.lite.TFLiteConverter.from_saved_model(str(sm))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    out.write_bytes(tflite_model)
    print(f"Wrote: {out} ({out.stat().st_size} bytes)")

if __name__ == "__main__":
    main()
