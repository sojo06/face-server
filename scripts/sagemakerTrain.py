import os
import argparse
from training.train_script import train_model_from_directory

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-name', type=str, required=True)
    parser.add_argument('--train-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--output-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))

    args = parser.parse_args()

    model_name = args.model_name
    train_dir = args.train_dir
    output_dir = args.output_dir

    model_output_path = os.path.join(output_dir, f"{model_name}_face_recogniser.pkl")

    print(f"Training with data from: {train_dir}")
    print(f"Saving model to: {model_output_path}")

    # Call your existing training function
    train_model_from_directory(train_dir, model_output_path)

if __name__ == '__main__':
    main()
