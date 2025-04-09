# Import necessary libraries
from ultralytics import YOLO
import argparse


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--eval', action='store_true', help='Evaluate the model')
parser.add_argument('--no-eval', dest='eval', action='store_false', help='Predicting')
parser.set_defaults(eval=True)
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--device', type=int, default=0, help='Device to use')
parser.add_argument('--dataset', type=str, default='wgisd', help='Dataset to use')


args, _ = parser.parse_known_args()


# Load pretrained model
model = YOLO('global_model.pt')  


if args.eval:
    # Evaluation
    results = model.val(data='data_yolo.yaml', split='test', batch=args.batch_size, device=args.device)
    # Save evaluation results
    map_50 = results.box.map50  # mAP@50
    map_50_95 = results.box.map  # mAP@50-95

    with open("evaluation_results.txt", "w") as file:
        file.write("Metric,Value\n")
        file.write(f"mAP@50,{map_50:.4f}\n")
        file.write(f"mAP@50-95,{map_50_95:.4f}\n")

else:
    # Testing model
    results = model.predict(source=f"Datasets/{args.dataset}/test/images", save=True)



