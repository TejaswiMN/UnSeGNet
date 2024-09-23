from segment import Segmentation
from datasets.dataset import Dataset
from argparse import ArgumentParser
import torch

# """
# for venv:
# python -m venv venv
# Set-ExecutionPolicy RemoteSigned -Scope Process
# .\venv\Scripts\activate 
# """

parser = ArgumentParser()
parser.add_argument("--bs", type=bool, default=False)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--resolution", nargs=2, type=int, default=[224, 224])
parser.add_argument("--activation", type=str, default='selu')
parser.add_argument("--loss_type", type=str, default='DMON')
parser.add_argument("--process", type=str, default='DINO')
parser.add_argument("--dataset", type=str, default='ECSSD')
parser.add_argument("--threshold", type=float, default=0)
parser.add_argument("--conv_type", type=str, default='ARMA')

args = parser.parse_args()

if __name__ == '__main__':
    # Check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")


    seg = Segmentation(args.process, args.bs, args.epochs, tuple(args.resolution), args.activation, args.loss_type, args.threshold, args.conv_type)
    # seg = Segmentation(args.process, args.bs, args.epochs, tuple(args.resolution), args.activation, args.loss_type, args.threshold, args.conv_type).to(device)

    ds = Dataset(args.dataset)

    total_iou = 0
    total_samples = 0
    while ds.loader > 0:
        for img, mask in ds.load_samples():
            # img, mask = img.to(device), mask.to(device)  # Ensure data is also on GPU
            try: 
                iou, _, _ = seg.segment(img, mask)
                total_iou += iou
                total_samples += 1
                print(f"IoU for Current Image: {iou:.2f}  mIoU so far: {(total_iou/total_samples):.2f}")

            except Exception as e:
                print(e)
                continue
    
    print(f'Final mIoU: {(total_iou / total_samples):.4f}')

