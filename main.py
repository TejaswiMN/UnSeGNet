from segment import Segmentation
from datasets.dataset import Dataset
from argparse import ArgumentParser
import torch
import matplotlib.pyplot as plt
import os

# """
# for venv:
# python -m venv venv
# Set-ExecutionPolicy RemoteSigned -Scope Process
# .\venv2\Scripts\activate 
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
parser.add_argument("--debug", type=bool, default=False)
parser.add_argument("--debug_samples(number)", type=int, default=-1)

args = parser.parse_args()

if __name__ == '__main__':
    # Check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    filename = "_".join(args.conv_type, args.activation, args.loss_type, args.dataset, args.process, args.threshold, args.epochs)
    log_file = "log_"+ filename
    image_file_folder = "../Failed_" + filename
    os.mkdir(image_file_folder)

    seg = Segmentation(args.process, args.bs, args.epochs, tuple(args.resolution), args.activation, args.loss_type, args.threshold, args.conv_type)
    # seg = Segmentation(args.process, args.bs, args.epochs, tuple(args.resolution), args.activation, args.loss_type, args.threshold, args.conv_type).to(device)
    torch.cuda.empty_cache()
    ds = Dataset(args.dataset)

    total_iou = 0
    total_samples = 0
    while ds.loader > 0:
        for img, mask in ds.load_samples():
            # img, mask = img.to(device), mask.to(device)  # Ensure data is also on GPU
            try: 
                iou, segmentation, segmentation_over_image = seg.segment(img, mask)
                total_iou += iou
                total_samples += 1
                if args.debug and iou <= 0.3:
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    axes[0].imshow(segmentation_over_image)
                    axes[0].axis("off")
                    axes[1].imshow(segmentation)
                    axes[1].axis("off")
                    axes[2].imshow(mask)
                    axes[2].axis("off")
                    plt.savefig(f'{image_file_folder}/failed_{total_samples}.png', bbox_inches='tight', dpi=300)
                    plt.close(fig)
                print(f"IoU for Current Image {total_samples} : {iou:.2f}  mIoU so far: {(total_iou/total_samples):.2f}")
                fl = f"{log_file}.log"
                with open(fl, "a") as f:
                    f.write(f"IoU for Current Image {total_samples} : {iou:.2f}  mIoU so far: {(total_iou/total_samples):.2f}\n")

                torch.cuda.empty_cache()

                if args.debug and args.debug_samples!=-1 and total_samples>args.debug_samples:
                    exit()

            except Exception as e:
                print(e)
                continue
    
    print(f'Final mIoU: {(total_iou / total_samples):.4f}')

torch.cuda.empty_cache()