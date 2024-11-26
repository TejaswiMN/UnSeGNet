from segment import Segmentation
from datasets.dataset import Dataset
from argparse import ArgumentParser
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import os
import gc

# """
# for venv:
# python -m venv venv
# Set-ExecutionPolicy RemoteSigned -Scope Process
# .\venv2\Scripts\activate 
# """

parser = ArgumentParser()
parser.add_argument("--bs", type=bool, default=True)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--resolution", nargs=2, type=int, default=[224, 224])
parser.add_argument("--activation", type=str, default='selu')
parser.add_argument("--loss_type", type=str, default='DMON')
parser.add_argument("--process", type=str, default='DINO')
parser.add_argument("--dataset", type=str, default='ECSSD')
parser.add_argument("--threshold", type=float, default=0.3)
parser.add_argument("--conv_type", type=str, default='GAT')
parser.add_argument("--debug", type=bool, default=True)
parser.add_argument("--debug_samples", type=int, default=-1)
parser.add_argument("--seg_type", type=str, default='simple')

args = parser.parse_args()

if __name__ == '__main__':

    # Check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    filename = "_".join([args.conv_type, args.activation, args.loss_type, args.dataset, args.process, str(args.threshold), str(args.epochs)])
    log_file = "log_"+ "2heads_2clusters_" + filename
    # log_file = "log3_2clusters_3heads_ioulist_ncut_gat_ecssd_0.3"
    image_file_folder = "../Failed_" + log_file
    if os.path.exists(image_file_folder):
        now = datetime.now().strftime("_%m_%d_%Y_%H_%M_%S")
        log_file += now
        image_file_folder += now
    os.mkdir(image_file_folder)

    seg = Segmentation(args.process, args.bs, args.epochs, tuple(args.resolution), args.activation, args.loss_type, args.threshold, args.conv_type, args.seg_type)
    gc.collect()
    torch.cuda.empty_cache()
    ds = Dataset(args.dataset)

    total_iou = 0
    total_samples = 0

    while ds.loader > 0:
        for img, mask in ds.load_samples():
            #print(f"Loaded sample {total_samples}")  # Debugging line
            try:
                iou, best_segmentation, segmentation_over_image = seg.segment(img, mask)
                total_iou += iou
                total_samples += 1
              
                if args.debug and iou <= 0.5:
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    axes[0].imshow(segmentation_over_image)
                    axes[0].axis("off")
                    axes[1].imshow(best_segmentation)
                    axes[1].axis("off")
                    axes[2].imshow(mask)
                    axes[2].axis("off")
                    plt.savefig(f'{image_file_folder}/failed_{total_samples}.png', bbox_inches='tight', dpi=300)
                    plt.close(fig)
                    
                print(f"IoU for Current Image {total_samples} : {iou:.2f}  mIoU so far: {(total_iou/total_samples):.2f}")
                fl = f"{log_file}.log"
                with open(fl, "a") as f:
                    f.write(f"IoU for Current Image {total_samples} : {iou:.2f}  mIoU so far: {(total_iou/total_samples):.2f}\n")
                    
                gc.collect()
                torch.cuda.empty_cache()
                
                if args.debug and args.debug_samples != -1 and total_samples > args.debug_samples:
                    exit()

            except Exception as e:
                print(e)
                continue

    if total_samples > 0:
        print(f'Final mIoU: {(total_iou / total_samples):.4f}')
    else:
        print("No valid samples processed, mIoU cannot be computed.")


    gc.collect()
    torch.cuda.empty_cache()