from bilateral_solver import bilateral_solver_output
from features_extract import deep_features
from torch_geometric.data import Data
from extractor import ViTExtractor
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import torch
import util
import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from gnn_pool import GNNpool 
from transformers import SamModel, SamProcessor

class Segmentation:
    def __init__(self, process, bs=False, epochs=20, resolution=(224, 224), activation=None, loss_type=None, threshold=None, conv_type=None, seg_type=None):
        if process not in ["KMEANS_DINO", "DINO", "MEDSAM_INFERENCE"]:
            raise ValueError(f'Process: {process} is not supported')
        self.process = process
        self.resolution = resolution
        self.bs = bs
        self.epochs = epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_type = loss_type
        self.threshold = threshold
        self.seg_type = seg_type
        if process in ["DINO", "KMEANS_DINO"]:
            self.feats_dim = 384
            pretrained_weights = './dino_deitsmall8_pretrain_full_checkpoint.pth'
            if not os.path.exists(pretrained_weights):
                url = 'https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain_full_checkpoint.pth'
                util.download_url(url, pretrained_weights)
            self.extractor = ViTExtractor(model_dir=pretrained_weights, device=self.device)
            self.model = GNNpool(self.feats_dim, 64, 32, 2, self.device, activation, loss_type, conv_type).to(self.device)
        elif process == "MEDSAM_INFERENCE":
            self.processor = SamProcessor.from_pretrained("wanglab/medsam-vit-base")
            self.model = SamModel.from_pretrained("wanglab/medsam-vit-base").to(self.device)
        torch.save(self.model.state_dict(), 'model.pt')
        self.model.train()

    def segment2(self, image, mask): # not using this
    #     """
    #     @param image: Image to segment (numpy array)
    #     @param mask: Ground truth mask (binary numpy array)
    #     """
    #     if self.process == "MEDSAM_INFERENCE":
    #         medsam_seg = self.medsam_inference(image, mask)
    #         segmentation = cv2.resize(medsam_seg.astype('float'), (image[:, :, 0].shape[1], image[:, :, 0].shape[0]), interpolation=cv2.INTER_NEAREST)
    #     else:
    #         image_tensor, image = util.load_data_img(image, self.resolution)
    #         if self.process in ["DINO", "KMEANS_DINO"]:
    #             F = deep_features(image_tensor, self.extractor, device=self.device)

    #         if self.process == "KMEANS_DINO":
    #             kmeans = KMeans(n_clusters=2, random_state = 42)
    #             kmeans.fit(F)
    #             labels = kmeans.labels_
    #             S = torch.tensor(labels)
    #         else:
    #             W = util.create_adj(F, self.loss_type, self.threshold)
    #             node_feats, edge_index, edge_weight = util.load_data(W, F)
    #             data = Data(node_feats, edge_index, edge_weight).to(self.device)
    #             self.model.load_state_dict(torch.load('./model.pt', map_location=self.device))
    #             opt = optim.AdamW(self.model.parameters(), lr=0.001)

    #             for _ in range(self.epochs):
    #                 opt.zero_grad()
    #                 A, S = self.model(data, torch.from_numpy(W).to(self.device))
    #                 loss = self.model.loss(A, S)
    #                 loss.backward()
    #                 opt.step()

    #             S = S.detach().cpu()
    #             S = torch.argmax(S, dim=-1)
                
    #         segmentation = util.graph_to_mask(S, image_tensor, image)

    #     if self.bs:
    #         segmentation = bilateral_solver_output(image, segmentation)[1]
        
    #     segmentation = np.where(segmentation==True, 1,0).astype(np.uint8)  
        
    #     iou = Segmentation.iou(segmentation, mask)
    #     segmentation_over_image  = util.apply_seg_map(image, segmentation, 0.1)

    #     return iou, segmentation, segmentation_over_image
        pass

    def segment(self, image, mask):
        """
        @param image: Image to segment (numpy array)
        @param mask: Ground truth mask (binary numpy array)
        """
        if self.process == "MEDSAM_INFERENCE":
            medsam_seg = self.medsam_inference(image, mask)
            segmentation = cv2.resize(medsam_seg.astype('float'), 
                                    (image[:, :, 0].shape[1], 
                                        image[:, :, 0].shape[0]), 
                                    interpolation=cv2.INTER_NEAREST)
        else:
            image_tensor, image = util.load_data_img(image, self.resolution)
            if self.process in ["DINO", "KMEANS_DINO"]:
                F = deep_features(image_tensor, self.extractor, device=self.device)

            if self.process == "KMEANS_DINO":
                kmeans = KMeans(n_clusters=2, random_state=42)
                kmeans.fit(F)
                labels = kmeans.labels_
                S = torch.tensor(labels)
            else:
                W = util.create_adj(F, self.loss_type, self.threshold)
                node_feats, edge_index, edge_weight = util.load_data(W, F)
                data = Data(node_feats, edge_index, edge_weight).to(self.device)
                self.model.load_state_dict(torch.load('./model.pt', map_location=self.device))
                opt = optim.AdamW(self.model.parameters(), lr=0.001)

                for _ in range(self.epochs):
                    opt.zero_grad()
                    A, S = self.model(data, torch.from_numpy(W).to(self.device))
                    loss = self.model.loss(A, S)
                    loss.backward()
                    opt.step()

                S = S.detach().cpu()
                S = torch.argmax(S, dim=-1)

            segmentation = util.graph_to_mask(S, image_tensor, image)

        if self.bs:
            segmentation = bilateral_solver_output(image, segmentation)[1]

        segmentation = np.where(segmentation == True, 1, 0).astype(np.uint8)

        # if self.seg_type == "simple":
        #     iou1 = Segmentation.iou(segmentation, mask)
        #     segmentation_over_image  = util.apply_seg_map(image, segmentation, 0.1)
        #     return iou1, segmentation, segmentation_over_image

        # Calculate IoU for all segments
        unique_segments = np.unique(segmentation)        
        best_iou = 0
        best_segment_id = None
        is_inverted = False  # Flag to indicate if the best segment is inverted

        for segment_id in unique_segments:
            if segment_id == 0:  # Skip background
                continue
            segment_mask = (segmentation == segment_id).astype(np.uint8)
            iou = Segmentation.iou(segment_mask, mask)

            # Check for inverted mask
            inverted_mask = (segment_mask == 0).astype(np.uint8)
            inverted_iou = Segmentation.iou(inverted_mask, mask)
            print ("All ious: ", iou, inverted_iou)

            # Use the maximum IoU from both normal and inverted
            if iou > inverted_iou:
                max_iou = iou
                current_segment_id = segment_id
                current_is_inverted = False
            else:
                max_iou = inverted_iou
                current_segment_id = segment_id
                current_is_inverted = True

            # Update best IoU and segment ID
            if max_iou > best_iou:
                best_iou = max_iou
                best_segment_id = current_segment_id
                is_inverted = current_is_inverted  # Track if the best segment is inverted

        # Create mask for best segment
        best_segment_mask = (segmentation == best_segment_id).astype(np.uint8)
        if is_inverted:  # If the best segment is inverted
            best_segment_mask = (best_segment_mask == 0).astype(np.uint8)  # Invert the mask
        segmentation_over_image = util.apply_seg_map(image, best_segment_mask, 0.1)

        return best_iou, best_segment_mask, segmentation_over_image


    def medsam_inference(self, image, mask):
        """
        @param image: Image to segment (numpy array)
        @param mask: Ground truth mask (binary numpy array)
        """
        image = cv2.resize(image.astype('float'), self.resolution, interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask.astype('float'), self.resolution, interpolation=cv2.INTER_NEAREST)
        mask = mask.astype('uint8')
        input_boxes = Segmentation.get_bounding_box(mask)
        inputs = self.processor(image, input_boxes=[[[input_boxes]]], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, multimask_output=False)
        
        medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
        # convert soft mask to hard mask
        medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
        medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)

        return medsam_seg

    @staticmethod
    def get_bounding_box(ground_truth_map):
        """
        Get bounding box from ground truth mask
        @param ground_truth_map: Binary numpy array
        """
        # get bounding box from mask
        y_indices, x_indices = np.where(ground_truth_map > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = ground_truth_map.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
        bbox = [x_min, y_min, x_max, y_max]

        return bbox
    
    @staticmethod
    def iou(mask1, mask2):
        """
        Calculate Intersection over Union between two masks
        @param mask1: Binary numpy array
        @param mask2: Binary numpy array
        """
        x = mask1.ravel()
        y = mask2.ravel()
        intersection = np.logical_and(x, y)
        union = np.logical_or(x, y)
        similarity = np.sum(intersection)/ np.sum(union)
        return similarity
