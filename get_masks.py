import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
csv_file="/home/zlalena/data/ex2103-DIVE11-videos/example 1/locations.csv"
reader = csv.reader(open(csv_file))
mylines = list(reader)
mylines = mylines[1:]
image_folder = "/home/zlalena/data/ex2103-DIVE11-videos/example 1/images/"

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    for image_data in mylines:
        #print("Image data:", image_data)
        image_name = image_folder + image_data[0] +".png"
        #print("Image name:", image_name)
        image = cv2.imread(image_name)
        #print("Read image successfully:", image is not None)
        # cv2.imshow("image",image)
        # cv2.waitKey(0)
        predictor.set_image(image)
        #print("Set image successfully")
        i=1
        data = image_data
        mask_list = []
        #print("Data length:", len(data))
        while i < len(data):
            #print(f"i={i}, data[i]={data[i] if i < len(data) else 'N/A'}, data[i+1]={data[i+1] if i+1 < len(data) else 'N/A'}, data[i+2]={data[i+2] if i+2 < len(data) else 'N/A'}")
            if i+2 < len(data) and float(data[i+2])>0:
                point_cords =[]
                point_cords.append([float(data[i]),float(data[i+1])])
                if len(point_cords) == 0:
                    print("No valid points found for this image, skipping...")
                    continue
                point_cords = np.array(point_cords)
                # comfidende
                confidence = float(data[i+2])
                # point lables is array of 1 same length as point_cords
                point_labels = np.ones(len(point_cords))
                masks, _, _ = predictor.predict(point_coords=point_cords, point_labels=point_labels)
                if confidence > 0.15:
                    mask_list.append(masks[0])
                #print(f"Added point: [{float(data[i])}, {float(data[i+1])}] with confidence {float(data[i+2])}")
            i+=3
        combined_mask = np.zeros_like(image)
        for mask in mask_list:
            if mask.sum()/(mask.shape[0]*mask.shape[1]) < 0.10:
                combined_mask[mask.astype(bool)] = 1

        # save combined mask
        cv2.imwrite(f"/home/zlalena/data/ex2103-DIVE11-videos/example 1/masks/combined_mask_{image_data[0]}.png", combined_mask*255)
        print("Prediction completed")
      