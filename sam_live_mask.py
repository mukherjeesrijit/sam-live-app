import os
import numpy as np
import nibabel as nib
from PIL import Image
import pandas as pd
from skimage.io import imread
import matplotlib.pyplot as plt
import cv2
import shutil
import cv2
import torch
import base64
import numpy as np
import supervision as sv
from pathlib import Path
from supervision.assets import download_assets, VideoAssets
from sam2.build_sam import build_sam2_video_predictor
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import os
from skimage import segmentation, measure
from sklearn.cluster import KMeans
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from scipy.ndimage import binary_fill_holes
from skimage.segmentation import mark_boundaries
import glob

def show_mask(mask, ax, random_color=False, borders = False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

# Callback function for mouse clicks and bounding box drawing
def click_event_with_buttons(event, x, y, flags, param):
    global points, labels, rectangles, drawing, ix, iy, active_mode, image_display, original_image, base_image_display

    predictor = param["predictor"]
    
    if y >= image_display.shape[0]:  # Check if click is on the button area
        button_y = y - image_display.shape[0]
        if 0 <= button_y < button_height:
            if 0 <= x < button_width:  # Positive button
                active_mode = "positive"
                print("Active mode changed to: positive")
            elif button_width <= x < 2 * button_width:  # Negative button
                active_mode = "negative"
                print("Active mode changed to: negative")
            elif 2 * button_width <= x < 3 * button_width:  # Bounding box button
                active_mode = "bounding_box"
                print("Active mode changed to: bounding_box")
        return

    if active_mode == "positive" and event == cv2.EVENT_LBUTTONDOWN:  # Left-click (Positive)
        points.append([x, y])
        labels.append(1)
        run_live_prediction(predictor)

    elif active_mode == "negative" and event == cv2.EVENT_LBUTTONDOWN:  # Left-click (Negative)
        points.append([x, y])
        labels.append(0)
        run_live_prediction(predictor)

    elif active_mode == "bounding_box" and event == cv2.EVENT_LBUTTONDOWN:  # Start drawing bounding box
        drawing = True
        ix, iy = x, y

    elif active_mode == "bounding_box" and event == cv2.EVENT_MOUSEMOVE and drawing:  # Update bounding box while drawing
        temp_image = image_display.copy()
        cv2.rectangle(temp_image, (ix, iy), (x, y), (255, 0, 0), 2)
        combined_image = np.vstack((temp_image, button_image))
        cv2.imshow("Annotate Points", combined_image)  # Show temporary bounding box

    elif active_mode == "bounding_box" and event == cv2.EVENT_LBUTTONUP and drawing:  # Finish drawing bounding box
        drawing = False
        rectangles.append((ix, iy, x, y))
        run_live_prediction(predictor)

    # Update display with buttons
    combined_image = np.vstack((image_display, button_image))
    cv2.imshow("Annotate Points", combined_image)

def run_live_prediction(predictor):
    global points, labels, rectangles, image_display, base_image_display, resize_scale, original_image, previous_logits
    
    if not points and not rectangles:
        return
    
    # Start with base image
    image_display = base_image_display.copy()
    
    # Rescale points to original image size
    pts = np.array(points, dtype=np.float32)
    pts_rescaled = pts.copy()
    pts_rescaled[:, 0] *= resize_scale[0]
    pts_rescaled[:, 1] *= resize_scale[1]
    lbls = np.array(labels, dtype=np.int32)
    
    # Rescale box to original image size
    box = None
    if rectangles:
        r = rectangles[-1]
        box = np.array([r[0]*resize_scale[0], r[1]*resize_scale[1],
                       r[2]*resize_scale[0], r[3]*resize_scale[1]], dtype=np.float32)
    
    # Run prediction with previous logits
    with torch.inference_mode():
        masks, scores, logits = predictor.predict(
            point_coords=pts_rescaled if len(pts_rescaled) else None,
            point_labels=lbls if len(lbls) else None,
            box=box[None, :] if box is not None else None,
            mask_input=previous_logits,
            multimask_output=True,
        )
    
    # Sort by score and keep the best mask
    sorted_ind = np.argsort(scores)[::-1]
    mask = masks[sorted_ind[0]]  # Best mask
    previous_logits = logits[sorted_ind[0:1]]  # Keep only best logit with shape [1, 1, H, W]
    
    # Resize mask to display size
    mask_display = cv2.resize(mask.astype(np.uint8), 
                             (image_display.shape[1], image_display.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
    
    # Create colored mask overlay
    mask_color = np.zeros_like(image_display, dtype=np.uint8)
    mask_color[mask_display > 0] = [255, 144, 30]  # BGR: Blue=255, Green=144, Red=30
    
    # Blend with alpha
    alpha = 0.4
    image_display = cv2.addWeighted(image_display, 1, mask_color, alpha, 0)
    
    # Redraw points
    for (px, py), lbl in zip(points, labels):
        color = (0, 255, 0) if lbl == 1 else (0, 0, 255)
        cv2.circle(image_display, (int(px), int(py)), 5, color, -1)
    
    # Redraw boxes
    for r in rectangles:
        cv2.rectangle(image_display, (r[0], r[1]), (r[2], r[3]), (255, 0, 0), 2)

def get_points_labels_with_buttons(image, predictor):
    global points, labels, rectangles, drawing, ix, iy, image_display, active_mode, button_height, button_width, button_image, base_image_display, resize_scale, original_image, previous_logits

    # Store original image
    original_image = image.copy()
    
    # Initialize global variables
    points = []
    labels = []
    rectangles = []
    drawing = False
    ix, iy = -1, -1
    active_mode = "positive"  # Default mode
    previous_logits = None  # Reset logits for new annotation session

    # Resize image to fit screen
    screen_res = 1280, 720  # Example screen resolution
    scale_width = screen_res[0] / image.shape[1]
    scale_height = screen_res[1] / image.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(image.shape[1] * scale)
    window_height = int(image.shape[0] * scale)
    image_display = cv2.resize(image, (window_width, window_height))
    base_image_display = image_display.copy()  # Store base image for overlays

    # Adjust points to match resized image
    resize_scale = (image.shape[1] / window_width, image.shape[0] / window_height)

    # Create a blank image for buttons
    button_height = 50
    button_image = np.zeros((button_height, window_width, 3), dtype=np.uint8)

    # Draw buttons
    button_width = window_width // 3
    cv2.rectangle(button_image, (0, 0), (button_width, button_height), (0, 255, 0), -1)  # Positive button
    cv2.putText(button_image, "Positive (P)", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.rectangle(button_image, (button_width, 0), (2 * button_width, button_height), (0, 0, 255), -1)  # Negative button
    cv2.putText(button_image, "Negative (N)", (button_width + 10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.rectangle(button_image, (2 * button_width, 0), (window_width, button_height), (255, 0, 0), -1)  # Bounding box button
    cv2.putText(button_image, "Bounding Box (B)", (2 * button_width + 10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Combine image and buttons
    combined_image = np.vstack((image_display, button_image))

    cv2.imshow("Annotate Points", combined_image)
    cv2.setMouseCallback("Annotate Points", click_event_with_buttons, {"predictor": predictor})

    # Wait for user interaction
    while True:
        key = cv2.waitKey(1)
        if key == 13:  # Enter key to finish
            break

    cv2.destroyAllWindows()  # Close OpenCV window

    # Convert to numpy and adjust points back to original scale
    points_array = np.array(points, dtype=np.float32)
    points_array[:, 0] *= resize_scale[0]
    points_array[:, 1] *= resize_scale[1]
    labels_array = np.array(labels, dtype=np.int32)

    # Adjust rectangles back to original scale
    box_coords = None
    if rectangles:
        box_coords = []
        for rect in rectangles:
            x1, y1, x2, y2 = rect
            x1 *= resize_scale[0]
            y1 *= resize_scale[1]
            x2 *= resize_scale[0]
            y2 *= resize_scale[1]
            box_coords.append([x1, y1, x2, y2])
        box_coords = np.array(box_coords, dtype=np.float32)

    return points_array, labels_array, box_coords
        
def sam2_image(image_path):

    # image = imread(image_path, as_gray=True)
    # image = np.stack([image] * 3, axis=-1)

    image = imread(image_path) #.astype(np.float32)
    image = image[:,:,:3]

    sam2_checkpoint = rf"C:\Users\szm6596\OneDrive - The Pennsylvania State University\Desktop\PhD_Research\sam2\checkpoints\sam2.1_hiera_large.pt"
    model_cfg = rf"C:\Users\szm6596\OneDrive - The Pennsylvania State University\Desktop\PhD_Research\sam2\sam2\configs\sam2.1\sam2.1_hiera_l.yaml"

    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    predictor.set_image(image)

    points, labels, box_coords = get_points_labels_with_buttons(image, predictor)
    
    # Generate and save final mask if points or boxes were provided
    if len(points) > 0 or box_coords is not None:
        masks, scores, logits = predictor.predict(
            point_coords=points if len(points) > 0 else None,
            point_labels=labels if len(labels) > 0 else None,
            box=box_coords,
            mask_input=previous_logits,
            multimask_output=True,
        )
        
        # Sort by score and use the best mask
        sorted_ind = np.argsort(scores)[::-1]
        mask = masks[sorted_ind[0]]
        best_score = scores[sorted_ind[0]]
        print(f"Best mask score: {best_score:.3f}")
        
        # Save mask as PNG (binary: 0 or 255)
        mask_save_path = image_path.replace('.png', '_mask.png')
        cv2.imwrite(mask_save_path, (mask * 255).astype(np.uint8))
        print(f"Mask saved to: {mask_save_path}")
        
        # Also save the overlay visualization
        overlay_save_path = image_path.replace('.png', '_overlay.png')
        mask_overlay = mask.astype(np.uint8) * 255
        colored_mask = np.zeros_like(image, dtype=np.uint8)
        colored_mask[:, :, 0] = mask_overlay * 30 // 255
        colored_mask[:, :, 1] = mask_overlay * 144 // 255
        colored_mask[:, :, 2] = mask_overlay * 255 // 255
        
        alpha = 0.6
        overlaid_image = image.copy()
        overlaid_image[mask > 0] = (alpha * colored_mask[mask > 0] + (1 - alpha) * image[mask > 0]).astype(np.uint8)
        cv2.imwrite(overlay_save_path, cv2.cvtColor(overlaid_image, cv2.COLOR_RGB2BGR))
        print(f"Overlay saved to: {overlay_save_path}")
    
    return None

image_path = rf"image.png"
sam2_image(image_path)