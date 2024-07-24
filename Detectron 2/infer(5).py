import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import os
import cv2
import rasterio
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import geopandas as gpd
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode

# Configuration setup
cfg = get_cfg()
cfg.OUTPUT_DIR = r"E:\Random Python Scripts\FarmLand-Segmentation-main\Detectron 2\Models\model_50000_epochs (7.5k)"
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # Update if needed
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
#cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
predictor = DefaultPredictor(cfg)

# Directory paths
input_dir = r"E:\Random Python Scripts\FarmLand-Segmentation-main\Detectron 2\infer 2.0 trial\GeoRef Data"  # Change to your input directory
output_dir = r"E:\Random Python Scripts\FarmLand-Segmentation-main\Detectron 2\infer 2.0 trial\predicted images"  # Change to your output directory
shapefile_dir = r"E:\Random Python Scripts\FarmLand-Segmentation-main\Detectron 2\infer 2.0 trial\result"  # Change to your shapefile output directory

# Ensure output directories exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(shapefile_dir):
    os.makedirs(shapefile_dir)

def read_tiff(file_path):
    with rasterio.open(file_path) as src:
        image = src.read([1, 2, 3])  # Read RGB bands
        image = image.transpose((1, 2, 0))  # Reorder dimensions
    return image

def display_image(image, title="Image"):
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

def predict_and_save(file_path, output_image_path):
    # Read TIFF file
    image = read_tiff(file_path)
    
    # Perform inference
    outputs = predictor(image)
    
    # Visualize results
    v = Visualizer(image[:, :, ::-1], instance_mode=ColorMode.SEGMENTATION)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    
    # Save the segmented image
    segmented_image = out.get_image()[:, :, ::-1]
    cv2.imwrite(output_image_path, cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))
    
    print(f"Segmented image saved to: {output_image_path}")

def handle_overlapping_polygons(polygons):
    updated_polygons = []
    
    for poly in polygons:
        # Check if the polygon is valid
        if not poly.is_valid:
            poly = poly.buffer(0)  # Attempt to repair invalid polygons
        
        for existing_poly in updated_polygons:
            # Check if the existing polygon is valid
            if not existing_poly.is_valid:
                existing_poly = existing_poly.buffer(0)  # Attempt to repair invalid polygons
                
            if poly.intersects(existing_poly):
                intersection = poly.intersection(existing_poly)
                intersection_area = intersection.area
                poly_area = poly.area
                existing_poly_area = existing_poly.area

                if intersection_area / poly_area > 0.5 or intersection_area / existing_poly_area > 0.5:
                    if poly_area > existing_poly_area:
                        poly = poly.union(existing_poly)
                        updated_polygons.remove(existing_poly)
                    else:
                        existing_poly = existing_poly.union(poly)
                        poly = existing_poly
                
                elif intersection_area / poly_area <= 0.5 and intersection_area / existing_poly_area <= 0.5:
                    if poly_area > existing_poly_area:
                        poly = poly.difference(existing_poly)
                    else:
                        existing_poly = existing_poly.difference(poly)
                        
        updated_polygons.append(poly)

    return updated_polygons

def create_polygons_from_masks(masks, transform):
    polygons = []
    for mask in masks:
        mask = mask.numpy()
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if len(contour) >= 3:  # valid polygon must have at least 3 points
                poly_points = [transform * (point[0], point[1]) for point in contour.squeeze()]
                polygon = Polygon(poly_points)
                
                # Check if the polygon is valid
                if not polygon.is_valid:
                    polygon = polygon.buffer(0)  # Attempt to repair invalid polygons
                
                polygons.append(polygon)
    return polygons

def create_shapefile_from_predictions(outputs, shapefile_folder, shapefile_base_name, transform, crs):
    pred_masks = outputs["instances"].pred_masks.to("cpu")
    polygons = create_polygons_from_masks(pred_masks, transform)
    polygons = handle_overlapping_polygons(polygons)
    
    # Check if there are valid polygons to save
    if not polygons:
        print(f"No valid polygons found for {shapefile_base_name}. Skipping shapefile creation.")
        return
    
    gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
    
    # Create directory for shapefile
    os.makedirs(shapefile_folder, exist_ok=True)
    
    # Save shapefile
    shapefile_path = os.path.join(shapefile_folder, shapefile_base_name + ".shp")
    gdf.to_file(shapefile_path)
    
    print(f"Shapefile created: {shapefile_path}")

def process_tiff_files(input_dir, output_dir, shapefile_dir):
    tiff_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.tiff', '.tif', '.TIF', '.TIFF'))]

    for file_name in tiff_files:
        file_path = os.path.join(input_dir, file_name)
        output_image_path = os.path.join(output_dir, file_name.replace('.tiff', '.jpg').replace('.tif', '.jpg').replace('.TIFF', '.jpg').replace('.TIF', '.jpg'))
        shapefile_folder = os.path.join(shapefile_dir, os.path.splitext(file_name)[0])
        base_name = os.path.splitext(file_name)[0]

        # Read image for transform and crs
        with rasterio.open(file_path) as src:
            transform = src.transform
            crs = src.crs

        # Predict and save segmented image
        predict_and_save(file_path, output_image_path)
        
        # Perform prediction again to get the outputs for shapefile creation
        image = read_tiff(file_path)
        outputs = predictor(image)
        
        # Create shapefile from predictions
        create_shapefile_from_predictions(outputs, shapefile_folder, base_name, transform, crs)

# Run the processing
process_tiff_files(input_dir, output_dir, shapefile_dir)

print("Processing complete. Segmented images and shapefiles saved.")
