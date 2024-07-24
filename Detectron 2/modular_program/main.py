import os
import rasterio  
from config import setup_config
from image_utils import read_tiff, save_image
from predictor import perform_inference, visualize_predictions
from shapefile_utils import create_shapefile_from_predictions

def process_tiff_files(input_dir, output_dir, shapefile_dir, predictor):
    tiff_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.tiff', '.tif', '.TIF', '.TIFF'))]
    for file_name in tiff_files:
        file_path = os.path.join(input_dir, file_name)
        output_image_path = os.path.join(output_dir, file_name.replace('.tiff', '.jpg').replace('.tif', '.jpg').replace('.TIFF', '.jpg').replace('.TIF', '.jpg'))
        shapefile_folder = os.path.join(shapefile_dir, os.path.splitext(file_name)[0])
        base_name = os.path.splitext(file_name)[0]

        with rasterio.open(file_path) as src:
            transform = src.transform
            crs = src.crs

        image = read_tiff(file_path)
        outputs = perform_inference(predictor, image)
        segmented_image = visualize_predictions(image, outputs)
        save_image(segmented_image, output_image_path)
        
        create_shapefile_from_predictions(outputs, shapefile_folder, base_name, transform, crs)

if __name__ == "__main__":
    cfg = setup_config()
    from detectron2.engine import DefaultPredictor
    predictor = DefaultPredictor(cfg)
    
    input_dir = r"E:\Random Python Scripts\FarmLand-Segmentation-main\Detectron 2\infer 2.0 trial\GeoRef Data"
    output_dir = r"E:\Random Python Scripts\FarmLand-Segmentation-main\Detectron 2\infer 2.0 trial\predicted images"
    shapefile_dir = r"E:\Random Python Scripts\FarmLand-Segmentation-main\Detectron 2\infer 2.0 trial\result"
    
    process_tiff_files(input_dir, output_dir, shapefile_dir, predictor)

    print("Processing complete. Segmented images and shapefiles saved.")
