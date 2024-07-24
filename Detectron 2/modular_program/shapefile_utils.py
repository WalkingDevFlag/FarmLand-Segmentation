import os
from shapely.geometry import Polygon
import geopandas as gpd
import numpy as np
import cv2

def handle_overlapping_polygons(polygons):
    updated_polygons = []
    for poly in polygons:
        if not poly.is_valid:
            poly = poly.buffer(0)  # Attempt to repair invalid polygons
        for existing_poly in updated_polygons:
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
            if len(contour) >= 3:
                poly_points = [transform * (point[0], point[1]) for point in contour.squeeze()]
                polygon = Polygon(poly_points)
                if not polygon.is_valid:
                    polygon = polygon.buffer(0)
                polygons.append(polygon)
    return polygons

def create_shapefile_from_predictions(outputs, shapefile_folder, shapefile_base_name, transform, crs):
    pred_masks = outputs["instances"].pred_masks.to("cpu")
    polygons = create_polygons_from_masks(pred_masks, transform)
    polygons = handle_overlapping_polygons(polygons)
    if not polygons:
        print(f"No valid polygons found for {shapefile_base_name}. Skipping shapefile creation.")
        return
    gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
    os.makedirs(shapefile_folder, exist_ok=True)
    shapefile_path = os.path.join(shapefile_folder, shapefile_base_name + ".shp")
    gdf.to_file(shapefile_path)
    print(f"Shapefile created: {shapefile_path}")
