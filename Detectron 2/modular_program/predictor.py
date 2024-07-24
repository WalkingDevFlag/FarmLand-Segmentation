from detectron2.utils.visualizer import Visualizer, ColorMode

def perform_inference(predictor, image):
    outputs = predictor(image)
    return outputs

def visualize_predictions(image, outputs):
    v = Visualizer(image[:, :, ::-1], instance_mode=ColorMode.SEGMENTATION)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    segmented_image = out.get_image()[:, :, ::-1]
    return segmented_image
