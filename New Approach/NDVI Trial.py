from PIL import Image, ImageFilter
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import time
import os

matplotlib.use('Agg')

class PreProcess:

    def __init__(self, sourcePath, imagePath):
        self.imagePath = imagePath
        self.sourcePath = sourcePath

    def resize(self):
        if not os.path.exists(self.imagePath):
            raise FileNotFoundError(f"No such file or directory: '{self.imagePath}'")

        img = Image.open(self.imagePath)
        imgResize = img.resize((800, 600), Image.LANCZOS)
        self.timestamp = "{}-{}-{}-{}-{}-{}".format(time.gmtime().tm_year, time.gmtime().tm_mon, time.gmtime().tm_mday, time.gmtime().tm_hour, time.gmtime().tm_min, time.gmtime().tm_sec)
        output_dir = os.path.join(self.sourcePath, "output")
        os.makedirs(output_dir, exist_ok=True)
        imgResizePath = "{}/resized{}.png".format(output_dir, self.timestamp)
        imgResize.save(imgResizePath, "png")
        return imgResizePath

class NDVIProcess:

    def __init__(self):
        self.colors = [(1.0000, 1.0000, 1.0000), (0.9804, 0.9804, 0.9804), (0.9647, 0.9647, 0.9647), 
                       (0.9490, 0.9490, 0.9490), (0.9333, 0.9333, 0.9333), (0.9137, 0.9137, 0.9137),
                       (0.8980, 0.8980, 0.8980), (0.8824, 0.8824, 0.8824), (0.8667, 0.8667, 0.8667),
                       (0.8471, 0.8471, 0.8471), (0.8314, 0.8314, 0.8314), (0.8157, 0.8157, 0.8157),
                       (0.8000, 0.8000, 0.8000), (0.7843, 0.7843, 0.7843), (0.7647, 0.7647, 0.7647),
                       (0.7490, 0.7490, 0.7490), (0.7333, 0.7333, 0.7333), (0.7176, 0.7176, 0.7176),
                       (0.6980, 0.6980, 0.6980), (0.6824, 0.6824, 0.6824), (0.6667, 0.6667, 0.6667),
                       (0.6510, 0.6510, 0.6510), (0.6314, 0.6314, 0.6314), (0.6157, 0.6157, 0.6157),
                       (0.6000, 0.6000, 0.6000), (0.5843, 0.5843, 0.5843), (0.5686, 0.5686, 0.5686),
                       (0.5490, 0.5490, 0.5490), (0.5333, 0.5333, 0.5333), (0.5176, 0.5176, 0.5176),
                       (0.5020, 0.5020, 0.5020), (0.4824, 0.4824, 0.4824), (0.4667, 0.4667, 0.4667),
                       (0.4510, 0.4510, 0.4510), (0.4353, 0.4353, 0.4353), (0.4157, 0.4157, 0.4157),
                       (0.4000, 0.4000, 0.4000), (0.3843, 0.3843, 0.3843), (0.3686, 0.3686, 0.3686),
                       (0.3529, 0.3529, 0.3529), (0.3333, 0.3333, 0.3333), (0.3176, 0.3176, 0.3176),
                       (0.3020, 0.3020, 0.3020), (0.2863, 0.2863, 0.2863), (0.2667, 0.2667, 0.2667),
                       (0.2510, 0.2510, 0.2510), (0.2353, 0.2353, 0.2353), (0.2196, 0.2196, 0.2196),
                       (0.2039, 0.2039, 0.2039), (0.2196, 0.2196, 0.2196), (0.2353, 0.2353, 0.2353),
                       (0.2510, 0.2510, 0.2510), (0.2667, 0.2667, 0.2667), (0.2863, 0.2863, 0.2863),
                       (0.3020, 0.3020, 0.3020), (0.3176, 0.3176, 0.3176), (0.3333, 0.3333, 0.3333),
                       (0.3529, 0.3529, 0.3529), (0.3686, 0.3686, 0.3686), (0.3843, 0.3843, 0.3843),
                       (0.4000, 0.4000, 0.4000), (0.4157, 0.4157, 0.4157), (0.4353, 0.4353, 0.4353),
                       (0.4510, 0.4510, 0.4510), (0.4667, 0.4667, 0.4667), (0.4824, 0.4824, 0.4824),
                       (0.5020, 0.5020, 0.5020), (0.5176, 0.5176, 0.5176), (0.5333, 0.5333, 0.5333),
                       (0.5490, 0.5490, 0.5490), (0.5686, 0.5686, 0.5686), (0.5843, 0.5843, 0.5843),
                       (0.6000, 0.6000, 0.6000), (0.6157, 0.6157, 0.6157), (0.6314, 0.6314, 0.6314),
                       (0.6510, 0.6510, 0.6510), (0.6667, 0.6667, 0.6667), (0.6824, 0.6824, 0.6824),
                       (0.6980, 0.6980, 0.6980), (0.7176, 0.7176, 0.7176), (0.7333, 0.7333, 0.7333),
                       (0.7490, 0.7490, 0.7490), (0.7647, 0.7647, 0.7647), (0.7843, 0.7843, 0.7843),
                       (0.8000, 0.8000, 0.8000), (0.8157, 0.8157, 0.8157), (0.8314, 0.8314, 0.8314),
                       (0.8471, 0.8471, 0.8471), (0.8667, 0.8667, 0.8667), (0.8824, 0.8824, 0.8824),
                       (0.8980, 0.8980, 0.8980), (0.9137, 0.9137, 0.9137), (0.9333, 0.9333, 0.9333),
                       (0.9490, 0.9490, 0.9490), (0.9647, 0.9647, 0.9647), (0.9804, 0.9804, 0.9804),
                       (1.0000, 1.0000, 1.0000)]

    def calculate_ndvi(self, imagePath):
            if not os.path.exists(imagePath):
                raise FileNotFoundError(f"No such file or directory: '{imagePath}'")

            img = Image.open(imagePath)
            arr = np.array(img)
            band1 = arr[:,:,0].astype(np.float32)
            band2 = arr[:,:,1].astype(np.float32)
            ndvi = (band2 - band1) / (band2 + band1 + 1e-5)
            cmap = LinearSegmentedColormap.from_list("NDVI", self.colors)
            ndvi_img = cmap(ndvi)
            ndvi_img = (ndvi_img[:, :, :3] * 255).astype(np.uint8)
            ndvi_image = Image.fromarray(ndvi_img)
            
            # Get the directory of the input image
            directory = os.path.dirname(imagePath)
            
            # Generate the output file name based on the input file name
            input_filename = os.path.basename(imagePath)
            output_filename = os.path.splitext(input_filename)[0] + "_ndvi.png"
            
            # Construct the output file path
            output_path = os.path.join(directory, output_filename)
            
            ndvi_image.save(output_path)
            return output_path

def main(image_path):
        ndvi_processor = NDVIProcess()
        ndvi_image_path = ndvi_processor.calculate_ndvi(image_path)
        print(f"NDVI image saved at: {ndvi_image_path}")

if __name__ == "__main__":
    image_path = "E:\Sid Folder\Random Python Scripts\Farm Land Detection\Farmland2.png"
    main(image_path)
