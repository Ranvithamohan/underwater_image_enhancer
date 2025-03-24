import sys
import math
import numpy as np
from typing import Tuple, Dict
from PIL import Image, ImageStat, ImageFilter, ImageOps

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = self.error_message_detail(error_message, error_detail=error_detail)
    
    def __str__(self):
        return self.error_message
    
    @staticmethod
    def error_message_detail(error, error_detail: sys) -> str:
        _, _, exc_tb = error_detail.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        error_message = f"Error occurred in python script [{file_name}] line [{exc_tb.tb_lineno}] error: {str(error)}"
        return error_message

def extract_image_features(image: Image.Image) -> Dict[str, float]:
    """
    Extract numerical features from an underwater image for quality prediction.
    Args:
        image: Input PIL Image in RGB format
    Returns:
        Dictionary containing 12 different image quality metrics
    """
    try:
        # Basic image properties
        width, height = image.size
        aspect_ratio = width / height
        
        # Color channel statistics
        r, g, b = image.split()
        r_mean = np.mean(np.array(r, dtype=np.float32))
        g_mean = np.mean(np.array(g, dtype=np.float32))
        b_mean = np.mean(np.array(b, dtype=np.float32))
        
        # Color balance ratios
        rg_ratio = r_mean / g_mean if g_mean != 0 else 0
        bg_ratio = b_mean / g_mean if g_mean != 0 else 0
        
        # Contrast and brightness from grayscale
        gray_image = image.convert('L')
        gray_array = np.array(gray_image)
        contrast = np.std(gray_array)
        brightness = np.mean(gray_array)
        
        # Entropy calculation
        def calculate_entropy(img):
            hist = img.histogram()
            hist_length = sum(hist)
            probs = [float(h) / hist_length for h in hist if h != 0]
            return -sum(p * math.log(p, 2) for p in probs)
        
        entropy = calculate_entropy(gray_image)
        
        # Edge strength (sharpness metric)
        edges = gray_image.filter(ImageFilter.FIND_EDGES)
        edge_strength = np.mean(np.array(edges))
        
        # Colorfulness metric (Hasler and Süsstrunk, 2003)
        rg = r_mean - g_mean
        yb = 0.5 * (r_mean + g_mean) - b_mean
        std_rg = np.std(np.array(r) - np.array(g))
        std_yb = np.std(0.5 * (np.array(r) + np.array(g)) - np.array(b))
        mean_rgyb = math.sqrt(std_rg**2 + std_yb**2)
        std_rgyb = math.sqrt(std_rg**2 + std_yb**2)
        colorfulness = mean_rgyb + 0.3 * std_rgyb
        
        return {
            'width': float(width),
            'height': float(height),
            'aspect_ratio': float(aspect_ratio),
            'r_mean': float(r_mean),
            'g_mean': float(g_mean),
            'b_mean': float(b_mean),
            'rg_ratio': float(rg_ratio),
            'bg_ratio': float(bg_ratio),
            'contrast': float(contrast),
            'brightness': float(brightness),
            'entropy': float(entropy),
            'edge_strength': float(edge_strength),
            'colorfulness': float(colorfulness)
        }
        
    except Exception as e:
        raise CustomException(f"Feature extraction failed: {str(e)}", sys)

def compensate_RB(image: Image.Image, flag: int) -> Image.Image:
    """[Your original docstring here]"""
    try:
        imager, imageg, imageb = image.split()
        imageR = np.array(imager, np.float64)
        imageG = np.array(imageg, np.float64)
        imageB = np.array(imageb, np.float64)
        
        minR, maxR = imager.getextrema()
        minG, maxG = imageg.getextrema()
        minB, maxB = imageb.getextrema()
        
        imageR = (imageR - minR) / (maxR - minR)
        imageG = (imageG - minG) / (maxG - minG)
        imageB = (imageB - minB) / (maxB - minB)
        
        meanR, meanG, meanB = np.mean(imageR), np.mean(imageG), np.mean(imageB)
        
        if flag == 0:  # Red and Blue compensation
            imageR = (imageR + (meanG - meanR) * (1 - imageR) * imageG) * maxR
            imageB = (imageB + (meanG - meanB) * (1 - imageB) * imageG) * maxB
        else:  # Only Red compensation
            imageR = (imageR + (meanG - meanR) * (1 - imageR) * imageG) * maxR
        
        imageG = imageG * maxG
        imageB = imageB * maxB
        
        compensateIm = np.zeros((image.size[1], image.size[0], 3), dtype="uint8")
        compensateIm[:, :, 0] = imageR
        compensateIm[:, :, 1] = imageG
        compensateIm[:, :, 2] = imageB
        
        return Image.fromarray(compensateIm)
    except Exception as e:
        raise CustomException(f"RB compensation failed: {str(e)}", sys)

def gray_world(image: Image.Image) -> Image.Image:
    """[Your original docstring here]"""
    try:
        imager, imageg, imageb = image.split()
        imagegray = image.convert('L')
        
        imageR = np.array(imager, np.float64)
        imageG = np.array(imageg, np.float64)
        imageB = np.array(imageb, np.float64)
        imageGray = np.array(imagegray, np.float64)
        
        meanR, meanG, meanB = np.mean(imageR), np.mean(imageG), np.mean(imageB)
        meanGray = np.mean(imageGray)
        
        imageR = imageR * meanGray / meanR
        imageG = imageG * meanGray / meanG
        imageB = imageB * meanGray / meanB
        
        whitebalancedIm = np.zeros((image.size[1], image.size[0], 3), dtype="uint8")
        whitebalancedIm[:, :, 0] = imageR
        whitebalancedIm[:, :, 1] = imageG
        whitebalancedIm[:, :, 2] = imageB
        
        return Image.fromarray(whitebalancedIm)
    except Exception as e:
        raise CustomException(f"Gray world balancing failed: {str(e)}", sys)

def sharpen(wbimage: Image.Image, original: Image.Image) -> Image.Image:
    """[Your original docstring here]"""
    try:
        smoothed_image = wbimage.filter(ImageFilter.GaussianBlur(radius=2))
        smoothedr, smoothedg, smoothedb = smoothed_image.split()
        
        imager, imageg, imageb = wbimage.split()
        
        imageR = np.array(imager, np.float64)
        imageG = np.array(imageg, np.float64)
        imageB = np.array(imageb, np.float64)
        smoothedR = np.array(smoothedr, np.float64)
        smoothedG = np.array(smoothedg, np.float64)
        smoothedB = np.array(smoothedb, np.float64)
        
        imageR = 2 * imageR - smoothedR
        imageG = 2 * imageG - smoothedG
        imageB = 2 * imageB - smoothedB
        
        sharpenIm = np.zeros((wbimage.size[1], wbimage.size[0], 3), dtype="uint8")
        sharpenIm[:, :, 0] = imageR
        sharpenIm[:, :, 1] = imageG
        sharpenIm[:, :, 2] = imageB
        
        return Image.fromarray(sharpenIm)
    except Exception as e:
        raise CustomException(f"Sharpening failed: {str(e)}", sys)

def hsv_global_equalization(image: Image.Image) -> Image.Image:
    """[Your original docstring here]"""
    try:
        hsvimage = image.convert('HSV')
        Hue, Saturation, Value = hsvimage.split()
        equalizedValue = ImageOps.equalize(Value)
        
        equalizedIm = np.zeros((image.size[1], image.size[0], 3), dtype="uint8")
        equalizedIm[:, :, 0] = Hue
        equalizedIm[:, :, 1] = Saturation
        equalizedIm[:, :, 2] = equalizedValue
        
        return Image.merge('HSV', [Image.fromarray(channel) for channel in equalizedIm]).convert('RGB')
    except Exception as e:
        raise CustomException(f"HSV equalization failed: {str(e)}", sys)

def average_fusion(image1: Image.Image, image2: Image.Image) -> Image.Image:
    """[Your original docstring here]"""
    try:
        image1r, image1g, image1b = image1.split()
        image2r, image2g, image2b = image2.split()
        
        image1R = np.array(image1r, np.float64)
        image1G = np.array(image1g, np.float64)
        image1B = np.array(image1b, np.float64)
        image2R = np.array(image2r, np.float64)
        image2G = np.array(image2g, np.float64)
        image2B = np.array(image2b, np.float64)
        
        image1R = (image1R + image2R) / 2
        image1G = (image1G + image2G) / 2
        image1B = (image1B + image2B) / 2
        
        fusedIm = np.zeros((image1.size[1], image1.size[0], 3), dtype="uint8")
        fusedIm[:, :, 0] = image1R
        fusedIm[:, :, 1] = image1G
        fusedIm[:, :, 2] = image1B
        
        return Image.fromarray(fusedIm)
    except Exception as e:
        raise CustomException(f"Average fusion failed: {str(e)}", sys)

def pca_fusion(image1: Image.Image, image2: Image.Image) -> Image.Image:
    """[Your original docstring here]"""
    try:
        def get_pca_coeffs(data):
            cov = np.cov(data)
            val, vec = np.linalg.eig(cov)
            return vec[:, np.argmax(val)] / np.sum(vec[:, np.argmax(val)])
        
        image1r, image1g, image1b = image1.split()
        image2r, image2g, image2b = image2.split()
        
        # Process each channel
        channels = []
        for c1, c2 in zip([image1r, image1g, image1b], [image2r, image2g, image2b]):
            arr1 = np.array(c1, np.float64).flatten()
            arr2 = np.array(c2, np.float64).flatten()
            mean1, mean2 = np.mean(arr1), np.mean(arr2)
            stacked = np.vstack((arr1 - mean1, arr2 - mean2))
            coef = get_pca_coeffs(stacked)
            channel = coef[0] * np.array(c1, np.float64) + coef[1] * np.array(c2, np.float64)
            channels.append(channel)
        
        fusedIm = np.zeros((image1.size[1], image1.size[0], 3), dtype="uint8")
        fusedIm[:, :, 0] = channels[0]
        fusedIm[:, :, 1] = channels[1]
        fusedIm[:, :, 2] = channels[2]
        
        return Image.fromarray(fusedIm)
    except Exception as e:
        raise CustomException(f"PCA fusion failed: {str(e)}", sys)

def underwater_image_enhancement(image: Image.Image, flag: int) -> Tuple[Image.Image, Image.Image]:
    """
    Complete underwater image enhancement pipeline
    Args:
        image: Input PIL Image
        flag: 0 for Greenish images, 1 for Bluish images
    Returns:
        Tuple of (PCA fused image, Average fused image)
    Raises:
        CustomException: If any step in the pipeline fails
    """
    try:
        compensated = compensate_RB(image, flag)
        whitebalanced = gray_world(compensated)
        contrastenhanced = hsv_global_equalization(whitebalanced)
        sharpened = sharpen(whitebalanced, image)
        pcafused = pca_fusion(sharpened, contrastenhanced)
        averagefused = average_fusion(sharpened, contrastenhanced)
        return pcafused, averagefused
    except Exception as e:
        raise CustomException(f"Enhancement pipeline failed: {str(e)}", sys)