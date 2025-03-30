import os
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from brisque import BRISQUE
from sewar.full_ref import mse, rmse, uqi, sam, vifp
import pywt
from skimage.feature import local_binary_pattern
from skimage.filters import frangi, hessian


class FingerVeinQualityAnalyzer:
    def __init__(self, original_path, compressed_paths):
        """
        Initialize the analyzer with original and compressed image paths
        
        Args:
            original_path (str): Path to the original PNG image
            compressed_paths (dict): Dictionary of format {'format': 'path'} 
                                     e.g. {'JPEG': 'path/to/jpeg.jpg'}
        """
        self.original_path = original_path
        self.compressed_paths = compressed_paths
        self.results = []
        
        # Load images
        self.original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
        if self.original is None:
            raise ValueError(f"Could not load original image from {original_path}")
            
        self.compressed = {}
        for fmt, path in compressed_paths.items():
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not load {fmt} image from {path}")
            # Resize compressed to match original if needed
            if img.shape != self.original.shape:
                img = cv2.resize(img, (self.original.shape[1], self.original.shape[0]))
            self.compressed[fmt] = img
    
    def analyze_all(self):
        """Run all quality analysis metrics"""
        for fmt, compressed_img in self.compressed.items():
            result = {
                'format': fmt,
                'file_size_original': os.path.getsize(self.original_path),
                'file_size_compressed': os.path.getsize(self.compressed_paths[fmt])
            }
            
            # Standard full-reference metrics
            result.update(self.calculate_full_reference_metrics(compressed_img))
            
            # No-reference/blind metrics
            result.update(self.calculate_blind_metrics(compressed_img))
            
            # Vascular-specific metrics
            result.update(self.calculate_vascular_metrics(compressed_img))
            
            self.results.append(result)
        
        return pd.DataFrame(self.results)
    
    def calculate_full_reference_metrics(self, compressed_img):
        """Calculate standard full-reference quality metrics"""
        metrics = {}
        
        # PSNR
        metrics['psnr'] = psnr(self.original, compressed_img, data_range=255)
        
        # SSIM
        metrics['ssim'] = ssim(self.original, compressed_img, data_range=255)
        
        # MSE
        metrics['mse'] = mse(self.original, compressed_img)
        
        # RMSE
        metrics['rmse'] = rmse(self.original, compressed_img)
        
        # Universal Quality Index
        metrics['uqi'] = uqi(self.original, compressed_img)
        
        # Spectral Angle Mapper
        metrics['sam'] = sam(self.original, compressed_img)
        
        # Visual Information Fidelity
        metrics['vifp'] = vifp(self.original, compressed_img)
        
        return metrics
    
    def calculate_blind_metrics(self, compressed_img):
        """Calculate no-reference/blind quality metrics"""
        metrics = {}
        
        # BRISQUE requires RGB input, so convert grayscale to RGB
        if len(compressed_img.shape) == 2:  # Grayscale image
            compressed_img_rgb = cv2.cvtColor(compressed_img, cv2.COLOR_GRAY2RGB)
        else:
            compressed_img_rgb = compressed_img
        
        # BRISQUE - Blind/Referenceless Image Spatial Quality Evaluator
        brisq = BRISQUE()
        metrics['brisque'] = brisq.score(compressed_img_rgb)  # Pass RGB image
        
        # Variance of Laplacian (works on grayscale)
        metrics['laplacian_var'] = cv2.Laplacian(compressed_img, cv2.CV_64F).var()
        
        return metrics
    
    def calculate_vascular_metrics(self, compressed_img):
        """Calculate vascular-specific quality metrics"""
        metrics = {}
        
        # Vein contrast preservation
        metrics['vein_contrast'] = self.vein_contrast_preservation(compressed_img)
        
        # Vein structure similarity
        metrics['vein_ssim'] = self.vein_structure_similarity(compressed_img)
        
        # Frangi filter response similarity
        metrics['frangi_similarity'] = self.frangi_filter_similarity(compressed_img)
        
        # Local Binary Pattern histogram similarity
        metrics['lbp_similarity'] = self.lbp_histogram_similarity(compressed_img)
        
        # Wavelet energy preservation
        metrics['wavelet_energy'] = self.wavelet_energy_preservation(compressed_img)
        
        return metrics
    
    def vein_contrast_preservation(self, compressed_img):
        """
        Measure how well vein contrast is preserved in compressed image
        """
        # Apply adaptive thresholding to both images
        orig_thresh = cv2.adaptiveThreshold(
            self.original, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        comp_thresh = cv2.adaptiveThreshold(
            compressed_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Calculate contrast in vein regions
        orig_vein_contrast = np.mean(self.original[orig_thresh == 0])
        orig_bg_contrast = np.mean(self.original[orig_thresh == 255])
        orig_contrast = orig_bg_contrast - orig_vein_contrast
        
        comp_vein_contrast = np.mean(compressed_img[comp_thresh == 0])
        comp_bg_contrast = np.mean(compressed_img[comp_thresh == 255])
        comp_contrast = comp_bg_contrast - comp_vein_contrast
        
        return comp_contrast / orig_contrast
    
    def vein_structure_similarity(self, compressed_img):
        """
        Calculate SSIM specifically in vein regions
        """
        # Create vein mask using adaptive thresholding
        vein_mask = cv2.adaptiveThreshold(
            self.original, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Calculate SSIM only in vein regions
        return ssim(
            self.original, compressed_img, 
            data_range=255, 
            win_size=7,
            mask=vein_mask
        )
    
    def frangi_filter_similarity(self, compressed_img):
        """
        Compare Frangi filter responses which enhance tubular structures
        """
        # Calculate Frangi filter responses
        orig_frangi = frangi(self.original, scale_range=(1, 5), scale_step=1)
        comp_frangi = frangi(compressed_img, scale_range=(1, 5), scale_step=1)
        
        # Normalize
        orig_frangi = (orig_frangi - orig_frangi.min()) / (orig_frangi.max() - orig_frangi.min())
        comp_frangi = (comp_frangi - comp_frangi.min()) / (comp_frangi.max() - comp_frangi.min())
        
        return ssim(orig_frangi, comp_frangi, data_range=1.0)
    
    def lbp_histogram_similarity(self, compressed_img):
        """
        Compare Local Binary Pattern histograms which capture texture patterns
        """
        # Parameters for LBP
        radius = 3
        n_points = 8 * radius
        n_bins = n_points + 2
        
        # Calculate LBP
        lbp_orig = local_binary_pattern(self.original, n_points, radius, method='uniform')
        lbp_comp = local_binary_pattern(compressed_img, n_points, radius, method='uniform')
        
        # Calculate histograms
        hist_orig, _ = np.histogram(lbp_orig, bins=n_bins, range=(0, n_bins))
        hist_comp, _ = np.histogram(lbp_comp, bins=n_bins, range=(0, n_bins))
        
        # Normalize histograms
        hist_orig = hist_orig.astype(float) / hist_orig.sum()
        hist_comp = hist_comp.astype(float) / hist_comp.sum()
        
        # Calculate histogram intersection
        return np.minimum(hist_orig, hist_comp).sum()
    
    def wavelet_energy_preservation(self, compressed_img):
        """
        Compare wavelet energy distribution which is important for vein patterns
        """
        # Wavelet decomposition
        wavelet = 'db1'
        level = 3
        
        # Original image decomposition
        coeffs_orig = pywt.wavedec2(self.original, wavelet, level=level)
        _, (h_orig, v_orig, d_orig) = coeffs_orig[0], coeffs_orig[1]
        
        # Compressed image decomposition
        coeffs_comp = pywt.wavedec2(compressed_img, wavelet, level=level)
        _, (h_comp, v_comp, d_comp) = coeffs_comp[0], coeffs_comp[1]
        
        # Calculate energy in detail coefficients
        energy_orig = np.sum(h_orig**2) + np.sum(v_orig**2) + np.sum(d_orig**2)
        energy_comp = np.sum(h_comp**2) + np.sum(v_comp**2) + np.sum(d_comp**2)
        
        return energy_comp / energy_orig
    

def main():
    # Example usage
    original_image = "input.png"
    compressed_images = {
        "JPEG": "output/comp.jpg",
        "JPEG2000": "output/comp.jp2",
        #"JPEGXR": "output/comp.jxr",
        #"JPEGXL": "output/comp.jxl"
    }
    
    analyzer = FingerVeinQualityAnalyzer(original_image, compressed_images)
    results_df = analyzer.analyze_all()
    
    # Save results to CSV
    results_df.to_csv("compression_quality_analysis.csv", index=False)
    print("Analysis complete. Results saved to compression_quality_analysis.csv")
    
    # Print summary
    print("\nQuality Analysis Summary:")
    print(results_df)

if __name__ == "__main__":
    main()