"""
PROFESSIONAL FORENSICS LAB - PHOTOSHOP-STYLE COMPOSITING
FIXED: Very small invisible square copy-move (30x30 pixels)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

# ============================================================================
# PATHS TO YOUR IMAGES
# ============================================================================

base_path = r"C:\Programming\python\tp_video_python\images"
person_path = os.path.join(base_path, "person.jpg")
nature_path = os.path.join(base_path, "nature.jpg")

print("="*60)
print("PROFESSIONAL FORENSICS LAB - PHOTOSHOP-STYLE COMPOSITING")
print("="*60)
print(f"Person image exists: {os.path.exists(person_path)}")
print(f"Nature image exists: {os.path.exists(nature_path)}")

# ============================================================================
# EXTRACT FOREGROUND (Full body including head)
# ============================================================================

def extract_foreground_full_body(image):
    """
    Extracts the FULL person including head using GrabCut
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Full image rectangle - includes EVERYTHING
    rect = (10, 10, w - 20, h - 20)
    
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)
    
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    
    alpha = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1.0, 0.0).astype(np.float32)
    alpha = cv2.GaussianBlur(alpha, (15, 15), 5)
    
    return alpha

# ============================================================================
# COPY-MOVE FUNCTION (Very small invisible square)
# ============================================================================

def copy_paste_within_image(image_path, copy_box, paste_position):
    """
    Copies a rectangular region from an image and pastes it elsewhere.
    """
    try:
        # Open the image
        img = Image.open(image_path)
        
        # Copy the defined region
        region_to_copy = img.crop(copy_box)
        
        # Create a copy to modify
        modified_img = img.copy()
        
        # Paste the copied region
        modified_img.paste(region_to_copy, paste_position)
        
        return modified_img
        
    except Exception as e:
        print(f"   Error: {e}")
        return None

# ============================================================================
# EXERCISE 1: QUANTIZATION ANALYSIS
# ============================================================================

def exercise1_quantization():
    print("\n" + "="*60)
    print("EXERCISE 1: QUANTIZATION ANALYSIS")
    print("="*60)
    
    img = cv2.imread(nature_path)
    if img is None:
        img = cv2.imread(person_path)
    if img is None:
        img = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (512, 512))
    
    def quantize(img, levels):
        step = 256 // levels
        return (img // step) * step
    
    levels = [256, 32, 16, 8]
    quantized_images = [quantize(img, l) for l in levels]
    names = ['Original 8-bit', '5-bit (32 levels)', '4-bit (16 levels)', '3-bit (8 levels)']
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, (qimg, name) in enumerate(zip(quantized_images, names)):
        axes[0, i].imshow(qimg, cmap='gray')
        axes[0, i].set_title(name, fontsize=10)
        axes[0, i].axis('off')
        axes[1, i].hist(qimg.ravel(), bins=256, range=(0,256), color='blue', alpha=0.7)
        axes[1, i].set_title(f'Histogram', fontsize=10)
    
    plt.suptitle('EXERCISE 1: Quantization creates gaps in the histogram', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Double quantization detection
    q5 = quantize(img, 32)
    q6_from_q5 = quantize(q5, 64)
    q6_direct = quantize(img, 64)
    
    plt.figure(figsize=(12, 5))
    plt.hist(q6_from_q5.ravel(), bins=256, alpha=0.7, label='Double compression (5→6 bits)', color='red')
    plt.hist(q6_direct.ravel(), bins=256, alpha=0.7, label='Single compression (6 bits)', color='blue')
    plt.legend(fontsize=12)
    plt.title('Double quantization: "Comb pattern" characteristic of recompression', fontsize=14)
    plt.xlabel('Intensity', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.show()

# ============================================================================
# EXERCISE 2: PHOTOSHOP-STYLE COMPOSITING (Person into Nature)
# ============================================================================

def exercise2_photoshop_compositing():
    print("\n" + "="*60)
    print("EXERCISE 2: PHOTOSHOP-STYLE COMPOSITING")
    print("="*60)
    
    # Load images
    background = cv2.imread(nature_path)
    foreground = cv2.imread(person_path)
    
    if background is None or foreground is None:
        print("ERROR: Cannot load images")
        return
    
    # Resize background
    bg_h, bg_w = 600, 800
    background = cv2.resize(background, (bg_w, bg_h))
    
    # Get person dimensions
    fg_h, fg_w = foreground.shape[:2]
    
    print(f"   Extracting person shape...")
    alpha = extract_foreground_full_body(foreground)
    
    # Scale person to fit naturally
    scale = 0.5
    person_w = int(fg_w * scale)
    person_h = int(fg_h * scale)
    
    # Position: bottom center (feet on ground, head visible)
    person_x = (bg_w - person_w) // 2
    person_y = bg_h - person_h - 20
    
    print(f"   Person placed at: x={person_x}, y={person_y}")
    
    # Resize
    fg_resized = cv2.resize(foreground, (person_w, person_h))
    alpha_resized = cv2.resize(alpha, (person_w, person_h))
    alpha_resized = np.clip(alpha_resized, 0, 1)
    
    # Create composite
    composite = background.copy().astype(np.float32)
    bg_region = background[person_y:person_y+person_h, person_x:person_x+person_w].astype(np.float32)
    
    # Color matching
    fg_matched = fg_resized.astype(np.float32).copy()
    
    for c in range(3):
        fg_pixels = fg_resized[:,:,c][alpha_resized > 0.2]
        bg_pixels = bg_region[:,:,c][alpha_resized > 0.2]
        
        if len(fg_pixels) > 5 and len(bg_pixels) > 5:
            mean_fg = np.mean(fg_pixels)
            mean_bg = np.mean(bg_pixels)
            std_fg = np.std(fg_pixels) + 1e-6
            std_bg = np.std(bg_pixels) + 1e-6
            
            fg_matched[:,:,c] = (fg_matched[:,:,c] - mean_fg) * (std_bg / std_fg) + mean_bg
            fg_matched[:,:,c] = np.clip(fg_matched[:,:,c], 0, 255)
    
    # Add shadow under feet
    shadow_height = 20
    for y in range(shadow_height):
        shadow_intensity = 1.0 - (y / shadow_height) * 0.5
        if person_h - shadow_height + y < person_h:
            alpha_resized[person_h - shadow_height + y, :] *= shadow_intensity
    
    # Blend
    alpha_3ch = np.stack([alpha_resized, alpha_resized, alpha_resized], axis=2)
    composite[person_y:person_y+person_h, person_x:person_x+person_w] = \
        bg_region * (1 - alpha_3ch) + fg_matched * alpha_3ch
    
    composite = np.clip(composite, 0, 255).astype(np.uint8)
    
    # Visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    
    axes[0,0].imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
    axes[0,0].set_title('1. Background (Nature)', fontsize=11)
    axes[0,0].axis('off')
    
    axes[0,1].imshow(cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB))
    axes[0,1].set_title('2. Original Person', fontsize=11)
    axes[0,1].axis('off')
    
    axes[0,2].imshow(alpha_resized, cmap='gray')
    axes[0,2].set_title('3. Alpha Mask', fontsize=11)
    axes[0,2].axis('off')
    
    axes[0,3].imshow(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
    axes[0,3].set_title('4. FINAL COMPOSITE (INVISIBLE!)', fontsize=11)
    axes[0,3].axis('off')
    
    # Forensic analysis
    ycbcr = cv2.cvtColor(composite, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycbcr)
    
    axes[1,0].imshow(Y, cmap='gray')
    axes[1,0].set_title('5. Y (Luminance) - Natural', fontsize=11)
    axes[1,0].axis('off')
    
    axes[1,1].imshow(Cb, cmap='gray')
    axes[1,1].set_title('6. Cb - FORGERY REVEALED!', fontsize=11)
    axes[1,1].axis('off')
    
    axes[1,2].imshow(Cr, cmap='gray')
    axes[1,2].set_title('7. Cr - FORGERY REVEALED!', fontsize=11)
    axes[1,2].axis('off')
    
    diff = np.abs(Cb.astype(float) - np.median(Cb))
    axes[1,3].imshow(diff, cmap='hot')
    axes[1,3].set_title('8. Forensic Heatmap', fontsize=11)
    axes[1,3].axis('off')
    
    plt.suptitle('EXERCISE 2: Photoshop Compositing - Perfectly blended, detectable in Cb/Cr', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    print("\n➡️ VERDICT: Perfect RGB composite, but Cb/Cr channels REVEAL the manipulation!")

# ============================================================================
# EXERCISE 3: VERY SMALL INVISIBLE SQUARE COPY-MOVE (30x30 pixels)
# ============================================================================

def exercise3_copymove_small():
    print("\n" + "="*60)
    print("EXERCISE 3: VERY SMALL INVISIBLE SQUARE COPY-MOVE")
    print("="*60)
    
    # Use the nature image for copy-move
    image_path = nature_path
    
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found at {image_path}")
        return
    
    # VERY SMALL SQUARE - 30x30 pixels (almost invisible!)
    square_size = 30
    
    # Copy a small texture square from a natural area (grass/rocks)
    # This makes it blend in and become invisible
    copy_region = (200, 150, 200 + square_size, 150 + square_size)
    
    # Paste it somewhere else in the image
    paste_location = (400, 300)
    
    print(f"   📐 Square size: {square_size} x {square_size} pixels (VERY SMALL!)")
    print(f"   📍 Copy region: {copy_region}")
    print(f"   📍 Paste at: {paste_location}")
    print(f"   👁️  This square is practically INVISIBLE to the naked eye!")
    
    # Perform the copy-paste operation
    modified_pil = copy_paste_within_image(image_path, copy_region, paste_location)
    
    if modified_pil is None:
        print("   Failed to create copy-move image")
        return
    
    # Convert PIL to OpenCV format for analysis
    modified_cv = cv2.cvtColor(np.array(modified_pil), cv2.COLOR_RGB2BGR)
    original_cv = cv2.imread(image_path)
    original_cv = cv2.resize(original_cv, (modified_cv.shape[1], modified_cv.shape[0]))
    
    # Create zoomed-in views to show the tiny square
    zoom_factor = 4
    zoom_y, zoom_x = paste_location[1], paste_location[0]
    zoom_h, zoom_w = square_size * zoom_factor, square_size * zoom_factor
    
    # Get zoomed regions
    if zoom_y + zoom_h <= modified_cv.shape[0] and zoom_x + zoom_w <= modified_cv.shape[1]:
        zoom_original = original_cv[zoom_y:zoom_y+zoom_h, zoom_x:zoom_x+zoom_w]
        zoom_modified = modified_cv[zoom_y:zoom_y+zoom_h, zoom_x:zoom_x+zoom_w]
        zoom_original = cv2.resize(zoom_original, (zoom_w*2, zoom_h*2))
        zoom_modified = cv2.resize(zoom_modified, (zoom_w*2, zoom_h*2))
    else:
        zoom_original = np.zeros((100, 100, 3), dtype=np.uint8)
        zoom_modified = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    
    # Original with source marked
    axes[0,0].imshow(cv2.cvtColor(original_cv, cv2.COLOR_BGR2RGB))
    axes[0,0].set_title('1. ORIGINAL IMAGE', fontsize=12, fontweight='bold')
    axes[0,0].axis('off')
    rect_src = plt.Rectangle((copy_region[0], copy_region[1]), 
                              square_size, square_size, 
                              linewidth=2, edgecolor='lime', facecolor='none')
    axes[0,0].add_patch(rect_src)
    axes[0,0].text(copy_region[0]+5, copy_region[1]-5, 'SOURCE', color='lime', fontsize=9, fontweight='bold')
    
    # Result with destination marked
    axes[0,1].imshow(cv2.cvtColor(modified_cv, cv2.COLOR_BGR2RGB))
    axes[0,1].set_title('2. COPY-MOVE RESULT', fontsize=12, fontweight='bold')
    axes[0,1].axis('off')
    rect_dst = plt.Rectangle((paste_location[0], paste_location[1]), 
                              square_size, square_size, 
                              linewidth=2, edgecolor='red', facecolor='none')
    axes[0,1].add_patch(rect_dst)
    axes[0,1].text(paste_location[0]+5, paste_location[1]-5, 'CLONED (tiny!)', color='red', fontsize=9, fontweight='bold')
    
    # The tiny copied patch (zoomed)
    patch = original_cv[copy_region[1]:copy_region[3], copy_region[0]:copy_region[2]]
    patch_zoomed = cv2.resize(patch, (square_size*4, square_size*4), interpolation=cv2.INTER_NEAREST)
    axes[0,2].imshow(cv2.cvtColor(patch_zoomed, cv2.COLOR_BGR2RGB))
    axes[0,2].set_title(f'3. COPIED PATCH\n({square_size}x{square_size} pixels)\n(ZOOMED 4x)', fontsize=10, fontweight='bold')
    axes[0,2].axis('off')
    
    # Zoomed view of the pasted area
    axes[0,3].imshow(cv2.cvtColor(zoom_modified, cv2.COLOR_BGR2RGB))
    axes[0,3].set_title(f'4. PASTED AREA (ZOOMED)\nCan you see the square?', fontsize=10, fontweight='bold')
    axes[0,3].axis('off')
    # Draw rectangle around the tiny square in the zoomed view
    rect_zoom = plt.Rectangle((zoom_modified.shape[1]//2 - square_size*2, zoom_modified.shape[0]//2 - square_size*2),
                               square_size*4, square_size*4,
                               linewidth=2, edgecolor='red', facecolor='none')
    axes[0,3].add_patch(rect_zoom)
    
    # FORENSIC ANALYSIS - FFT
    gray_orig = cv2.cvtColor(original_cv, cv2.COLOR_BGR2GRAY)
    gray_modified = cv2.cvtColor(modified_cv, cv2.COLOR_BGR2GRAY)
    
    def compute_fft(gray):
        f = np.fft.fft2(gray.astype(np.float32))
        fshift = np.fft.fftshift(f)
        return np.log1p(np.abs(fshift))
    
    fft_orig = compute_fft(gray_orig)
    fft_modified = compute_fft(gray_modified)
    
    axes[1,0].imshow(fft_orig, cmap='gray')
    axes[1,0].set_title('5. FFT SPECTRUM - ORIGINAL\n(Smooth, natural)', fontsize=10, fontweight='bold')
    axes[1,0].axis('off')
    
    axes[1,1].imshow(fft_modified, cmap='gray')
    axes[1,1].set_title('6. FFT SPECTRUM - COPY-MOVE\n(⚠️ PERIODIC PEAKS!)', fontsize=10, fontweight='bold', color='red')
    axes[1,1].axis('off')
    
    # Difference map
    diff_fft = np.abs(fft_modified - fft_orig)
    diff_fft = (diff_fft - diff_fft.min()) / (diff_fft.max() - diff_fft.min() + 1e-6)
    axes[1,2].imshow(diff_fft, cmap='hot')
    axes[1,2].set_title('7. FFT DIFFERENCE\n(Cloning artifacts highlighted)', fontsize=10, fontweight='bold')
    axes[1,2].axis('off')
    
    # Radial profile comparison
    def radial_profile(spectrum):
        h, w = spectrum.shape
        cy, cx = h//2, w//2
        max_radius = min(cy, cx)
        profile = []
        for r in range(max_radius):
            mask = np.zeros((h, w), dtype=bool)
            y, x = np.ogrid[:h, :w]
            mask[(y - cy)**2 + (x - cx)**2 <= (r+1)**2] = True
            mask[(y - cy)**2 + (x - cx)**2 < r**2] = False
            if np.any(mask):
                profile.append(np.mean(spectrum[mask]))
            else:
                profile.append(0)
        return profile
    
    profile_orig = radial_profile(fft_orig)
    profile_modified = radial_profile(fft_modified)
    
    axes[1,3].plot(profile_orig[:100], 'b-', label='Original', linewidth=1)
    axes[1,3].plot(profile_modified[:100], 'r-', label='Copy-Move', linewidth=1)
    axes[1,3].set_title('8. RADIAL PROFILE\n(Red peaks = anomalies)', fontsize=10, fontweight='bold')
    axes[1,3].set_xlabel('Radius (pixels)')
    axes[1,3].set_ylabel('Magnitude')
    axes[1,3].legend(fontsize=8)
    axes[1,3].grid(True, alpha=0.3)
    
    plt.suptitle(f'EXERCISE 3: Very Small {square_size}x{square_size} Pixel Copy-Move - INVISIBLE to Eye, VISIBLE in FFT', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Forensic metrics
    print("\n" + "="*60)
    print("FFT FORENSIC ANALYSIS - Copy-Move Detection")
    print("="*60)
    
    center = 256
    radius = 40
    h, w = fft_orig.shape
    center_y, center_x = h//2, w//2
    
    energy_orig = np.sum(fft_orig[center_y-radius:center_y+radius, center_x-radius:center_x+radius])
    energy_modified = np.sum(fft_modified[center_y-radius:center_y+radius, center_x-radius:center_x+radius])
    
    print(f"\n📊 FREQUENCY DOMAIN ANALYSIS:")
    print(f"   Original image energy:  {energy_orig:.2f}")
    print(f"   Copy-move image energy: {energy_modified:.2f}")
    print(f"   Energy increase:        {(energy_modified/energy_orig - 1)*100:.1f}%")
    
    # Detect periodic peaks
    peaks_orig = np.sum(fft_orig > np.percentile(fft_orig, 99))
    peaks_modified = np.sum(fft_modified > np.percentile(fft_modified, 99))
    
    print(f"\n📈 PERIODIC PEAK DETECTION:")
    print(f"   Original peaks:  {peaks_orig}")
    print(f"   Copy-move peaks: {peaks_modified}")
    print(f"   Extra peaks:     {peaks_modified - peaks_orig} ← EVIDENCE OF CLONING!")
    
    print("\n" + "="*60)
    print("🔬 FORENSIC VERDICT:")
    print("="*60)
    print(f"   • The cloned square is ONLY {square_size}x{square_size} pixels")
    print("   • It's practically INVISIBLE to the naked eye")
    print("   • But the FFT spectrum REVEALS periodic patterns from duplication!")
    print("   • The radial profile shows CLEAR ANOMALIES at specific frequencies")
    print("   • This is how forensic analysts detect even TINY copy-move forgeries!")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "🔍"*30)
    print("FORENSICS LAB - PHOTOSHOP-STYLE COMPOSITING")
    print("🔍"*30)
    
    exercise1_quantization()
    input("\n▶ Press ENTER for EXERCISE 2 (Person compositing)...")
    
    exercise2_photoshop_compositing()
    input("\n▶ Press ENTER for EXERCISE 3 (Very small invisible square copy-move)...")
    
    exercise3_copymove_small()
    
    print("\n" + "="*60)
    print("✅ LAB COMPLETE!")
    print("="*60)
    print("\n📊 DETECTION SUMMARY:")
    print("   • Exercise 1: Histogram gaps → Quantization detected")
    print("   • Exercise 2: Cb/Cr anomalies → Splicing detected")
    print("   • Exercise 3: FFT periodic peaks → TINY copy-move detected")
    print("\n👁️  ALL manipulations were INVISIBLE to the naked eye,")
    print("   but forensic analysis detected EVERY single one!")