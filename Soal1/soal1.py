import cv2
from PIL import Image
import numpy as np
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def rgb_to_ycbcr(image_rgb):
    r, g, b = image_rgb[:,:,0], image_rgb[:,:,1], image_rgb[:,:,2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128
    ycbcr_output = np.stack([y, cb, cr], axis=-1)
    ycbcr_output = np.clip(ycbcr_output, 0, 255).astype(np.uint8)
    return ycbcr_output

def ycbcr_to_rgb(image_ycbcr):
    y, cb, cr = image_ycbcr[:,:,0].astype(np.float32), image_ycbcr[:,:,1].astype(np.float32), image_ycbcr[:,:,2].astype(np.float32)

    cb = cb - 128
    cr = cr - 128

    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb

    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb

def dct2d(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

def idct2d(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')

def get_quantization_matrix(quality_factor=50):
    quantization_matrix = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])

    if quality_factor < 50:
        s = 5000 / quality_factor
    else:
        s = 200 - 2 * quality_factor

    quant_matrix_scaled = np.floor((quantization_matrix * s + 50) / 100)
    quant_matrix_scaled[quant_matrix_scaled == 0] = 1
    return quant_matrix_scaled.astype(np.int32) 

def quantize(block, quantization_matrix_scaled):
    quant_block = np.round(block / quantization_matrix_scaled)
    return quant_block

def dequantize(quant_block, quantization_matrix_scaled):
    dequant_block = quant_block * quantization_matrix_scaled
    return dequant_block


if __name__ == "__main__":
    image_path = 'selfie.jpg'
    block_size = 8
    quality = 50

    try:
        img_pil = Image.open(image_path).convert('RGB')
        img_np_rgb = np.array(img_pil)

        img_ycbcr = rgb_to_ycbcr(img_np_rgb)

        height, width, _ = img_ycbcr.shape

        start_row = (height // 2) - (block_size // 2)
        start_col = (width // 2) - (block_size // 2)

        start_row = max(0, min(start_row, height - block_size))
        start_col = max(0, min(start_col, width - block_size))

        original_macroblock_rgb = img_np_rgb[start_row : start_row + block_size,
                                             start_col : start_col + block_size]
        original_macroblock_ycbcr = img_ycbcr[start_row : start_row + block_size,
                                              start_col : start_col + block_size]

        original_macroblock_y = img_ycbcr[start_row : start_row + block_size,
                                          start_col : start_col + block_size, 0].astype(np.float32)
        original_macroblock_cb = img_ycbcr[start_row : start_row + block_size,
                                           start_col : start_col + block_size, 1].astype(np.float32)
        original_macroblock_cr = img_ycbcr[start_row : start_row + block_size,
                                           start_col : start_col + block_size, 2].astype(np.float32)

        quant_matrix_scaled = get_quantization_matrix(quality)

        # Encoding
        dct_macroblock_y = dct2d(original_macroblock_y - 128)
        quantized_macroblock_y = quantize(dct_macroblock_y, quant_matrix_scaled)

        dct_macroblock_cb = dct2d(original_macroblock_cb - 128)
        quantized_macroblock_cb = quantize(dct_macroblock_cb, quant_matrix_scaled)

        dct_macroblock_cr = dct2d(original_macroblock_cr - 128)
        quantized_macroblock_cr = quantize(dct_macroblock_cr, quant_matrix_scaled)

        # Decoding
        dequantized_macroblock_y = dequantize(quantized_macroblock_y, quant_matrix_scaled)
        decoded_macroblock_y = idct2d(dequantized_macroblock_y) + 128
        decoded_macroblock_y = np.clip(decoded_macroblock_y, 0, 255).astype(np.uint8)

        dequantized_macroblock_cb = dequantize(quantized_macroblock_cb, quant_matrix_scaled)
        decoded_macroblock_cb = idct2d(dequantized_macroblock_cb) + 128
        decoded_macroblock_cb = np.clip(decoded_macroblock_cb, 0, 255).astype(np.uint8)

        dequantized_macroblock_cr = dequantize(quantized_macroblock_cr, quant_matrix_scaled)
        decoded_macroblock_cr = idct2d(dequantized_macroblock_cr) + 128
        decoded_macroblock_cr = np.clip(decoded_macroblock_cr, 0, 255).astype(np.uint8)

        decoded_ycbcr_macroblock = np.stack([decoded_macroblock_y, decoded_macroblock_cb, decoded_macroblock_cr], axis=-1)
        decoded_rgb_macroblock = ycbcr_to_rgb(decoded_ycbcr_macroblock)

        # error
        error_macroblock = np.abs(original_macroblock_rgb.astype(np.float32) - decoded_rgb_macroblock.astype(np.float32)).mean(axis=2)

        # Visualisasi
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))# Baris 2, Kolom 4

        # Makroblok Asli (RGB)
        axes[0, 0].imshow(original_macroblock_rgb)
        axes[0, 0].set_title('Makroblok Asli')
        axes[0, 0].axis('off')

        # Makroblok Rekonstruksi (RGB)
        axes[0, 1].imshow(decoded_rgb_macroblock)
        axes[0, 1].set_title('Makroblok Rekonstruksi')
        axes[0, 1].axis('off')

        # (Error)
        norm_error = Normalize(vmin=0, vmax=np.max(error_macroblock))
        im_error = axes[0, 2].imshow(error_macroblock, cmap='hot', norm=norm_error)
        axes[0, 2].set_title('Perbedaan (Error)')
        axes[0, 2].set_xticks(np.arange(block_size))
        axes[0, 2].set_yticks(np.arange(block_size))
        fig.colorbar(im_error, ax=axes[0, 2], fraction=0.046, pad=0.04)

        # Tabel Kuantisasi
        im_quant = axes[0, 3].imshow(quant_matrix_scaled, cmap='viridis')
        axes[0, 3].set_title('Tabel Kuantisasi')
        axes[0, 3].set_xticks(np.arange(block_size))
        axes[0, 3].set_yticks(np.arange(block_size))
        for i in range(block_size):
            for j in range(block_size):
                axes[0, 3].text(j, i, str(quant_matrix_scaled[i, j]),
                                ha='center', va='center', color='white', fontsize=8)
        fig.colorbar(im_quant, ax=axes[0, 3], fraction=0.046, pad=0.04)

        # DCT Blok 1 (Kanal Y)
        im_dct_y = axes[1, 0].imshow(np.log(np.abs(dct_macroblock_y) + 1), cmap='cividis')
        axes[1, 0].set_title('DCT (Y)')
        axes[1, 0].set_xticks(np.arange(block_size))
        axes[1, 0].set_yticks(np.arange(block_size))
        fig.colorbar(im_dct_y, ax=axes[1, 0], fraction=0.046, pad=0.04)

        # DCT Blok 2 (Kanal Cb)
        im_dct_cb = axes[1, 1].imshow(np.log(np.abs(dct_macroblock_cb) + 1), cmap='cividis')
        axes[1, 1].set_title('DCT (Cb)')
        axes[1, 1].set_xticks(np.arange(block_size))
        axes[1, 1].set_yticks(np.arange(block_size))
        fig.colorbar(im_dct_cb, ax=axes[1, 1], fraction=0.046, pad=0.04)

        # Quantized Blok 1 (Kanal Y)
        im_quant_y = axes[1, 2].imshow(np.log(np.abs(quantized_macroblock_y) + 1), cmap='cividis')
        axes[1, 2].set_title(f'Kuantisasi (Y)')
        axes[1, 2].set_xticks(np.arange(block_size))
        axes[1, 2].set_yticks(np.arange(block_size))
        fig.colorbar(im_quant_y, ax=axes[1, 2], fraction=0.046, pad=0.04)

        # Quantized Blok 2 (Kanal Cb)
        im_quant_cb = axes[1, 3].imshow(np.log(np.abs(quantized_macroblock_cb) + 1), cmap='cividis')
        axes[1, 3].set_title(f'Kuantisasi (Cb)')
        axes[1, 3].set_xticks(np.arange(block_size))
        axes[1, 3].set_yticks(np.arange(block_size))
        fig.colorbar(im_quant_cb, ax=axes[1, 3], fraction=0.046, pad=0.04)

        plt.suptitle(f'Visualisasi Kompresi JPEG pada Makroblok {block_size}x{block_size} piksel - Dengan kualitas {quality}', fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        # Perbandingan Nilai RGB untuk masing-masing blok
        fig_rgb_values, axes_rgb_values = plt.subplots(1, 2, figsize=(16, 8))
        axes_rgb_values[0].imshow(original_macroblock_rgb)
        axes_rgb_values[0].set_title('Makroblok Asli')
        axes_rgb_values[0].axis('off')
        for i in range(block_size):
            for j in range(block_size):
                r, g, b = original_macroblock_rgb[i, j]
                text_color = 'black'
                axes_rgb_values[0].text(j, i, f'[{r},{g},{b}]',
                             ha='center', va='center', color=text_color, fontsize=6)
        axes_rgb_values[1].imshow(decoded_rgb_macroblock)
        axes_rgb_values[1].set_title(f'Makroblok Hasil Decoding')
        axes_rgb_values[1].axis('off')
        for i in range(block_size):
            for j in range(block_size):
                r, g, b = decoded_rgb_macroblock[i, j]
                text_color = 'black' 
                axes_rgb_values[1].text(j, i, f'[{r},{g},{b}]',
                             ha='center', va='center', color=text_color, fontsize=6)
        plt.suptitle(f'Perbandingan Nilai RGB Makroblok Asli vs. Hasil Decoding')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        # Perbandingan Nilai YCbCr untuk masing-masing blok
        fig_ycbcr_values, axes_ycbcr_values = plt.subplots(1, 2, figsize=(16, 8))
        axes_ycbcr_values[0].imshow(ycbcr_to_rgb(original_macroblock_ycbcr))
        axes_ycbcr_values[0].set_title('Makroblok Asli')
        axes_ycbcr_values[0].axis('off')
        for i in range(block_size):
            for j in range(block_size):
                y, cb, cr = original_macroblock_ycbcr[i, j]
                text_color = 'black' 
                axes_ycbcr_values[0].text(j, i, f'Y:{y}\nCb:{cb}\nCr:{cr}',
                             ha='center', va='center', color=text_color, fontsize=5)
        axes_ycbcr_values[1].imshow(ycbcr_to_rgb(decoded_ycbcr_macroblock))
        axes_ycbcr_values[1].set_title(f'Makroblok Hasil Decoding')
        axes_ycbcr_values[1].axis('off')
        for i in range(block_size):
            for j in range(block_size):
                y, cb, cr = decoded_ycbcr_macroblock[i, j]
                text_color = 'black'
                axes_ycbcr_values[1].text(j, i, f'Y:{y}\nCb:{cb}\nCr:{cr}',
                             ha='center', va='center', color=text_color, fontsize=5)
        plt.suptitle(f'Perbandingan Nilai YCbCe Makroblok Asli vs. Hasil Decoding')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    except FileNotFoundError:
        print(f"Error: File '{image_path}' tidak ditemukan. Pastikan file selfie.jpg berada di direktori yang sama.")
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")