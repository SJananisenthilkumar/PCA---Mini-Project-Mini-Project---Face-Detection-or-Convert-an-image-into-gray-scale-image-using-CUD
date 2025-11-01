# PCA-Mini-Project---Face-Detection-or-Convert-an-image-into-gray-scale-image-using-CUD
Mini Project - Face Detection or Convert an image into gray scale image using CUDA GPU programming

## REG NO : 212223230086
## NAME : JANANI S

## Convert Image to Grayscale image using CUDA

## AIM:

The aim of this project is to demonstrate how to convert an image to grayscale using CUDA programming without relying on the OpenCV library. It serves as an example of GPU-accelerated image processing using CUDA.

## Procedure:

1.Load the input image using the stb_image library.

2.Allocate memory on the GPU for the input and output image buffers.

3.Copy the input image data from the CPU to the GPU.

4.Define a CUDA kernel function that performs the grayscale conversion on each pixel of the image.

5.Launch the CUDA kernel with appropriate grid and block dimensions.

6.Copy the resulting grayscale image data from the GPU back to the CPU.

7.Save the grayscale image using the stb_image_write library.

8.Clean up allocated memory.
## Program:

```
import cv2
from numba import cuda
import sys
from google.colab.patches import cv2_imshow

# Load the input image
image = cv2.imread('forest.jpg')  # Replace with your image filename
cv2_imshow(image)

# Check if image loaded successfully
if image is None:
    print("Error: Unable to load the input image.")
    sys.exit()

# CUDA kernel for grayscale conversion
@cuda.jit
def gpu_rgb_to_gray(input_image, output_image):
    x, y = cuda.grid(2)
    if x < input_image.shape[0] and y < input_image.shape[1]:
        # Weighted grayscale conversion formula
        r = input_image[x, y, 2] * 0.299
        g = input_image[x, y, 1] * 0.587
        b = input_image[x, y, 0] * 0.114
        gray = r + g + b
        output_image[x, y] = gray

# Allocate memory on GPU
d_input = cuda.to_device(image)
d_output = cuda.device_array((image.shape[0], image.shape[1]), dtype=image.dtype)

# Configure CUDA kernel
threads_per_block = (16, 16)
blocks_per_grid_x = (image.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
blocks_per_grid_y = (image.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

# Launch CUDA kernel
gpu_rgb_to_gray[blocks_per_grid, threads_per_block](d_input, d_output)

# Copy result back to CPU
grayscale_image = d_output.copy_to_host()

# Display and save the output
cv2_imshow(grayscale_image)
cv2.imwrite('grayscale_forest.jpg', grayscale_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
## OUTPUT:

Input Image

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/dbeb3be5-3384-4ba1-a7d7-896c8ae19f30" />


Grayscale Image


<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/d5c63bc2-d07e-4f7a-b18c-dce305fc6e7f" />


## Result:

The CUDA program successfully converts the input image to grayscale using the GPU. The resulting grayscale image is saved as an output file. This example demonstrates the power of GPU parallelism in accelerating image processing tasks.
