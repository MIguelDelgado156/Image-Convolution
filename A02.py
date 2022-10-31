from tokenize import Double
import numpy as np
import cv2
import sys
from pathlib import Path

def applyFilter(image,kernel):
    image = image.astype("float64")
    kernel = kernel.astype("float64")
 
    kernel = cv2.flip(kernel, -1)

    hkw = kernel.shape[1] // 2
    hkh = kernel.shape[0] // 2

    padImg = cv2.copyMakeBorder(image, hkh, hkh, hkw, hkw, borderType = cv2.BORDER_CONSTANT, value = 0)

    output = np.copy(image)


    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            output[r, c] = np.sum((padImg[r:(r+kernel.shape[0]), c:(c+kernel.shape[1])]) * kernel)
    
    return output

def main():

    if len(sys.argv) < 7:
        sys.exit("Exception Error ... Not Enough Arguments")
    
    path = sys.argv[1]
    output_dir = sys.argv[2]
    KH = sys.argv[3]
    KW = sys.argv[4]
    Alpha = sys.argv[5]
    Beta = sys.argv[6]

    if len(sys.argv) < (7 + ( int(KH) * int(KW) )):
        sys.exit("Exception Error ... Not Enough Arguments")
    
    Kernel = np.empty(int(KH) * int(KW))

    for i in range(int(KH) * int(KW)):
        Kernel[i] = int(sys.argv[7+i])

    Kernel = np.reshape(Kernel, (int(KH),int(KW)))

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        sys.exit("Exception Error ... Image is None")
    
    output = applyFilter(image, Kernel)

    output = cv2.convertScaleAbs(output, alpha=float(Alpha), beta=float(Beta))

    cv2.imwrite(output_dir, output)


if __name__ == "__main__":
    main()