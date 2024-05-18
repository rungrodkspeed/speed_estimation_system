import os
import cv2
import torch

from utils import ImageProcessor

image_windows = 0

def show_image(image):
    global image_windows
    
    cv2.imshow(f"image_{str(image_windows).zfill(3)}", image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    image_windows += 1

def are_lists_of_tensors_equal(list1, list2):
    if len(list1) != len(list2):
        return False
    
    for tensor1, tensor2 in zip(list1, list2):
        if not torch.equal(tensor1, tensor2):
            return False
    
    return True

if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_worker = os.cpu_count()
    
    image_processor = ImageProcessor(device=device, worker=num_worker)
    print(f"image processor: {image_processor}")
    
    image = cv2.imread("testcase_0.jpg")
    
    res_from_images = image_processor(source=[image], task="555555")
    print(res_from_images)
    
    # res_from_images = image_processor(source=[image]*100, task="preprocess")
    
    # print(len(res_from_images), res_from_images[0].shape, res_from_images[0])
    
    

    # show_image(image)
    # show_image(image)
    # print(f"all image are show : {image_windows}")