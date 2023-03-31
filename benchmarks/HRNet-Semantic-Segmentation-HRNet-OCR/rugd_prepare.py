import os
import cv2
import numpy as np

rugd = '/home/furkan/Downloads/Rellis-3D Downloads/RUGD_Original'
images = os.path.join(rugd, 'RUGD_frames-with-annotations')
labels = os.path.join(rugd, 'RUGD_annotations')
size = (1920, 1200)

save_dir = os.path.join("/home/furkan/RELLIS-3D/benchmarks/HRNet-Semantic-Segmentation-HRNet-OCR/RUGD")
os.makedirs(save_dir, exist_ok=True)

for folder in os.listdir(labels):
    os.makedirs(os.path.join(save_dir, folder, "pylon_camera_node_label_color"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, folder, "pylon_camera_node_label_id"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, folder, "pylon_camera_node"), exist_ok=True)
    print("Processing folder: ", folder)

    for file in os.listdir(os.path.join(labels, folder)):
        image = cv2.imread(os.path.join(images, folder, file), cv2.IMREAD_COLOR)
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(save_dir, folder, "pylon_camera_node", file), image)
        # cv2.imshow("image", image)

        label = cv2.imread(os.path.join(labels, folder, file), cv2.IMREAD_COLOR)   
        pixels = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

        pixels[np.all(pixels == [255, 229, 204], axis=-1)] = [108,64,20]
        pixels[np.all(pixels == [255, 128, 0], axis=-1)] = [110,22,138]
        pixels[np.all(pixels == [153, 76, 0], axis=-1)] = [108,64,20]
        pixels[np.all(pixels == [102, 102, 0], axis=-1)] = [110,22,138]
        pixels[np.all(pixels == [0, 255, 128], axis=-1)] = [255,255,0]
        pixels[np.all(pixels == [0, 102, 102], axis=-1)] = [0,153,153]
        pixels[np.all(pixels == [153, 204, 255], axis=-1)] = [110,22,138]
        pixels[np.all(pixels == [102, 255, 255], axis=-1)] = [255,0,0]
        pixels[np.all(pixels == [101, 101, 11], axis=-1)] = [170,170,170]
        pixels[np.all(pixels == [114, 85, 47], axis=-1)] = [255,0,127]
        
        label_color = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        label_color = cv2.resize(label_color, size, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(save_dir, folder, "pylon_camera_node_label_color", file), label_color)
        # cv2.imshow("label_color", label_color)

        pixels[np.all(pixels == [0,0,0], axis=-1)]  = [0, 0, 0]       
        pixels[np.all(pixels == [108, 64, 20], axis=-1)] = [1, 1, 1]  
        pixels[np.all(pixels == [0,102,0], axis=-1)] = [3, 3, 3]  
        pixels[np.all(pixels == [0,255,0], axis=-1)] = [4, 4, 4]  
        pixels[np.all(pixels == [0,153,153], axis=-1)] = [5, 5, 5]  
        pixels[np.all(pixels == [0,128,255], axis=-1)] = [6, 6, 6]  
        pixels[np.all(pixels == [0,0,255], axis=-1)] = [7, 7, 7]  
        pixels[np.all(pixels == [255,255,0], axis=-1)] = [8, 8, 8]  
        pixels[np.all(pixels == [255,0,127], axis=-1)] = [9, 9, 9]  
        pixels[np.all(pixels == [64,64,64], axis=-1)] = [10, 10, 10]  
        pixels[np.all(pixels == [255,0,0], axis=-1)] = [12, 12, 12]  
        pixels[np.all(pixels == [102,0,0], axis=-1)] = [15, 15, 15]  
        pixels[np.all(pixels == [204,153,255], axis=-1)] = [17, 17, 17]  
        pixels[np.all(pixels == [102, 0, 204], axis=-1)] = [18, 18, 18]  
        pixels[np.all(pixels == [255,153,204], axis=-1)] = [19, 19, 19]  
        pixels[np.all(pixels == [170,170,170], axis=-1)] = [23, 23, 23]  
        pixels[np.all(pixels == [41,121,255], axis=-1)] = [27, 27, 27]  
        pixels[np.all(pixels == [134,255,239], axis=-1)] = [31, 31, 31]  
        pixels[np.all(pixels == [99,66,34], axis=-1)] = [33, 33, 33]  
        pixels[np.all(pixels == [110,22,138], axis=-1)] = [34, 34, 34]  

        label_id = cv2.resize(pixels, size, interpolation=cv2.INTER_NEAREST)
        label_id = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
        cv2.imwrite(os.path.join(save_dir, folder, "pylon_camera_node_label_id", file), label_id)

        # print(label_id)
        # cv2.imshow("label_id", label_id)
        
        # print(np.unique(label_id))
        # label_mapping = {0: 0,
        #                  1: 0,
        #                  3: 1,
        #                  4: 2,
        #                  5: 3,
        #                  6: 4,
        #                  7: 5,
        #                  8: 6,
        #                  9: 7,
        #                  10: 8,
        #                  12: 9,
        #                  15: 10,
        #                  17: 11,
        #                  18: 12,
        #                  19: 13,
        #                  23: 14,
        #                  27: 15,
        #                  29: 1,
        #                  30: 1,
        #                  31: 16,
        #                  32: 4,
        #                  33: 17,
        #                  34: 18}  
        # temp = label_id.copy()
        # for k, v in label_mapping.items():
        #     label_id[temp == k] = v

        # print(label_id)
        # print(image.shape)
        # print(label_color.shape)
        # print(label_id.shape)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()



