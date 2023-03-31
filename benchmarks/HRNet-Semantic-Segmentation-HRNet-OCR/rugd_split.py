import os

rugd = "/home/furkan/RELLIS-3D/benchmarks/HRNet-Semantic-Segmentation-HRNet-OCR/RUGD"
rugd_train = open(os.path.join(rugd, "rugd_train.lst"), "w")
rugd_test = open(os.path.join(rugd, "rugd_test.lst"), "w")
rugd_val = open(os.path.join(rugd, "rugd_val.lst"), "w")

rellis_train = 3302
rellis_test = 1672
rellis_val = 983
total = rellis_test + rellis_train + rellis_val

for folder in os.listdir(rugd):
    print("\nProcessing folder: ", folder)
    if not os.path.isdir(os.path.join(rugd, folder)):
        continue
    images = os.path.join(rugd, folder, "pylon_camera_node")
    labels = os.path.join(rugd, folder, "pylon_camera_node_label_id")
    size = len(os.listdir(images))
    train_size = int(size * rellis_train / total)
    test_size = int(size * rellis_test / total)
    print("Size: ", size)
    print("Train size: ", train_size)
    print("Test size: ", test_size)
    print("Val size: ", size - train_size - test_size)
    for i, image in enumerate(os.listdir(images)):
        if i < train_size:
            rugd_train.write(os.path.join(folder, "pylon_camera_node", image) + " " + 
                             os.path.join(folder, "pylon_camera_node_label_id", image) + "\n")
        elif i < train_size + test_size:
            rugd_test.write(os.path.join(folder, "pylon_camera_node", image) + " " + 
                            os.path.join(folder, "pylon_camera_node_label_id", image) + "\n")
        else:
            rugd_val.write(os.path.join(folder, "pylon_camera_node", image) + " " + 
                           os.path.join(folder, "pylon_camera_node_label_id", image) + "\n")

rugd_train.close()
rugd_test.close()
rugd_val.close()            
            

