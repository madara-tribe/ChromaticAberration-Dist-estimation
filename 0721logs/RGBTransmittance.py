import matplotlib.pyplot as plt
import numpy as np
import cv2


def transparency(image_path, original, types):
    def calmain(color_name, channel_normal, channel_filtered):
        # Calculate the mean value of the green channel for both images
        mean_normal = np.mean(channel_normal)
        mean_filtered = np.mean(channel_filtered)
        
        # Calculate the green channel transmittance
        transmittance = mean_filtered / mean_normal
        
        # Print the results
        
        print(str(color_name), " Mean value for normal lens image:", mean_normal)
        print(str(color_name), " Mean value for filtered lens image:", mean_filtered)
        print(str(color_name), " Channel Transmittance:", transmittance)

    image_normal = cv2.cvtColor(cv2.imread(original), cv2.COLOR_BGR2RGB)
    image_normal = cv2.resize(image_normal, (1440, 1120))
    image_normal = image_normal[:, int(1440/2):] 
    #image_normal = image_normal[:, :int(1440/2)]
    image_filtered = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    image_filtered = cv2.resize(image_filtered, (1440, 1120))
    image_filtered = image_filtered[:, int(1440/2):]
    #image_filtered = image_filtered[:, :int(1440/2)]
    # Ensure images are the same size
    if image_normal.shape != image_filtered.shape:
        raise ValueError("Images must be the same size for comparison")
    
    # Convert images to float32 type for precision
    image_normal = image_normal.astype(np.float32)
    image_filtered = image_filtered.astype(np.float32)
    
    # Extract the green channel from both images
    if types==1:
        color_name = 'green'
        channel_normal = image_normal[:, :, types]  # OpenCV uses BGR, so index 1 is G
        channel_filtered = image_filtered[:, :, types]
    elif types==0:
        color_name = 'blue'
        channel_normal = image_normal[:, :, types]  # OpenCV uses BGR, so index 1 is G
        channel_filtered = image_filtered[:, :, types]
    elif types==2:
        color_name = 'red'
        channel_normal = image_normal[:, :, types]  # OpenCV uses BGR, so index 1 is G
        channel_filtered = image_filtered[:, :, types]
    p_ = np.hstack([channel_normal, channel_filtered])
    plt.imshow(p_),plt.show()
    calmain(color_name, channel_normal, channel_filtered)
    

