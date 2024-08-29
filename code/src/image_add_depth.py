import cv2
import numpy as np
import glob

from depth import DepthModel 


if __name__ == "__main__":

    path = '../../trash_data'

    depth_model = DepthModel("vits")

    for image_path in glob.glob(f'{path}/test/rgb_images/*.png'): 
        print("\n===================================================")
        
        image_name = image_path.split('/')[-1]
        print("Image name: ", image_name)

        rgb_image = cv2.imread(f'{path}/test/rgb_images/{image_name}')
        print(rgb_image.shape)

        depth_map = depth_model.predict(rgb_image)
        depth_array = np.asarray(depth_map)
        depth_map_expanded = np.expand_dims(depth_map, axis=2)

        rgbd_image = np.concatenate((rgb_image, depth_map_expanded), axis=2)
        print(rgbd_image.shape)

        cv2.imwrite(f'{path}/test/images/{image_name}', rgbd_image)
        