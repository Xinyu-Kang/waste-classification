import cv2
import numpy as np
import glob

if __name__ == "__main__":

    path_rgb = '../../compare_output_rgb'
    path_rgbd = '../../compare_output_rgbd'
    path_rgbd_super = '../../compare_output_rgbd_super'
    path_rgbd_only90 = '../../compare_output_only90'

    for image_path_rgb in glob.glob(f'{path_rgb}/*.png'): 

        image_name = image_path_rgb.split('/')[-1]

        image_path_rgbd = f'{path_rgbd}/{image_name}'
        image_path_rgbd_super = f'{path_rgbd_super}/{image_name}'
        image_path_only_90 = f'{path_rgbd_only90}/{image_name}'

        result_image_rgb = cv2.imread(image_path_rgb)
        result_image_rgbd = cv2.imread(image_path_rgbd)
        result_image_rgbd_super = cv2.imread(image_path_rgbd_super)
        result_image_only90 = cv2.imread(image_path_only_90)

        if result_image_rgb is not None and result_image_rgbd is not None and result_image_rgbd_super is not None and result_image_only90 is not None:

            result_concat = cv2.vconcat([result_image_rgb, result_image_rgbd, result_image_rgbd_super, result_image_only90])
            cv2.imwrite(f'../../compare_output/{image_name}', result_concat)
