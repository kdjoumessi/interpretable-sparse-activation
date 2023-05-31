import os
import pandas as pd


if __name__ == "__main__":
    meta = pd.read_csv("./dataset/meta_image_dr_patAnonKey.csv")
    image_names_wo_subfolders = [name.split('/')[1] for name in meta.Image]
    meta["Image"] = image_names_wo_subfolders
    
    meta.to_csv("./dataset/meta_image_dr.csv")
    
    image_names_good_quality = os.listdir("./dataset/good_quality/")
    intersect_image_names = list(set(image_names_wo_subfolders).intersection(set(image_names_good_quality)))
    
    meta_intersect = meta.query(f'Image in {list(intersect_image_names)}')
    
    meta_intersect.to_csv("./dataset/meta_image_dr_good_quality.csv")