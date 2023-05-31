import os
import pandas as pd
from multi_level_split.util import train_test_split

if __name__ == "__main__":
    meta_image_dr_pat = pd.read_csv("dataset/meta_image_dr_good_quality.csv")
    # Split data into train, validation and test.
    # 85% train & val, 15% test
    dev, test = train_test_split(
        meta_image_dr_pat, "Image", split_by="patAnonKey", test_split=0.15, seed=12345
    )
    # 75% train, 10% val, 15% test
    train, val = train_test_split(
        dev, "Image", split_by="patAnonKey", test_split=10/85, seed=12345
    )
    
    split_dict = {"train": train, "val": val, "test": test}
    split_dir = "dataset/splits/"
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)

    for k, v in split_dict.items():
        v.to_csv(
            os.path.join(split_dir, "".join((k, ".csv"))),
            index=False,
        )