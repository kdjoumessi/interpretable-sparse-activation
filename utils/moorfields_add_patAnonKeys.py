import pandas as pd


if __name__ == "__main__":
    meta_image_dr = pd.read_csv("dataset/MEH_DR.csv")
    meta_all_dr = pd.read_csv("dataset/DR_retinal_fundus_ALL.csv")
    meta_all_dr["scan_path"] = meta_all_dr["scan_path"].apply(lambda x: x.split('.')[0])
    
    patient_anon_keys = []
    for i, image in enumerate(meta_image_dr.Image):
        if i % 1000 == 0:
            print(i)
        dicom_name = image.split('/')[1].split('.')[0]
        patient_anon_keys.append(meta_all_dr[meta_all_dr["scan_path"]==dicom_name].patAnonKey.item())
        
    meta_image_dr["patAnonKey"] = patient_anon_keys
    
    meta_image_dr.to_csv("dataset/meta_image_dr_patAnonKey.csv")