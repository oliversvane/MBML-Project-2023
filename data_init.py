import kaggle
import os

dir_x = "data/"

kaggle.api.authenticate()
kaggle.api.dataset_download_files(
    "robikscube/flight-delay-dataset-20182022",
    path=dir_x,
    unzip=True,
)
for filename in os.listdir(dir_x):
    f = os.path.join(dir_x, filename)
    if f.m.endswith(".parquet") or filename == "Airlines.csv":
        pass
    else:
        os.remove(f)
