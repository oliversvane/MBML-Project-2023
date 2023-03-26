import kaggle

kaggle.api.authenticate()
kaggle.api.dataset_download_files(
    "yuanyuwendymu/airline-delay-and-cancellation-data-2009-2018",
    path="data/",
    unzip=True,
)
