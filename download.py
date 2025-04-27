import tarfile
import zipfile
import sys
import os
import pandas as pd
import pickle

os.makedirs("data/", exist_ok=True)
if sys.argv[1] == "physio":
    url = "https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz?download"
    os.system(f"cd data && curl -L -o physio.gz {url}")
    with tarfile.open("data/physio.gz", "r:gz") as t:
        t.extractall(path="data/physio")
    os.system("rm data/physio.gz")

elif sys.argv[1] == "pm25":
    # Download the dataset
    url = "https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/STMVL-Release.zip"
    os.system(f"cd data && curl -L -o STMVL-Release.zip {url}")
    with zipfile.ZipFile("data/STMVL-Release.zip") as z:
        z.extractall("data/pm25")
    os.system("rm data/STMVL-Release.zip")

    def create_normalizer_pm25():
        df = pd.read_csv(
            "./data/pm25/Code/STMVL/SampleData/pm25_ground.txt",
            index_col="datetime",
            parse_dates=True,
        )
        test_month = [3, 6, 9, 12]
        for i in test_month:
            df = df[df.index.month != i]
        mean = df.describe().loc["mean"].values
        std = df.describe().loc["std"].values
        path = "./data/pm25/pm25_meanstd.pk"
        with open(path, "wb") as f:
            pickle.dump([mean, std], f)

    create_normalizer_pm25()
