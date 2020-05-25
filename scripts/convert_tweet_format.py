import pandas as pd
import glob

dir_original_data = 'C:/Users/danie/Documents/Research Data/Original data/StockTwits SP500/'

all_files = glob.glob(dir_original_data + "*.tsv")
for f in all_files:
    pd.read_csv(
        f,
        encoding="ANSI", quotechar='"', delimiter="\t", engine='python').to_csv(f, sep="\t", quotechar='"', encoding="UTF-8")

