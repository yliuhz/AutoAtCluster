

import pandas as pd
import numpy as np
import os

if __name__ == "__main__":

    datasets = ["citeseer", "wiki", "pubmed", "amazon-photo", "amazon-computers"]

    for dataset in datasets:
        print(dataset)
        
        outfile_ami = "Ours_gs_AMI_{}.csv".format(dataset)
        outfile_nmi = "Ours_gs_NMI_{}.csv".format(dataset)
        outfile_ari = "Ours_gs_ARI_{}.csv".format(dataset)

        if dataset in ["citeseer", "wiki"]:
            threads = 5
        else:
            threads = 8

        df_data = pd.DataFrame(columns=["models","dataset","params","1m","2m","3m","4m","5m","6m","7m","8m","9m","10m"])
        df_data_ami = df_data.copy()
        df_data_nmi = df_data.copy()
        df_data_ari = df_data.copy()
        for thread in range(1,threads+1):
            df = pd.read_csv("Ours_gs_AMI_{}_{}.csv".format(dataset, thread))
            df_data_ami = pd.concat([df_data_ami, df], axis=0)

            df = pd.read_csv("Ours_gs_NMI_{}_{}.csv".format(dataset, thread))
            df_data_nmi = pd.concat([df_data_nmi, df], axis=0)

            df = pd.read_csv("Ours_gs_ARI_{}_{}.csv".format(dataset, thread))
            df_data_ari = pd.concat([df_data_ari, df], axis=0)

        df_data_ami.to_csv(outfile_ami, index=False)
        df_data_nmi.to_csv(outfile_nmi, index=False)
        df_data_ari.to_csv(outfile_ari, index=False)



