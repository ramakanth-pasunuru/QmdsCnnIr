import os
import logging
import argparse
import gdown
import zipfile
import git
import subprocess

logging.basicConfig(level = logging.INFO)

def main(args):
    """ Module to setup the codebase """
    # create the data folder
    if not os.path.exists("data"):
        logging.info("creating 'data' directory...")
        os.mkdir("data")
    if not os.path.exists("data/wikisum"):
        logging.info("creating 'data/wikisum' directory...")
        os.mkdir("data/wikisum")
    if not os.path.exists("data/qmdscnn"):
        logging.info("creating 'data/qmdscnn' directory...")
        os.mkdir("data/qmdscnn")
    # create the results folder
    if not os.path.exists("results"):
        logging.info("creating the 'results' directory...")
        os.mkdir("results")
    
    if not args.ignore_datasets:
        # download and unzip the wikisum dataset
        output_path = "data/wikisum/ranked_wiki_b40.zip"
        if not os.path.exists(output_path):
            logging.info("Downloading the encoded WikiSum dataset...")
            url = "https://drive.google.com/uc?id=1AnqeUpLkO9MR3PH0V8q32A6PEPDEZ0td&export=download"
            gdown.download(url, output_path, quiet=False)
            logging.info("Unziping the data...")
            with zipfile.ZipFile(output_path, "r") as zip_ref:
                zip_ref.extractall("data/wikisum")

        output_path = "data/wikisum/ranked_wiki_b40_query.zip"
        if not os.path.exists(output_path):
            logging.info("Downloading the encoded WikiSum dataset...")
            url = "https://drive.google.com/uc?id=1RdX-t3pznnyaGyrswFubAfoo9S9w9K5d&export=download"
            gdown.download(url, output_path, quiet=False)
            logging.info("Unziping the data...")
            with zipfile.ZipFile(output_path, "r") as zip_ref:
                zip_ref.extractall(data/wikisum)


        # download and unzip the QMDSCNN dataset
        output_path = "data/qmdscnn/pytorch_qmdscnn.zip" 
        if not os.path.exists(output_path):
            logging.info("Downloading the encoded QMDSCNN dataset...")
            url = "https://drive.google.com/uc?id=1KXsvfnK6s6cnYQzD8ZOkXPdA6r5-quPK&export=download"
            gdown.download(url, output_path, quiet=False)
            logging.info("Unziping the data...")
            with zipfile.ZipFile(output_path, "r") as zip_ref:
                zip_ref.extractall("data/qmdscnn")

        output_path = "data/qmdscnn/pytorch_qmdscnn_query.zip"
        if not os.path.exists(output_path):
            url = "https://drive.google.com/uc?id=12i_3dikeJLsOj-SQGPmc4w9Is7fB-hT-&export=download"
            gdown.download(url, output_path, quiet=False)
            logging.info("Unziping the data...")
            with zipfile.ZipFile(output_path, "r") as zip_ref:
                zip_ref.extractall(data/qmdscnn)


    # download the pyrouge git repo
    if not os.path.exists("pyrouge"):
        repo_url = "https://github.com/andersjo/pyrouge.git"
        logging.info(f"Downloading repo: {repo_url}")
        git.Git(".").clone(repo_url)


    # set the ROUGE path
    rouge_path = os.path.join(os.getcwd(),"pyrouge/tools/ROUGE-1.5.5")
    subprocess.run(["pyrouge_set_rouge_path", f"{rouge_path}"])

    


    




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ignore_datasets", default=False, action="store_true")
    args = parser.parse_args()
    main(args)
