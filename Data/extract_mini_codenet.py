import os
import shutil
from tqdm import tqdm
import pandas as pd


def create_mini(src_folder, tgt_folder, mini_percent, random_state=1337):
    src_path_to_meta = src_folder + "metadata"
    src_path_to_data = src_folder + "data"

    # For each metadata file...
    for meta in tqdm(os.listdir(src_path_to_meta)):
        if meta.startswith("p0"):
            # Step #1 – Take metadata file
            meta_df = pd.read_csv(src_path_to_meta + "/" + meta)

            # If there are no submissions, then ignore the problem
            if len(meta_df) == 0:
                continue

            # Step #2 – Remove unneeded columns from the DataFrame
            meta_df.drop(
                columns=["user_id", "date", "original_language"],
                inplace=True
            )

            # Step #3: Discard rows with non-trivial submission status
            meta_df = meta_df[
                ~meta_df["status"].str.lower().isin([
                    "judge not available",
                    "wa: presentation error",
                    "waiting for judging",
                    "waiting for re-judging",
                    "internal error",
                    "judge system error",
                ])
            ]

            # Step #4 – Random sample while keeping distribution of submission status
            meta_df = meta_df.groupby(["status"]).sample(
                frac=mini_percent,
                replace=False,
                random_state=random_state
            )

            # Step #5 – Read submission files in random sample
            submissions_contents = []
            columns = [
                "problem_id", "language",
                "submission_id", "filename_ext"
            ]
            for problem_id, lang, submission_id, file_ext in meta_df[columns].values:
                submission_path = f"{src_path_to_data}/{problem_id}/{lang}/{submission_id}.{file_ext}"
                with open(submission_path, "r") as submission_contents:
                    submissions_contents.append(submission_contents.read())

            # Step #6 – Append submission contents to the dataframe
            meta_df["solution"] = submissions_contents

            # Step #7 – Save dataframe as feather file
            meta_df.reset_index(inplace=True)
            meta_df.to_feather(tgt_folder + meta[:-4] + ".ftr")

            continue
        
        if meta == "problem_list.csv":
            shutil.copyfile(src_path_to_meta+"/"+meta, tgt_folder+meta)


if __name__ == '__main__':
    # percentage of the original dataset that mini should consist of | MODIFY-ABLE
    mini_percent = 0.4
    if mini_percent <= 1.0:
        create_mini(
            src_folder='./Project_CodeNet/',
            tgt_folder="./mini_codenet/",
            mini_percent=mini_percent
        )
        print('Finished creating mini_codenet')
    else:
        print('ERROR: MAKE SURE THAT mini_percent IS <= 1.0')
