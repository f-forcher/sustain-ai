import pandas as pd
import requests as rq
import os
import sys

def get_link_pdf(uni_idx, year, links_df):
    link = links_df.loc[uni_idx, year]
    if link == "":
        return None
    if link[-4:] != ".pdf":
        print(f"[WARN] Link {link} is not a pdf, skipping", sys.stderr)
        return None
    return link

def get_normalized_uni_names(links_df):
    return links_df.index.to_series().apply(lambda x: x.strip().replace(",", "").replace(".", "").replace(" ", "_"))

def create_files_in_dir(links_df ):
    unis = get_normalized_uni_names(links_df)
    years = links_df.columns.to_series()
    
    for u_idx, u_name in unis.items():
        dir_name = f"./pdf_download/{u_name}"
        try:
            os.makedirs(dir_name, exist_ok=True)
            print(f"Directory {dir_name} created.", sys.stderr)
        except OSError as error:
            sys.stderr.write(f"Error creating directory {dir_name}: {error}\n")
            continue
        
        for y in years:
            file_path = f"{dir_name}/{y}.pdf"
            if os.path.exists(file_path):
                sys.stderr.write(f"The file {file_path} already exists.\n")
                continue
            else:
                try:
                    # Get the file content
                    url = get_link_pdf(u_idx, y, links_df)
                    if url is None:
                        print(f"[INFO] No link, skipping {u_name}/{y}", sys.stderr)
                        continue
                    
                    response = rq.get(url)
                    response.raise_for_status()  # raise exception if invalid response

                    # Write the content to the file
                    with open(file_path, "wb") as file:
                        file.write(response.content)
                    print(f"File {file_path} created.", sys.stderr)
                except (OSError, rq.RequestException) as error:
                    sys.stderr.write(f"Error creating file {file_path}: {error}\n")    
    pass


def read_links_table(xls_table):
    df = pd.read_excel(xls_table, index_col=0).fillna("").applymap(lambda s: s.strip())
    return df

def run():
    links_df = read_links_table("./reports_links.xlsx")
    links_normalized = links_df
    links_normalized["HEI_names_norm"] = get_normalized_uni_names(links_df)
    links_normalized.set_index("HEI_names_norm", inplace=True)
    links_normalized.to_csv("./reports_norm.csv")
    create_files_in_dir(links_df)
    

if __name__ == "__main__":
    run()
