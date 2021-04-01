import re

import person_metadata
import logging
import json
import pandas as pd
from time import sleep
from tinydb import TinyDB, Query


def store_metadata(db_fn: str, person: dict):
    """
    Store the given videos_metadata
    :param db_fn:
    :param person:
    :return:
    """
    logging.info("store_metadata >>>")
    db = TinyDB(db_fn)
    q = Query()
    db.upsert(person, q.id == person['id'])
    db.close()
    logging.info("store_metadata <<<")


if __name__ == "__main__":
    logging.root.handlers = []
    #log_path = "/home/khaled/enrich_yt_videos/person_metadata.log"
    log_path = "/Users/kappa/repositories/enrich_youtube_videos/person_metadata.log"
    logging.basicConfig(format='%(asctime)s|%(name)s|%(levelname)s| %(message)s',
                        level=logging.INFO,
                        filename=log_path)

    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter(fmt='%(asctime)s|%(name)s|%(levelname)s| %(message)s',
                                  datefmt="%d-%m-%Y %H:%M:%S")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    logging.info("App started")
    with open("config/settings.json") as f:
        settings = json.load(f)
    people_df = pd.read_csv(settings['voxceleb_meta_csv_fn'], sep="\t")
    db_fn = settings['people_db_path']
    dataset_version = settings['voxceleb_version']
    name_col = settings['people_names_col']
    id_col = settings['id_names_col']
    for index, row in people_df.iterrows():
        current_person = row[name_col].replace("_", " ")
        # Just for name having a . in it (e.g. A.J. Buckley): Dbpedia expect a space after the dot char
        current_person = re.sub(r"([A-z])\.([A-z])", "\g<1>. \g<2>", current_person)
        current_id = row[id_col]
        metadata = person_metadata.get_person_metadata(current_person)
        if metadata:
            metadata['id'] = current_id
            metadata['dataset_name'] = dataset_version
            store_metadata(db_fn=db_fn, person=metadata)
        sleep(0.2)
