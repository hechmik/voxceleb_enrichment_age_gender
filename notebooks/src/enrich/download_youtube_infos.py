from tinydb import TinyDB, Query
import os
import json
import youtube_metadata
import logging
from googleapiclient.errors import HttpError
from time import sleep


def store_metadata(db_fn: str, video: dict):
    """
    Store the given videos_metadata
    :param db_fn:
    :param video:
    :return:
    """
    logging.info("store_metadata >>>")
    db = TinyDB(db_fn)
    q = Query()
    db.upsert(video, q.video_id == video['video_id'])
    db.close()
    logging.info("store_metadata <<<")


def retrieve_already_parsed_ids(db_path: str):
    db = TinyDB(db_path)
    q = Query()
    query_result = db.search(q.video_id.exists())
    video_ids = []
    for item in query_result:
        video_ids.append(item['video_id'])
    return video_ids


def get_videos(db_path:str):
    db = TinyDB(db_path)
    q = Query()
    query_result = db.search(q.video_id.exists())
    return query_result


def get_all_video_ids(base_path: str):
    logging.info("get_all_video_ids >>>")
    subdirectories = os.listdir(base_path)
    video_ids = []
    for subdir in subdirectories:
        subdir_complete_path = os.path.join(base_path, subdir)
        if os.path.isdir(subdir_complete_path):
            subdir_complete_path = os.path.join(base_path, subdir)
            current_video_ids = os.listdir(subdir_complete_path)
            video_ids = video_ids + current_video_ids
    video_ids_set = set(video_ids)
    video_ids = [*video_ids_set]
    logging.info("get_all_video_ids <<<")
    return video_ids


def get_delta(ids_in_dir, ids_already_parsed):

    logging.info("get_delta >>>")
    if not ids_already_parsed:
        return ids_in_dir
    delta = [item for item in ids_in_dir if item not in ids_already_parsed]
    logging.info("{} video to download!".format(len(delta)))
    logging.info("get_delta <<<")
    return delta


if __name__ == "__main__":
    logging.root.handlers = []
    #log_path = "/Users/kappa/repositories/enrich_youtube_videos/yt_metadata.log"
    log_path="/home/khaled/enrich_yt_videos/yt_metadata.log"
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
    logging.info("Settings loaded")
    db_fn = settings['db_path']
    credential = settings['api_key']
    yt_video_ids = get_all_video_ids(settings['video_id_basefolder'])
    already_parsed_ids = retrieve_already_parsed_ids(db_fn)
    yt_video_ids = get_delta(yt_video_ids, already_parsed_ids)
    api = youtube_metadata.load_api_instance(credential)

    for video_id in yt_video_ids:
        try:
            metadata_current_video = youtube_metadata.get_complete_video_infos(video_id, False, True, api)
            store_metadata(db_fn, metadata_current_video)
            logging.debug("{} APPENDED SUCCESSFULLY!".format(metadata_current_video))
            sleep(0.4)
        except HttpError as http_error:
            logging.error(http_error)
            sleep(3)
        except Exception as ex:

            logging.error("Unable to retrieve infos for the following video id: {video_id}".format(video_id=video_id))
            logging.error(ex)
            message = {"video_id": video_id,
                       "title": "",
                       "description": "",
                       "publishing_date": "",
                       "person": [],
                       "year_in_title": ""}
            store_metadata(db_fn, message)
            sleep(2)

