from apiclient.discovery import build
import json
import re
import nltk
nltk.download('averaged_perceptron_tagger')


def get_yt_credentials(fn="../credentials.json"):
    with open(fn, "r") as f:
        credential = json.load(f)
    developer_key = credential['api_key']
    return developer_key


def load_api_instance(developer_key: str):
    """
    Load YouTube API instance
    :param developer_key: YouTube API secret key
    :return:
    """
    youtube = build('youtube', 'v3', developerKey=developer_key)
    return youtube


def get_video_id(url: str):
    """
    Return the ID of the given YouTube video
    :param url: URL of YouTube video
    :return:
    """
    pattern = "watch\?v=([A-z0-9_\-]+)"
    try:
        video_id = re.search(pattern, url).group(1)
    except AttributeError:
        # ID not found using this pattern
        print("Unable to find video id for this url {}".format(url))
        video_id = ''
    return video_id


def get_youtube_video_metadata(yt_video_id: str, is_id_url: bool, add_description: bool, api):
    """
    Given a YouTube URL, return its title, publishing date and (optional) description
    :param yt_video_id: URL or video id of the YouTube video of interest
    :param is_id_url: whether the given youtube id is the complete URL or only the id
    :param add_description: Whether to include video description or not
    :param api: YouTube API wrapper
    :return:
    """
    if is_id_url:
        yt_video_id = get_video_id(yt_video_id)
    results = api.videos().list(id=yt_video_id, part='snippet').execute()
    video_info = results.get('items', [])[0]
    title = video_info['snippet']['title']
    date = video_info['snippet']['publishedAt']
    metadata_of_interest = {'video_id': yt_video_id, 'title': title, 'publishing_date': date}
    if add_description:
        description = video_info['snippet']['description']
        # Let's remove all white spaces, new lines and tabs
        description = ' '.join(description.split())
        metadata_of_interest['description'] = json.dumps(description)
    return metadata_of_interest


def get_author_and_date(youtube_title: str, settings_fn):
    """
    Extract from the given video title the involved subjects and the eventual date
    :param youtube_title: YouTube Title of Interest
    :param settings_fn: path where the settings JSON having the NER model and JAR paths is stored
    :return:
    """
    from nltk.tag.stanford import StanfordNERTagger
    with open(settings_fn) as f:
        settings = json.load(f)
    model_fn = settings['ner_model_fn']
    jar_fn = settings['ner_jar_fn']
    st = StanfordNERTagger(model_filename=model_fn,
                           path_to_jar=jar_fn)
    ner_tags = st.tag(youtube_title.split())
    result = {"person": []}
    for i in range(0, len(ner_tags) - 1):
        if ner_tags[i][1] == "PERSON" and ner_tags[i+1][1] == "PERSON":
            result['person'].append(ner_tags[i][0] + " " + ner_tags[i+1][0])
    # Find year
    year_pattern = re.compile('[1-2][0-9]{3}')
    match = year_pattern.search(youtube_title)
    if match is not None:
        result['year'] = match.group(0)
    return result


def get_complete_video_infos(video_id: str,
                             is_id_url: bool,
                             include_description: bool,
                             api,
                             settings_fn="config/settings.json"):
    """
    Given a video ID, extract metadata such as title, subjects in the title, year in title etc
    :param video_id: URL or ID of the YT video
    :param is_id_url: Whether the given identifier is a video ID or an URL
    :param include_description: whether to return the video description or not
    :param api: YouTube API wrapper
    :param settings_fn: path where the settings JSON having the NER model and JAR paths is stored
    :return:
    """
    video_infos = get_youtube_video_metadata(video_id, is_id_url, include_description, api)
    author_and_date = get_author_and_date(video_infos['title'], settings_fn)
    video_infos['person'] = author_and_date['person']
    if "year" in author_and_date.keys():
        video_infos['year_in_title'] = author_and_date['year']
    else:
        video_infos['year_in_title'] = ""
    return video_infos
