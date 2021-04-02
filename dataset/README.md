# Dataset
This repository contains the enriched dataset for Gender and Age recognition, together with the informations regarding train and test speakers.
## VoxCeleb enriched dataset
The complete dataset is a CSV file having the following columns:

- **Name**: Full name (or artistic name) of the given celebrity
- **gender_wiki**: gender value according to Wikidata
- **birth_date_wiki**:  birth date value according to Wikidata
- **nationality_wiki** nationality value according to Wikidata
- **gender_dbpedia**: gender value according to DBPedia
- **birth_date_dbpedia**: birth date value according to DBPedia
- **nationality_dbpedia**: nationality value according to DBPedia
- **gender_gkg**: gender value according to Google Knowledge Graph
- **birth_date_gkg**: birth date value according to Google Knowledge Graph
- **nationality_gkg**: nationality value according to Google Knowledge Graph
- **video_id**: YouTube video ID used for obtaining the utterance(s)
- **title**: Title of YouTube video
- **publishing_date**: Date of upload in YouTube
- **description**: Description of YouTube video
- **year_in_title**: Eventual year (numeric value of 4 digits) in the Title field
- **VoxCeleb_ID**: Id associated to the current celebrity, as found in VoxCeleb 1 and 2 meta CSV files
- **gender**: gender value if there is unanymous consensus among DBPedia, Google Knowledge Graph
- **birth_year**: year of birth if there is unanymous consensus among DBPedia, Google Knowledge Graph
- **year_upload_yt**: year when the YouTube video was uploaded
- **recording_year**: Recording year computed as described in the paper (year_upload_yt referenced in both Title and Description)
- **recording_year_title_only**: Alternative recording year used in some training scenario (year_upload_yt referenced only in Title)
- **speaker_age**: proposed age (recording_year - birth_year)
- **speaker_age_title_only**: alternative age value used in some training scenarios (recording_year_title_only - birth_year)
