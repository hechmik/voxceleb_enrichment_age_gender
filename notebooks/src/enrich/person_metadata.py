from SPARQLWrapper import SPARQLWrapper, JSON
import json
import requests


def setup_query(person_complete_name: str):
    """
    Return the SPARQL query for obtaining gender, birthdate and nationality (if available) of the given person from
    DBpedia
    :param person_complete_name: person whose metadata are of interest
    :return:
    """
    query_template = """
        SELECT *
        WHERE {{
            ?p foaf:name "{}"@en;
            foaf:gender ?gender;
            dbo:birthDate ?birthdate.
            optional {{ ?p dbp:nationality ?nationality_dbp }}
            optional {{ ?p dbo:nationality ?nationality_dbo }}
        }}
    """.format(person_complete_name)
    return query_template


def query_dbpedia_endpoint(person_complete_name, sparql):
    """
    Query the given SPARQL endpoint for obtaining metadata from the person of interes
    :param person_complete_name: person of interest
    :param sparql: SPARQL Wrapper that acts as an endpoint
    :return:
    """
    query = setup_query(person_complete_name)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    query_results = sparql.query().convert()
    return query_results


def extract_metadata_from_query_results(query_results):
    """
    Given a Sparql query result, extract nationality, gender and birthdate
    :param query_results:
    :return:
    """
    if query_results["results"]["bindings"]:
        raw_metadata = query_results["results"]["bindings"][0]
        gender = raw_metadata['gender']['value'].lower()
        birth_date = raw_metadata['birthdate']['value']
        if "nationality_dbp" in raw_metadata.keys():
            nationality = raw_metadata['nationality_dbp']['value'].lower()
        elif "nationality_dbo" in raw_metadata.keys():
            nationality = raw_metadata['nationality_dbo']['value'].lower()
        else:
            nationality = ""
        return birth_date, gender, nationality
    else:
        raise ValueError


def get_person_metadata(person_complete_name: str, endpoint: str):
    """
    Return a dictionary with gender, birth date and nationality of the person of interest
    :param person_complete_name: person of interest in the format "Name Surname"
    :param endpoint: which service to query
    :return:
    """
    if endpoint == "dbpedia":
        person_metadata = get_metadata_dbpedia(person_complete_name)
    elif endpoint == "wikidata":
        person_metadata = get_metadata_wikidata(person_complete_name)
    else:
        raise ValueError("Invalid endpoint")
    return person_metadata


def get_metadata_dbpedia(person_complete_name):
    """
    Return gender, birth date and nationality of the current person by querying DBpedia
    :param person_complete_name:
    :return:
    """
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    query_results = query_dbpedia_endpoint(person_complete_name, sparql)
    try:
        birth_date, gender, nationality = extract_metadata_from_query_results(query_results)

        person_metadata = {"complete_name": person_complete_name,
                           "gender": gender,
                           "birth_date": birth_date,
                           "nationality": nationality}
    except ValueError:
        print("Could not get metadata for {}: is the person's name spelled correctly?".format(person_complete_name))
        person_metadata = {}
    return person_metadata


def get_wikidata_entities(person_complete_name):
    """
    Return all the plausible Entities IDs associated with the current person.
    IDs are ordered from the most likely to the least likely (according to Wikidata)
    :param person_complete_name:
    :return:
    """
    endpoint = "https://www.wikidata.org/w/api.php?action=wbsearchentities&search={}&language=en&format=json".format(
        person_complete_name)
    content = json.loads(requests.get(endpoint).content)
    entities = content['search']
    entities_ids = [entity['id'] for entity in entities]
    return entities_ids


def get_wikidata_properties(entity_id):
    """
    Return birth date, gender and nationality of the given entity ID
    :param entity_id: Wikidata Entity (e.g. Q10490 for Ayrton Senna)
    :return:
    """
    entity_endpoint = "https://www.wikidata.org/w/api.php?action=wbgetclaims&entity={}&format=json"
    url_of_interest = entity_endpoint.format(
        entity_id
    )
    content = requests.get(url_of_interest).content
    content = json.loads(content)['claims']

    # Birth date
    birth_date = None
    try:
        birth_date = content['P569'][0]['mainsnak']['datavalue']['value']['time']
    except KeyError:
        print("Birth date not available")
    except Exception as ex:
        print(ex)

    # Sex/gender
    gender = None
    try:
        sex_entity = content['P21'][0]['mainsnak']['datavalue']['value']['id']
        sex_entity_id_desc = {
            "Q6581097": "male",
            "Q6581072": "female",
            "Q1097630": "intersex",
            "Q1052281": "transgender female",
            "Q2449503": "transgender male"
        }  # Source: https://www.wikidata.org/wiki/Property:P21
        gender = sex_entity_id_desc[sex_entity]
    except KeyError:
        print("Gender not available")
    except Exception as ex:
        print(ex)

    # Citizenship
    
    citizenship = None
    try:
        country_entity = content['P27'][0]['mainsnak']['datavalue']['value']['id']
        country_name_id = "P3417"
        url_of_interest = entity_endpoint.format(
            country_entity)
        country_content = requests.get(url_of_interest).content
        country_content = json.loads(country_content)['claims']
        citizenship = country_content[country_name_id][0]['mainsnak']['datavalue']['value']
    except KeyError:
        print("Citizenship not available")
    except Exception as ex:
        print(ex)

    person_metadata = {
                       "gender": gender,
                       "birth_date": birth_date,
                       "nationality": citizenship
    }
    return person_metadata


def get_metadata_wikidata(person_complete_name):
    """
    Get birth date, gender and nationality (expressed with country name) for the given person
    :param person_complete_name: Person you are interested in
    :return:
    """
    entities_ids = get_wikidata_entities(person_complete_name)
    person_metadata = {}
    for entity_id in entities_ids:
        if not person_metadata:
            try:
                person_metadata = get_wikidata_properties(entity_id)
                person_metadata['name'] = person_complete_name
            except Exception as ex:
                print(ex)
        else:
            break
    if not person_metadata:
        print("Could not get metadata for {}: is the person's name spelled correctly?".format(person_complete_name))
    return person_metadata
