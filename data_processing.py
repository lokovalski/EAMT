import json 

def get_subjects_by_Q_json(input_file, output_file):
    """
    Transforms the JSON data from the input file and writes the result to the output file.
    The output JSON uses the 'name' field from the 'answer' dictionary as keys and keeps
    the most complete version (fewer None values) of the label for each key.
    
    Args:
        input_file (str): Path to the input JSON file.
        output_file (str): Path to the output JSON file.
    """
    # Helper function to count non-None values in a label
    def count_non_none_values(label):
        return sum(1 for value in label.values() if value is not None)

    data = None
    # Load input JSON
    with open(input_file, 'r') as infile:
        data = json.load(infile)

    # Transform data
    transformed_data = {}
    for entry in data:
        answer = entry['answer']
        # print("this is the answer")
        # print(answer)
        if answer["answerType"] == 'entity' and answer["answer"] != None:
            local_answer = answer["answer"]
            #print(local_answer)
            key = local_answer[0]["name"]  # Use the "name" field in "answer" dictionary as key
            label = local_answer[0]["label"]
            
            # If the key is not in transformed_data or the current entry has more non-None fields, update the entry
            if key not in transformed_data or count_non_none_values(label) > count_non_none_values(transformed_data[key]["label"]):
                transformed_data[key] = {
                    "label": label,
                    "mention": entry["answer"]["mention"],
                    "category": entry["category"],
                    "complexityType": entry["complexityType"]
                }
        if answer["answerType"] == 'date':
            pass
            # maybe fill this in later if we think of something useful



    # Write transformed data to output file
    with open(output_file, 'w') as outfile:
        json.dump(transformed_data, outfile, indent=4)


def process_jsonl_and_json(jsonl_file, json_file, output_file, include_missing = True):
    """
    Processes a JSONL file and a JSON file to map entities with labels in specified languages.
    Args:
        jsonl_file (str): Path to the input JSONL file with the "entities" feature.
        json_file (str): Path to the input JSON file with entity details.
        output_file (str): Path to the output JSON file with modified entries.
    """
    # Load the second JSON file with entity details
    with open(json_file, 'r') as jf:
        entity_data = json.load(jf)
    
    # Initialize a list for the modified JSONL entries
    modified_entries = []

    # Process the JSONL file line by line
    cont_var = True
    with open(jsonl_file, 'r') as jlf:
        for line in jlf:
            cont_var = True

            # Parse the line as a dictionary
            entry = json.loads(line)
            
            # Extract necessary fields
            source_lang = entry.get("source_locale")
            target_lang = entry.get("target_locale")
            entities = entry.get("entities", [])
            
            # Build the "entities" dictionary for the output
            entities_dict = {}
            for entity_id in entities:
                # Check if the entity ID exists in the second JSON file
                if entity_id in entity_data:
                    entity_labels = entity_data[entity_id]["label"]
                    
                    # Extract the labels for the source and target languages
                    entities_dict[entity_id] = {
                        lang: entity_labels.get(lang)
                        for lang in [source_lang, target_lang]
                        if lang in entity_labels
                    }
                else:
                    if include_missing:
                        entities_dict[entity_id] = "NA"
                    else:
                        cont_var = False

            if not cont_var: # break out if we encounter something we can't translate
                continue
            # Add the new "entities" dictionary to the entry
            entry["entities"] = entities_dict
            
            # Append the modified entry to the list
            modified_entries.append(entry)

    # Write the modified entries to the output JSON file
    with open(output_file, 'w') as outf:
        for entry in modified_entries:
            outf.write(json.dumps(entry) + '\n')

def update_source_locale(jsonl_file, new_locale, output_file):
    """
    Updates the source_locale field for all elements in a JSONL file.

    Args:
        jsonl_file (str): Path to the input JSONL file.
        new_locale (str): The new value for the source_locale field.
        output_file (str): Path to the output JSONL file.
    """
    # Open the input JSONL file and process line by line
    with open(jsonl_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            entry = json.loads(line)  # Parse the line as a dictionary
            entry["source_locale"] = new_locale  # Update the source_locale field
            outfile.write(json.dumps(entry) + '\n')  # Write the updated entry

if __name__ == "__main__": 
    #get_subjects_by_Q_json("data/mintaka.json", "Q_data.json")
    #update_source_locale("data/spanish.jsonl", "es", "data/spanish_updated.jsonl")
    process_jsonl_and_json("/Users/lolakovalski/Desktop/School/csci375/EAMT/data/spanish_updated.jsonl", "/Users/lolakovalski/Desktop/School/csci375/EAMT/data/Q_data.json", "/Users/lolakovalski/Desktop/School/csci375/EAMT/data/spanish_w_labels.json", include_missing = False)