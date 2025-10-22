import os
import re
BOS = "<bos>"
EOS = "<eos>"
def read_schema(schema_path):
    '''
    Read the .schema file
    '''
    with open(schema_path, 'r') as f:
        schema = f.read()

    return schema

def extract_sql_query(response):
    '''
    Extract the SQL query from the model's response
    '''
    # TODO
    response = response.replace(BOS, "").replace(EOS, "")
    response = response.replace('<s>', '').replace('</s>', '')
    response = response.replace('<bos>', '').replace('<eos>', '')
    
    # If "SQL:" exists, take everything after the LAST occurrence
    if "SQL:" in response.upper():
        parts = response.upper().split("SQL:")
        response = parts[-1]
    
    # Try to extract SELECT query with semicolon
    match = re.search(r'(SELECT\s+.*?;)', response, re.DOTALL | re.IGNORECASE)
    query = match.group(1) if match else response

    query = re.sub(r'```(sql)?', '', query, flags=re.IGNORECASE)
    query = query.replace("\n", " ")
    query = re.sub(r'\s+', ' ', query).strip()

    if query and not query.endswith(';'):
        query += ';'
    return query


def save_logs(output_path, sql_em, record_em, record_f1, error_msgs):
    '''
    Save the logs of the experiment to files.
    You can change the format as needed.
    '''
    with open(output_path, "w") as f:
        f.write(f"SQL EM: {sql_em}\nRecord EM: {record_em}\nRecord F1: {record_f1}\nModel Error Messages: {error_msgs}\n")