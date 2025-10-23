import os, argparse, random, re
from tqdm import tqdm

import torch
from transformers import GemmaTokenizerFast, GemmaForCausalLM
from transformers import GemmaTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

from utils import set_random_seeds, compute_metrics, save_queries_and_records, compute_records
from prompting_utils import read_schema, save_logs, extract_sql_query
from load_data import load_prompting_data

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
MAX_NEW_TOKENS = 250  

BOS = "<bos>"
EOS = "<eos>"

def get_args():
    parser = argparse.ArgumentParser(description='Text-to-SQL experiments with prompting.')
    parser.add_argument('-s', '--shot', type=int, default=0)
    parser.add_argument('-p', '--ptype', type=int, default=0)
    parser.add_argument('-m', '--model', type=str, default='gemma')
    parser.add_argument('-q', '--quantization', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--experiment_name', type=str, default='experiment')
    args = parser.parse_args()
    return args


def create_prompt(sentence, k, train_x=None, train_y=None, schema=None, ptype=0):
    if ptype == 0:
        instruct = "Convert English questions to SQL queries.\n\n"
        
    elif ptype == 1:
        instruct = "Convert English questions to SQL queries for a flight database. Use proper SQL syntax with correct table joins and aliases. Output only the SQL query.\n\n"
        
    else:  
        instruct = "Convert English questions to SQL queries for a flight database. Use proper SQL syntax with SELECT, FROM, WHERE, and JOIN clauses. Always use table aliases like flight_1, city_1, airport_service_1, etc. Join tables correctly through their foreign key relationships. End all queries with a semicolon.\n\n"
        
        if schema:
            instruct += f"Schema: {schema[:300]}\n\n"
    
   
    if k > 0 and train_x and train_y:
        
        for idx in range(min(k, len(train_x))):
            sql = train_y[idx].strip()
            if not sql.endswith(';'):
                sql += ';'
            instruct += f"{train_x[idx]}\n{sql}\n\n"
    
    instruct += sentence

    return instruct


def exp_kshot(tokenizer, model, inputs, k, train_x=None, train_y=None, 
              schema=None, ptype=0, ground_truth=None):  

    raw_outputs = []
    extracted_queries = []

    for i, sentence in tqdm(enumerate(inputs)):
        prompt = create_prompt(sentence, k, train_x, train_y, schema, ptype)

        
        if i == 0:
            print(f"\n{'='*80}")
            print(f"PROMPT LENGTH: {len(prompt)} chars")
            print(f"NUMBER OF EXAMPLES: {k}")
            print(f"FIRST 500 CHARS:")
            print(prompt[:500])
            print(f"LAST 300 CHARS:")
            print(prompt[-300:])
            print(f"{'='*80}\n")

        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
        
        
        if i == 0:
            print(f"TOKENIZED LENGTH: {input_ids['input_ids'].shape[1]} tokens")
            if input_ids['input_ids'].shape[1] >= 2048:
                print("⚠️  WARNING: PROMPT TRUNCATED! Examples may be cut off!\n")
        
        outputs = model.generate(
            **input_ids, 
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.3,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        
        if len(prompt) < len(response):
            response = response[len(prompt):].strip()
        
        raw_outputs.append(response)
        extracted_query = extract_sql_query(response)
        extracted_queries.append(extracted_query)
        
        
        if i < 5:
            print(f"\n{'─'*80}")
            print(f"Example {i+1}")
            print(f"{'─'*80}")
            print(f"Question: {sentence}")
            print(f"\nGenerated (first 200 chars):")
            print(f"  {response[:200]}")
            print(f"\nExtracted SQL:")
            print(f"  {extracted_query[:150]}...")
            
            
            if ground_truth:
                print(f"\nGround Truth:")
                print(f"  {ground_truth[i][:150]}...")
            
            print(f"{'─'*80}")
    
    return raw_outputs, extracted_queries


def eval_outputs(eval_x, eval_y, gt_sql_pth, model_sql_path, gt_record_path, model_record_path, extracted_queries):
    
    save_queries_and_records(extracted_queries, model_sql_path, model_record_path)
    
    sql_em, record_em, record_f1, error_msgs = compute_metrics(
        gt_sql_pth, model_sql_path, gt_record_path, model_record_path
    )
    
    error_rate = sum(1 for msg in error_msgs if msg != "") / len(error_msgs) if len(error_msgs) > 0 else 0
    
    return sql_em, record_em, record_f1, error_msgs, error_rate


def initialize_model_and_tokenizer(model_name, to_quantize=False):

    if model_name == "gemma":
        model_id = "google/gemma-1.1-2b-it"
        tokenizer = GemmaTokenizerFast.from_pretrained(model_id)
        model = GemmaForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16
        ).to(DEVICE)
    elif model_name == "codegemma":
        model_id = "google/codegemma-7b-it"
        tokenizer = GemmaTokenizer.from_pretrained(model_id)
        if to_quantize:
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=nf4_config,
                device_map="auto"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16
            ).to(DEVICE)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer, model


def main():
    args = get_args()
    set_random_seeds(args.seed)
    
    os.makedirs('results', exist_ok=True)
    os.makedirs('records', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    data_folder = 'data'
    train_x, train_y, dev_x, dev_y, test_x = load_prompting_data(data_folder)

    schema = None
    if args.ptype in [1, 2]:
        schema_path = os.path.join(data_folder, 'flight_database.schema')
        if os.path.exists(schema_path):
            schema = read_schema(schema_path)

    tokenizer, model = initialize_model_and_tokenizer(args.model, args.quantization)

    for eval_split in ["dev", "test"]:
        eval_x, eval_y = (dev_x, dev_y) if eval_split == "dev" else (test_x, None)

        print(f"\n{'='*10}")
        print(f"Processing {eval_split} set...")
        print(f"{'='*10}")
        
        raw_outputs, extracted_queries = exp_kshot(
            tokenizer, model, eval_x, args.shot, train_x, train_y, schema, args.ptype,  ground_truth=eval_y
        )

        model_sql_path = f'results/{args.model}_{args.experiment_name}_{eval_split}.sql'
        model_record_path = f'records/{args.model}_{args.experiment_name}_{eval_split}.pkl'

        if eval_split == "dev":
            gt_sql_path = f'data/{eval_split}.sql'
            gt_record_path = f'records/{eval_split}_gt_records.pkl'

            sql_em, record_em, record_f1, model_error_msgs, error_rate = eval_outputs(
                eval_x, eval_y,
                gt_sql_pth=gt_sql_path,
                model_sql_path=model_sql_path,
                gt_record_path=gt_record_path,
                model_record_path=model_record_path,
                extracted_queries=extracted_queries
            )
            
            print(f"\n{'='*10}")
            print(f"RESULTS - {eval_split.upper()}")
            print(f"{'='*10}")
            print(f"Record F1: {record_f1:.4f}")
            print(f"Record EM: {record_em:.4f}")
            print(f"SQL EM: {sql_em:.4f}")
            print(f"Error rate: {error_rate*100:.2f}%")
            print(f"{'='*10}\n")

            log_path = f"logs/{args.model}_{args.experiment_name}_{eval_split}.log"
            save_logs(log_path, sql_em, record_em, record_f1, model_error_msgs)
        else:
            save_queries_and_records(extracted_queries, model_sql_path, model_record_path)
            print(f"{eval_split} set: Predictions saved to {model_sql_path}")


if __name__ == "__main__":
    main()