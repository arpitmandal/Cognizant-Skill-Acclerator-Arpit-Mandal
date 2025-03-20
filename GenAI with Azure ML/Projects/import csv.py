import csv
import json
import os

def convert_csv_to_jsonl(csv_file_path, jsonl_file_path):
    """
    Convert a CSV file to JSONL format
    
    Args:
        csv_file_path: Path to the input CSV file
        jsonl_file_path: Path to the output JSONL file
    """
    print(f"Converting {csv_file_path} to {jsonl_file_path}")
    
    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(jsonl_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Count total lines for progress reporting
    with open(csv_file_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f) - 1  # Subtract 1 for header
    
    with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        
        with open(jsonl_file_path, 'w', encoding='utf-8') as jsonl_file:
            for i, row in enumerate(csv_reader):
                # Convert row to JSON and write to file
                jsonl_file.write(json.dumps(row) + '\n')
                
                # Print progress every 1000 rows
                if (i + 1) % 1000 == 0 or i + 1 == total_lines:
                    print(f"Processed {i + 1}/{total_lines} rows ({(i + 1) / total_lines * 100:.2f}%)")
    
    print(f"Conversion complete. Output saved to {jsonl_file_path}")

if __name__ == "__main__":
    # Input CSV file path
    csv_file_path = r"c:\Users\IversonScarlettTAC\Downloads\generative-ai-with-azure-machine-learning\Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
    
    # Output JSONL file path with the specific name 'fine_tune_data.jsonl'
    output_dir = os.path.dirname(csv_file_path)
    jsonl_file_path = os.path.join(output_dir, "fine_tune_data.jsonl")
    
    # Convert CSV to JSONL
    convert_csv_to_jsonl(csv_file_path, jsonl_file_path)