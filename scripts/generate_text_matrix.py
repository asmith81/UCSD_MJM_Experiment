import json
import os

# Define the variables
models = ["pixtral", "llama", "doct"]
quant_levels = [4, 8, 16, 32]
prompt_types = ["basic_extraction", "detailed", "few_shot", "locational", "step_by_step"]
image_numbers = [1017, 1018, 1019, 1020, 1021, 1022, 1023, 1025, 1026, 1027, 
                1028, 1029, 1030, 1031, 1038, 1039, 1040, 1041, 1042, 1043]

# Create the JSON object
config = {}

# Iterate through the combinations
for model in models:
    config[model] = {}
    
    for quant_level in quant_levels:
        config[model][quant_level] = {}
        
        for prompt_type in prompt_types:
            config[model][quant_level][prompt_type] = {}
            
            for image_number in image_numbers:
                image_path = f"UCSD_MJM_Experiment/data/images/{image_number}.jpg"
                
                config[model][quant_level][prompt_type][str(image_number)] = {
                    "field_type": "both",
                    "image_path": image_path
                }

# Calculate total entries
total_entries = len(models) * len(quant_levels) * len(prompt_types) * len(image_numbers)
print(f"Total entries: {total_entries}")

# Get the current directory path
current_dir = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.join(current_dir, 'config.json')

# Save to a JSON file
with open(output_file, 'w') as f:
    json.dump(config, f, indent=2)

print(f"Configuration file '{output_file}' has been created with {total_entries} entries.")

# Verify the structure by checking a sample entry
sample = config["pixtral"][4]["basic_extraction"]["1017"]
print(f"Sample entry: {sample}")

# Save to a JSON file
with open('experiment_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"Configuration file 'experiment_config.json' has been created with {total_entries} entries.")

# Verify the structure by checking a sample entry
sample = config["pixtral"][4]["basic_extraction"]["1017"]
print(f"Sample entry: {sample}")