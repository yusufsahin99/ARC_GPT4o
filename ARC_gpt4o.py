import json
from PIL import Image, ImageDraw
import numpy as np
import base64
import requests
import os
import re

colors = [
    "Black",
    "White",
    "Red",
    "Green",
    "Blue",
    "Yellow",
    "Magenta",
    "Cyan",
    "Gray",
    "Maroon",
    "Olive",
    "Dark Green",
    "Turquoise",
    "Lavender"
]

def json_to_string(path, delimiter = "|"):
    with open(path, 'r') as file:
        data = json.load(file)
        strings = {'train':[], 'test':[]}
        for type in ['train', 'test']:
            for i, pair in enumerate(data[type]):
                input_s = ""
                for row in pair['input']:
                    input_s += delimiter.join([colors[i] for i in row])+ "\n"
                output_s = ""
                for row in pair['output']:
                    output_s += delimiter.join([colors[i] for i in row])+ "\n"
                strings[type].append({'input':input_s, 'output':output_s})
        return strings

def json_to_images(path):
    colors_rgb = [
    [0, 0, 0], # Black
    [255, 255, 255],        # White
    [255, 0, 0],      # Red
    [0, 255, 0],      # Green
    [0, 0, 255],      # Blue
    [255, 255, 0],    # Yellow
    [255, 0, 255],    # Magenta
    [0, 255, 255],    # Cyan
    [128, 128, 128],  # Gray
    [128, 0, 0],      # Maroon
    [128, 128, 0],    # Olive
    [0, 128, 0],      # Dark Green
    [128, 0, 128],    # Purple
    [0, 128, 128],    # Teal
    [0, 0, 128]       # Navy
    ]
    fname = path.split('/')[-1].split('.')[0]
    with open(path, 'r') as file:
        data = json.load(file)
        for type in ['train', 'test']:
            for i, pair in enumerate(data[type]):
                for x in ['input', 'output']:
                    rgb_array = np.array([[colors_rgb[val] for val in row] for row in pair[x]])
                    # Expand each pixel to 10x10
                    expanded_rgb_array = np.repeat(np.repeat(rgb_array, 10, axis=0), 10, axis=1)
                    image = Image.fromarray(np.uint8(expanded_rgb_array))
                    image.save(f"{fname}_{type}_{i}_{x}.png")


# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def get_encoded_images(path):
    colors_rgb = [
    [0, 0, 0], # Black
    [255, 255, 255],        # White
    [255, 0, 0],      # Red
    [0, 255, 0],      # Green
    [0, 0, 255],      # Blue
    [255, 255, 0],    # Yellow
    [255, 0, 255],    # Magenta
    [0, 255, 255],    # Cyan
    [128, 128, 128],  # Gray
    [128, 0, 0],      # Maroon
    [128, 128, 0],    # Olive
    [0, 128, 0],      # Dark Green
    [128, 0, 128],    # Purple
    [0, 128, 128],    # Teal
    [0, 0, 128]       # Navy
    ]
    images = {'train':[], 'test':[]}
    with open(path, 'r') as file:
        data = json.load(file)
        for type in ['train', 'test']:
            for i, pair in enumerate(data[type]):
                pair_dict = {}
                for x in ['input', 'output']:
                    rgb_array = np.array([[colors_rgb[val] for val in row] for row in pair[x]])
                    # Expand each pixel to 10x10
                    expanded_rgb_array = np.repeat(np.repeat(rgb_array, 10, axis=0), 10, axis=1)
                    image = Image.fromarray(np.uint8(expanded_rgb_array))
                    image.save(f"{type}_{i}_{x}.png")
                    pair_dict[x] = encode_image(f"{type}_{i}_{x}.png")
                images[type].append(pair_dict)
    return images

def make_content(strings, images=None):
    content = [{
            'type': 'text',
            'text': "Demonstrations"
        }]
    
    becomes = {
        'type': 'text',
        'text': "becomes"
    }
    for type in ['train', 'test']:
        for i in range(len(strings[type])):
            for in_or_out in ['input', 'output']:
                text_dict = {
                    'type': 'text',
                    'text': f"""{in_or_out} grid {i}\n 
                            {strings[type][i][in_or_out]}\n"""

                }
                if images:
                    im_dict = {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{images[type][i][in_or_out]}"
                        }
                        }
                
                if not (type ==  'test' and in_or_out == 'output'):
                    content.append(text_dict)
                    if images:
                        content.append(im_dict)
                    if in_or_out == 'input':
                        content.append(becomes)
                    
    return content

def make_promt(content):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
        }

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
            "role": "system",
            "content": """You are a chatbot with human-like reasoning and inference capabilities, adept at solving tasks concisely.
                        Let's engage in reasoning and logic-based tasks. Each task will demonstrate a transformation from an input grid to an output
                        grid. At the end, you'll receive a new input grid. Your task is to determine its corresponding output grid and describe the transformation steps from the input grid."""
            },
            {
            "role": "user",
            "content": content
            }
        ]
        }
    return headers, payload

def ask_gpt(content):
    headers, payload = make_promt(content)
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    return(response.json())

def filter_string_test(solution_path):
    for filename in os.listdir(solution_path):
        if filename.endswith('.json'):
            # load the json file
            with open(os.path.join(solution_path, filename), 'r') as file:
                data = json.load(file)
                if not data['solution_0'] == filter_string(data['solution_0']):
                    print('Mismatch:')
                    print(data['solution_0'])
                    print('##############')
                    print(filter_string(data['solution_0']))
                    return False
    return True

def filter_string(input_string):
    # Define the valid color names
    valid_colors = [
        "Black", "White", "Red", "Green", "Blue", "Yellow", "Magenta",
        "Cyan", "Gray", "Maroon", "Olive", "Dark Green", "Turquoise", "Lavender"
    ]

    # Create a regex pattern to match valid colors and delimiters ("|")
    valid_pattern = r'^((' + '|'.join(map(re.escape, valid_colors)) + r')\|?)+$'

    # Split the input text into lines
    lines = input_string.split('\n')

    # Filter lines that only contain valid colors and delimiters
    grid_lines = [line for line in lines if re.match(valid_pattern, line.strip())]

    # Join the grid lines back into a single string
    grid_text = '\n'.join(grid_lines)+'\n'

    return grid_text
def test_solutions(solution_path):
    num_tasks = 0
    num_correct = 0
    for filename in os.listdir(solution_path):
    # check if the file is a json file
        if filename.endswith('.json'):
            num_tasks += 1
            # load the json file
            with open(os.path.join(solution_path, filename), 'r') as file:
                data = json.load(file)
                # iterate over all the keys in the json          
                num_correct += filter_string(data['response']).strip() == data['solution_0'].strip()
    return num_tasks, num_correct, num_correct/num_tasks

def solved_tasks(solution_path):
    solved_tasks = []
    for filename in os.listdir(solution_path):
        # check if the file is a json file
        if filename.endswith('.json'):
            # load the json file
            with open(os.path.join(solution_path, filename), 'r') as file:
                data = json.load(file)
                # iterate over all the keys in the json                
                solved_tasks.append(data['path'])
    return solved_tasks

def unsolved_tasks(data_path, solution_path, response_data=False):
    solved = solved_tasks(solution_path)
    if response_data:
        all_tasks = solved_tasks(data_path)
    else:
        all_tasks = os.listdir(data_path)
        all_tasks = [os.path.join(data_path, task) for task in all_tasks]

    unsolved_tasks = [task for task in all_tasks if task not in solved]
    return unsolved_tasks


def string_to_image(input_string):
    # Define the color mapping
    color_mapping = {
    "Black": (0, 0, 0),
    "White": (255, 255, 255),
    "Red": (255, 0, 0),
    "Green": (0, 128, 0),
    "Blue": (0, 0, 255),
    "Yellow": (255, 255, 0),
    "Magenta": (255, 0, 255),
    "Cyan": (0, 255, 255),
    "Gray": (128, 128, 128),
    "Maroon": (128, 0, 0),
    "Olive": (128, 128, 0),
    "Dark Green": (0, 100, 0),
    "Turquoise": (64, 224, 208),
    "Lavender": (230, 230, 250)
}

    # Parse the input string
    rows = input_string.strip().split('\n')
    grid = [row.split('|') for row in rows]

    # Determine the size of the grid
    grid_height = len(grid)
    grid_width = len(grid[0]) if grid else 0

    # Define the size of each cell
    cell_size = 20

    # Create a new image with the appropriate size
    img_width = grid_width * cell_size
    img_height = grid_height * cell_size
    image = Image.new("RGB", (img_width, img_height), "white")
    draw = ImageDraw.Draw(image)

    # Draw the grid
    for y, row in enumerate(grid):
        for x, color in enumerate(row):
            top_left = (x * cell_size, y * cell_size)
            bottom_right = ((x + 1) * cell_size, (y + 1) * cell_size)
            draw.rectangle([top_left, bottom_right], fill=color_mapping.get(color, (0, 0, 0)))

    # Save or show the image
    return image

def make_task_solution_viz(file_path, save_dir):
    # make a directory in save_dir with the file name (json)
    file_name = file_path.split('/')[-1].split('.')[0]
    save_path = os.path.join(save_dir, file_name)
    os.makedirs(save_path, exist_ok=True)
    with open(file_path, 'r') as file:
        data = json.load(file)
        task_json = json_to_string(data['path'])
        
        # Plot training images
        for i, training_pair in enumerate(task_json['train']):
            input_image = string_to_image(training_pair['input'])
            output_image = string_to_image(training_pair['output'])
            # save images in the dir
            input_image.save(os.path.join(save_path, f'train_{i}_input.png'))
            output_image.save(os.path.join(save_path, f'train_{i}_output.png'))
        # Plot test images
        for i, test_pair in enumerate(task_json['test']):
            input_image = string_to_image(test_pair['input'])
            output_image = string_to_image(test_pair['output'])
            response_image = string_to_image(filter_string(data['response']).strip())
            # save images in the dir
            input_image.save(os.path.join(save_path, f'test_{i}_input.png'))
            output_image.save(os.path.join(save_path, f'test_{i}_output.png'))
            response_image.save(os.path.join(save_path, f'test_{i}_response.png'))

if __name__ == '__main__':
    api_key = "<API_KEY>"
    data_path = '<DATA_PATH>'
    solutions_path = '<SOLUTIONS_PATH>'
    target_num_solutions = 50
    num_solved = 0
    while(num_solved < target_num_solutions):
        try:
            path = np.random.choice(unsolved_tasks(data_path, solutions_path, response_data=True), 1)[0]
            # images = get_encoded_images(path)
            strings = json_to_string(path)
            content = make_content(strings)
            response = ask_gpt(content)
            response_task = response['choices'][0]['message']['content']
            # save the response as json
            save_dict = {'path': path, 'response': response_task}
            for idx, solution in enumerate(strings['test']):
                solution_key = f'solution_{idx}'
                save_dict[solution_key] = solution['output']

            save_path = os.path.join(solutions_path, f"{path.split('/')[-1].split('.')[0]}_response.json")
            with open(save_path, 'w') as file:
                json.dump(save_dict, file)
            num_solved += 1
            print(f"{num_solved}/{target_num_solutions} Solved task: {path}")
            
        except:
            print("Error")
            print(json.dumps(response, indent=4))
            continue

    print(test_solutions(solutions_path))