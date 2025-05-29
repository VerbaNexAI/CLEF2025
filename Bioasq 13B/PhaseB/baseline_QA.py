# %%
import ast
import requests
import re
import os
import json
from dotenv import load_dotenv
import datetime
import pickle
import traceback
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import string
import time
import random
import threading
import logging
from requests.exceptions import RequestException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] [%(levelname)s] - %(message)s",
    handlers=[logging.FileHandler("openrouter_api.log"), logging.StreamHandler()],
)

# Thread-local storage for unique thread identification
thread_local = threading.local()

load_dotenv()


class RateLimiter:
    """Simple rate limiter to prevent hitting API rate limits."""

    def __init__(self, calls_per_minute=10):
        self.calls_per_minute = calls_per_minute
        self.interval = 60.0 / calls_per_minute  # Time between requests in seconds
        self.last_call_time = 0
        self.lock = threading.Lock()

    def wait(self):
        """Wait until it's safe to make another API call."""
        with self.lock:
            current_time = time.time()
            time_since_last_call = current_time - self.last_call_time

            if time_since_last_call < self.interval:
                sleep_time = self.interval - time_since_last_call
                logging.debug(f"Rate limiting: Sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)

            self.last_call_time = time.time()


# Create a global rate limiter
# OpenRouter recommends staying under 10 RPM for free tier
RATE_LIMITER = RateLimiter(calls_per_minute=8)  # Keep below 10 for safety


def str_to_list(input_string):
    """
    Converts a string representation of a list to an actual Python list
    using ast.literal_eval for safety.

    Args:
        input_string (str): The string to convert

    Returns:
        list: The converted list
    """
    # Handle empty or None input
    if not input_string:
        return []

    # Clean up input
    input_string = input_string.strip()

    # Try to handle JSON-like string (add brackets if needed)
    if not (input_string.startswith("[") and input_string.endswith("]")):
        # Handle comma-separated values without brackets
        if "," in input_string:
            input_string = f"[{input_string}]"

    try:
        # Use ast.literal_eval to safely evaluate the string as a Python literal
        result = ast.literal_eval(input_string)

        # Handle the case where the result is a dict with 'entities' key
        if isinstance(result, dict) and "entities" in result:
            return result["entities"]

        # If it's already a list, return it
        if isinstance(result, list):
            return result

        # If it's some other type, wrap it in a list
        return [result]

    except (SyntaxError, ValueError):
        # Fallback: split by commas if literal_eval fails
        if "," in input_string:
            return [item.strip().strip("'\"[]") for item in input_string.split(",")]

        # Last resort: split by whitespace
        return [item.strip().strip("'\"[]") for item in input_string.split()]


# %%
import requests
import re
import os
import json
from dotenv import load_dotenv
import datetime
import pickle
import traceback
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import string
import time
from requests.exceptions import RequestException

load_dotenv()


def get_openrouter_completion(messages, model_name, max_retries=3, retry_delay=2):
    """Get completion from OpenRouter API with retry mechanism

    Args:
        messages (list): List of message dictionaries
        model_name (str): Model name to use on OpenRouter
        max_retries (int): Maximum number of retries on failure
        retry_delay (int): Base delay between retries in seconds (will be exponentially increased)

    Returns:
        str: The completion text
    """
    # Get thread ID for logging
    thread_id = threading.current_thread().name

    # Get API key from environment variables
    # api_key = os.getenv("OPENROUTER_API_KEY")
    # if not api_key:
    #     logging.error(
    #         f"[{thread_id}] OPENROUTER_API_KEY not found in environment variables"
    #     )
    #     raise ValueError("OPENROUTER_API_KEY not found in environment variables")
    # # Log the request (but don't log full messages to avoid exposing sensitive data)
    # logging.info(
    #     f"[{thread_id}] Making request to OpenRouter API using model: {model_name}"
    # )

    api_key = (
        "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"
    )
    retries = 0
    while retries <= max_retries:
        try:
            logging.debug(f"[{thread_id}] Attempt {retries+1}/{max_retries+1}")

            # Apply rate limiting before making the request
            RATE_LIMITER.wait()

            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://bioasq2025.example.com",  # Optional: helps with analytics
                    "X-Title": "BioASQ Question Answering",  # Optional: helps with analytics
                },
                json={
                    "model": model_name,
                    "messages": messages,
                    "temperature": 0.0,  # Low temperature for deterministic outputs
                },
                timeout=60,  # Add timeout to prevent hanging requests
            )

            # Check if the request was successful
            if response.status_code == 200:
                response_data = response.json()
                completion_text = response_data["choices"][0]["message"]["content"]
                # Log successful response (truncated)
                preview = (
                    completion_text[:50] + "..."
                    if len(completion_text) > 50
                    else completion_text
                )
                logging.info(f"[{thread_id}] Received successful response: {preview}")
                return completion_text

            # Handle rate limiting (429) or server errors (5xx)
            if response.status_code in [429, 500, 502, 503, 504]:
                retries += 1
                if retries > max_retries:
                    break

                # Exponential backoff with jitter
                delay = retry_delay * (2**retries) + (0.1 * random.random())
                logging.warning(
                    f"[{thread_id}] Rate limited or server error ({response.status_code}). Retrying in {delay:.2f} seconds..."
                )
                time.sleep(delay)
                continue

            # Other errors
            logging.error(
                f"[{thread_id}] Request failed with status code {response.status_code}: {response.text}"
            )
            response.raise_for_status()

        except RequestException as e:
            retries += 1
            if retries > max_retries:
                break

            delay = retry_delay * (2**retries)
            logging.warning(
                f"[{thread_id}] Request exception: {str(e)}. Retrying in {delay} seconds..."
            )
            time.sleep(delay)

    # If we've exhausted retries
    error_message = f"OpenRouter API error after {max_retries} retries: {response.status_code} - {response.text}"
    logging.error(f"[{thread_id}] {error_message}")
    raise Exception(error_message)


def read_json_file(file_path):
    """Reads a JSON file and returns the parsed data."""
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


def read_jsonl_as_json(file_path):
    """Reads a JSONL file but treats it as JSON if needed."""
    try:
        # First try reading as JSON
        return read_json_file(file_path)
    except json.JSONDecodeError:
        # If that fails, try reading as JSONL
        examples = []
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                examples.append(json.loads(line))
        return examples


def remove_punctuation_and_lowercase(text):
    # Lowercase the string
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text[:4]
    text = text.strip()
    return text


def get_completion(messages, model_name):
    """Get completion from OpenRouter API"""
    return get_openrouter_completion(messages, model_name)


# %%
def extract_snippets(snippets):
    # Check if snippets is already a list of dictionaries with 'text' key
    if isinstance(snippets, list) and all(
        isinstance(s, dict) and "text" in s for s in snippets
    ):
        return [snippet["text"] for snippet in snippets]
    # If it's a single dictionary with 'text' key
    elif isinstance(snippets, dict) and "text" in snippets:
        return [snippets["text"]]
    # If it's already a list of strings
    elif isinstance(snippets, list) and all(isinstance(s, str) for s in snippets):
        return snippets
    # If we can't determine the structure, return an empty list
    return []


# %%
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("neuml/pubmedbert-base-embeddings")


def find_similar_snippets(
    question, snippets, model_name="neuml/pubmedbert-base-embeddings", top_k=5
):
    """
    Find snippets that are most similar to a given question.

    Parameters:
    question (str): The question text to compare against
    snippets (list): A list of snippet texts
    model_name (str): The name of the SentenceTransformer model to use
    top_k (int): Number of top similar snippets to return

    Returns:
    list: A list of tuples containing (similarity_score, snippet_text, snippet_index)
    """
    # Load the model

    # Create a list with question as the first item followed by all snippets
    all_texts = [question] + snippets

    # Generate embeddings for all texts
    embeddings = model.encode(all_texts)

    # Get the question embedding (first item) and snippet embeddings (rest of items)
    question_embedding = embeddings[0].reshape(1, -1)  # Reshape for cosine_similarity
    snippet_embeddings = embeddings[1:]

    # Calculate cosine similarity between question and each snippet
    similarities = cosine_similarity(question_embedding, snippet_embeddings)[0]

    # Create list of (similarity, snippet_text, snippet_index) tuples
    similarity_results = [
        (similarities[i], snippets[i], i) for i in range(len(snippets))
    ]

    # Sort by similarity score in descending order
    similarity_results.sort(reverse=True)

    # Return top_k results (or all if less than top_k)
    return similarity_results[:top_k]


# %%


# Example with your BioASQ data:
def get_relevant_snippets(question, snippet_list, top_k=5):
    """
    Extract the most relevant snippets for a given question.

    Parameters:
    question (str): The question text
    snippet_list (list): List of snippet dictionaries with 'text' key
    top_k (int): Number of top similar snippets to return

    Returns:
    list: The most relevant snippets
    """
    # Extract just the text from the snippets
    snippet_texts = extract_snippets(snippet_list)

    # Find the most similar snippets
    similar_snippets = find_similar_snippets(
        question=question, snippets=snippet_texts, top_k=top_k
    )

    # Return the relevant snippets with their metadata intact
    relevant_snippets = []
    for _, _, idx in similar_snippets:
        relevant_snippets.append(snippet_list[idx])

    return relevant_snippets


# %%


def generate_exact_answer(question, snippets, n_shots, examples, model_name):
    exact_answer = []
    system_message = {
        "role": "system",
        "content": """You are a specialized biomedical expert AI assistant with deep knowledge in medical research, clinical practice, and scientific literature.""",
    }
    messages = [system_message]

    # Generate and add the few-shot examples to the message list
    if n_shots > 0 and examples:
        few_shot_examples = generate_n_shot_examples(examples, n_shots)
        messages.extend(few_shot_examples)

    # Get the snippets and body from the question directly
    snippet_texts = extract_snippets(snippets)

    relevant_snippets = get_relevant_snippets(
        question=question["body"],
        snippet_list=snippet_texts,
        top_k=5,  # Adjust as needed
    )

    # Define log file
    log_file_name = "exact_answer_log.txt"

    # Construct the final user message for the actual question
    if question["type"] == "yesno":
        user_message = {
            "role": "user",
            "content": f"""Based on the following biomedical research snippets:

{relevant_snippets}

Question: {question['body']}

Instructions:
1. Analyze the provided snippets carefully
2. Determine if the evidence supports a 'yes' or 'no' answer
3. Respond with ONLY the word 'yes' or 'no'
4. Do not include any additional text or explanation""",
        }
        messages.append(user_message)
        # Log the prompt
        print("\n======= YESNO PROMPT =======")
        print(json.dumps(messages, indent=2))
        print("===========================\n")

        answer = get_completion(messages, model_name)
        # Log the raw response
        print("\n======= YESNO RESPONSE =======")
        print(answer)
        print("==============================\n")

        exact_answer = remove_punctuation_and_lowercase(answer)
        # Log the processed answer
        print("\n======= YESNO EXACT ANSWER =======")
        print(exact_answer)
        print("=================================\n")

        # Save to log file
        with open(log_file_name, "a", encoding="utf-8") as f:
            f.write(f"--- Question ID: {question['id']} ---\n")
            f.write(f"--- Question Type: {question['type']} ---\n")
            f.write("--- Question ---\n")
            f.write(f"{question['body']}\n\n")
            f.write("--- Prompt ---\n")
            f.write(f"{json.dumps(messages, indent=2)}\n\n")
            f.write("--- LLM Answer ---\n")
            f.write(f"{answer}\n\n")
            f.write("--- Processed Exact Answer ---\n")
            f.write(f"{exact_answer}\n")
            f.write("=" * 40 + "\n\n")

    elif question["type"] == "factoid":
        user_message = {
            "role": "user",
            "content": f"""Based on the following biomedical research snippets:

{relevant_snippets}

Question: {question['body']}

Instructions:
1. Analyze the provided snippets to identify relevant entities
2. Extract up to 5 most relevant entities, ordered by confidence
3. Format your response as a JSON object with an 'entities' array
4. Each entity should be a short expression (name, number, or term)
5. Example format: {{"entities": ["entity1", "entity2", "entity3", "entity4", "entity5"]}}""",
        }
        messages.append(user_message)
        # Log the prompt
        print("\n======= FACTOID PROMPT =======")
        print(json.dumps(messages, indent=2))
        print("============================\n")

        answer = get_completion(messages, model_name)
        # Log the raw response
        print("\n======= FACTOID RESPONSE =======")
        print(answer)
        print("===============================\n")

        try:
            factoids = json.loads(answer)
            wrapped_list = [[item] for item in factoids["entities"]]
            exact_answer = wrapped_list
        except json.JSONDecodeError:
            print("ERROR: Failed to parse JSON response from model")
            print(f"Raw response: {answer}")
            exact_answer = []

        # Log the processed answer
        print("\n======= FACTOID EXACT ANSWER =======")
        print(exact_answer)
        print("====================================\n")

        # Save to log file
        with open(log_file_name, "a", encoding="utf-8") as f:
            f.write(f"--- Question ID: {question['id']} ---\n")
            f.write(f"--- Question Type: {question['type']} ---\n")
            f.write("--- Question ---\n")
            f.write(f"{question['body']}\n\n")
            f.write("--- Prompt ---\n")
            f.write(f"{json.dumps(messages, indent=2)}\n\n")
            f.write("--- LLM Answer ---\n")
            f.write(f"{answer}\n\n")
            f.write("--- Processed Exact Answer ---\n")
            f.write(f"{json.dumps(exact_answer, indent=2)}\n")
            f.write("=" * 40 + "\n\n")

    elif question["type"] == "list":
        user_message = {
            "role": "user",
            "content": f"""Based on the following biomedical research snippets:

{relevant_snippets}

Question: {question['body']}

Instructions:
1. Analyze the provided snippets to identify all relevant items
2. Create a comprehensive list of relevant entities
3. Each item should be:
   - A short expression (max 100 characters)
   - Relevant to the question
   - Based on the provided evidence
4. Return ONLY a JSON array of strings
5. Maximum 100 items allowed
6. Example format: ["item1", "item2", "item3"]""",
        }
        messages.append(user_message)
        # Log the prompt
        print("\n======= LIST PROMPT =======")
        print(json.dumps(messages, indent=2))
        print("=========================\n")

        answer = get_completion(messages, model_name)
        # Log the raw response
        print("\n======= LIST RESPONSE =======")
        print(answer)
        print("===========================\n")

        exact_answer = str_to_list(answer)
        # Log the processed answer
        print("\n======= LIST EXACT ANSWER =======")
        print(exact_answer)
        print("===============================\n")

        # Save to log file
        with open(log_file_name, "a", encoding="utf-8") as f:
            f.write(f"--- Question ID: {question['id']} ---\n")
            f.write(f"--- Question Type: {question['type']} ---\n")
            f.write("--- Question ---\n")
            f.write(f"{question['body']}\n\n")
            f.write("--- Prompt ---\n")
            f.write(f"{json.dumps(messages, indent=2)}\n\n")
            f.write("--- LLM Answer ---\n")
            f.write(f"{answer}\n\n")
            f.write("--- Processed Exact Answer ---\n")
            f.write(
                f"{json.dumps(exact_answer, indent=2)}\n"
            )  # Save list as JSON string
            f.write("=" * 40 + "\n\n")

    return exact_answer


# %%


def generate_n_shot_examples(examples, n_shots):
    """
    Generate n-shot examples using exact_answer for specific question types
    (yesno, list, factoid) to be used in prompts for the LLM.

    Parameters:
        examples (list): List of example dictionaries from the dataset,
                         ideally pre-filtered by type.
        n_shots (int): Number of examples to include.

    Returns:
        list: List of formatted messages (user/assistant pairs) for n-shot learning.
    """
    # Select the first n_shots examples (assuming they are relevant type)
    selected_examples = examples[:n_shots]

    # Format the examples as messages for the prompt
    messages = []

    for i, example in enumerate(selected_examples):
        # Extract snippets
        snippet_texts = []
        if "snippets" in example:
            snippet_texts = extract_snippets(example["snippets"])
            # Limit to first 3 snippets for brevity in the prompt
            snippet_texts = snippet_texts[:3]

        # Format the example as a user message
        user_message = {
            "role": "user",
            "content": f"RELEVANT SNIPPETS: {snippet_texts}\n\nQUESTION: '{example['body']}'.",
        }

        # Initialize assistant message
        assistant_message = {
            "role": "assistant",
            "content": "",  # Default empty content
        }

        # Get the type of the current example
        example_type = example.get("type")

        # Format the assistant's response based on the EXAMPLE's type using exact_answer
        if example_type == "yesno":
            # For yes/no questions, the answer is simply 'yes' or 'no'
            assistant_message["content"] = example.get("exact_answer", "")

        elif example_type == "list":
            # For list questions, format as a JSON list of entities
            if "exact_answer" in example:
                exact_answer = example["exact_answer"]
                # Handle potential variations if needed, assuming it's already a list or list-like
                if isinstance(exact_answer, list):
                    # Ensure items are strings if needed, basic formatting
                    entities = [str(item) for item in exact_answer]
                    # Format as JSON array string
                    try:
                        assistant_message["content"] = json.dumps(entities)
                    except TypeError:
                        # Fallback if items aren't directly serializable
                        assistant_message["content"] = json.dumps(
                            [str(e) for e in entities]
                        )

        elif example_type == "factoid":
            # For factoid questions, format as a JSON list of entities (limit if necessary)
            if "exact_answer" in example:
                exact_answer = example["exact_answer"]
                if isinstance(exact_answer, list):
                    # Assuming exact_answer for factoids might be nested [[item1], [item2]]
                    # Flatten if necessary, or adjust based on actual data structure
                    entities = []
                    for item in exact_answer:
                        if isinstance(item, list) and len(item) > 0:
                            entities.append(str(item[0]))
                        elif not isinstance(item, list):
                            entities.append(str(item))
                    # Limit to 5 as per original logic, though this might be better handled in data prep
                    entities = entities[:5]
                    # Format as JSON object string as per original 'factoid' prompt instructions
                    try:
                        assistant_message["content"] = json.dumps(
                            {"entities": entities}
                        )
                    except TypeError:
                        assistant_message["content"] = json.dumps(
                            {"entities": [str(e) for e in entities]}
                        )

        # Only add messages if the assistant content could be formatted
        if assistant_message["content"]:
            messages.append(user_message)
            messages.append(assistant_message)

    return messages


# %%


def generate_ideal_answer(question, snippets, n_shots, examples, model_name):
    system_message = {
        "role": "system",
        "content": """You are BioASQ-GPT, a biomedical expert. Your task is to provide clear, concise answers to biomedical questions. Follow these rules:

        1. Answer Format:
        - Write a single paragraph
        - Maximum 200 words
        - Use clear medical terminology
        """,
    }
    messages = [system_message]
    snippet_texts = extract_snippets(snippets)
    relevant_snippets = get_relevant_snippets(
        question=question["body"],
        snippet_list=snippet_texts,
        top_k=5,  # Adjust as needed
    )

    # Generate few-shot examples (using the updated function)
    # Note: These examples use exact_answer. If ideal_answer examples are needed
    # for the summary task, a different approach/function would be required.
    # few_shot_examples = generate_n_shot_examples(examples, n_shots)
    # print("question", examples) # Keep for debugging if needed
    # print("few_shot_examples", few_shot_examples) # Keep for debugging if needed

    # *** IMPORTANT: Decide if/how to include few_shot_examples in the prompt ***
    # If you want to use them for the ideal_answer task:
    # messages.extend(few_shot_examples)
    # However, using exact_answer examples for an ideal_answer task might confuse the model.
    # It might be better to omit few-shot examples here or create specific summary examples.

    user_message = {
        "role": "user",
        "content": f"""RELEVANT SNIPPETS: {relevant_snippets}\n\n
             QUESTION: '{question['body']}'.
             Write a single paragraph-sized text ideally summarizing the most relevant information from snippets retrieved.
             """,
    }
    messages.append(user_message)

    # Log the prompt
    print("\n======= IDEAL ANSWER PROMPT =======")
    print(json.dumps(messages, indent=2))
    print("=================================\n")

    answer = get_completion(messages, model_name)

    # Log the raw response
    print("\n======= IDEAL ANSWER RESPONSE =======")
    print(answer)
    print("===================================\n")

    # Define log file
    log_file_name = "ideal_answer_log.txt"

    # Save to log file
    with open(log_file_name, "a", encoding="utf-8") as f:
        f.write(f"--- Question ID: {question['id']} ---\n")
        f.write(f"--- Question Type: {question['type']} (Ideal Answer) ---\n")
        f.write("--- Question ---\n")
        f.write(f"{question['body']}\n\n")
        f.write("--- Prompt ---\n")
        f.write(f"{json.dumps(messages, indent=2)}\n\n")
        f.write("--- LLM Answer ---\n")
        f.write(f"{answer}\n")
        f.write("=" * 40 + "\n\n")

    return answer


# %%


def save_state(data, file_path):
    """Save the current state to a pickle file."""
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def load_state(file_path):
    """Load the state from a pickle file if it exists, otherwise return None."""
    try:
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                return pickle.load(f)
    except EOFError:  # Handles empty pickle file scenario
        return None
    return None


def get_examples_by_type(examples, question_type):
    # This function is no longer needed after pre-filtering,
    # but we'll keep it for now in case it's used elsewhere,
    # or remove it later if confirmed unused.
    examples_by_type = []
    for example in examples:
        if example["type"] == question_type:
            examples_by_type.append(example)
    return examples_by_type


def process_question(question, examples_by_type_dict, n_shots, model_name):
    question_type = question["type"]

    # Get the relevant articles and snippets
    relevant_snippets = question.get("snippets", [])

    # Get the pre-filtered list of examples for the current question type
    # Use .get(question_type, []) to safely handle cases where a type might be missing in examples
    type_specific_examples = examples_by_type_dict.get(question_type, [])

    # Generate the exact answer and ideal answer
    try:
        exact_answer = generate_exact_answer(
            question,
            relevant_snippets,
            n_shots,
            # Pass the pre-filtered list directly
            type_specific_examples,
            model_name,
        )
        ideal_answer = generate_ideal_answer(
            question,
            relevant_snippets,
            n_shots,
            # Pass the pre-filtered list directly
            type_specific_examples,
            model_name,
        )
    except Exception as e:
        print(f"Error processing question {question['id']}: {e}")
        traceback.print_exc()
        exact_answer = []
        ideal_answer = []

    # Create a dictionary to store the results for this question
    question_results = {
        "id": question["id"],
        "type": question_type,
        "body": question["body"],
        "documents": question.get("documents", []),
        "snippets": relevant_snippets,
        "ideal_answer": ideal_answer,
        "exact_answer": exact_answer,
    }

    return question_results


# %%
# set the example
example_try = False
example_number = 10

# Update to OpenRouter model - using qwen2.5-32b-instruct
model_name = "qwen/qwen3-235b-a22b"

n_shots = 2  # Using fewer shots for testing


# Get the current timestamp in a sortable format
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Configure file paths
pickl_name = model_name.replace("/", "-")
pickl_file = f"{pickl_name}-{n_shots}-shot.pkl"

# Load the training data
with open("./training13b.json", encoding="utf-8") as input_file:
    data = json.load(input_file)
    if example_try:
        data["questions"] = data["questions"][
            :example_number
        ]  # Only process 4 items for testing
    else:
        data["questions"] = data["questions"][:]  # process all the data

# Load the example data from example.json instead of JSONL file
example_file = "./example.json"
examples_data = read_json_file(example_file)  # Use existing read_json_file function
# Extract the list of questions from the loaded data structure
# Assuming the structure is {"questions": [...]}
example_list = examples_data.get("questions", [])
if not example_list:
    print(f"Warning: No 'questions' found in {example_file} or the list is empty.")

# --- Optimization: Pre-filter examples by type ---
examples_by_type_dict = {
    "yesno": [],
    "list": [],
    "factoid": [],
    "summary": [],
    # Add other types if they exist in example.json
}
for ex in example_list:
    q_type = ex.get("type")
    if q_type in examples_by_type_dict:
        examples_by_type_dict[q_type].append(ex)
    else:
        # Handle unexpected types if necessary
        print(f"Warning: Example {ex.get('id')} has unexpected type: {q_type}")
# --- End Optimization ---

# Define columns for the DataFrame
columns = [
    "id",
    "body",
    "type",
    "documents",
    "snippets",
    "ideal_answer",
    "exact_answer",
]

# Initialize empty DataFrame
questions_df = pd.DataFrame(columns=columns)

saved_df = load_state(pickl_file)

if saved_df is not None and not saved_df.empty:
    processed_ids = set(saved_df["id"])
    questions_df = saved_df
else:
    processed_ids = set()

questions_to_process = [q for q in data["questions"] if q["id"] not in processed_ids]

# Filter questions to process
questions_to_process = [q for q in data["questions"] if q["id"] not in processed_ids]

# Process questions in parallel using ThreadPoolExecutor
print(f"Processing {len(questions_to_process)} questions in parallel...")
max_workers = min(
    4, len(questions_to_process)
)  # Limit concurrent requests to avoid rate limits
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # Create a future for each question
    future_to_question = {
        executor.submit(
            process_question, question, examples_by_type_dict, n_shots, model_name
        ): question
        for question in questions_to_process
    }

    # Process completed futures as they complete
    for future in as_completed(future_to_question):
        question = future_to_question[future]
        try:
            result = future.result()
            if result:
                # Append result to the DataFrame
                result_df = pd.DataFrame([result])
                questions_df = pd.concat([questions_df, result_df], ignore_index=True)
                save_state(questions_df, pickl_file)
                print(f"Question {question['id']} processed successfully")
        except Exception as e:
            print(f"Error processing question {question['id']}: {e}")
            traceback.print_exc()

# Create output filename
model_name_pretty = model_name.split("/")[-1]
output_file_name = f"./Results/{timestamp}_{model_name_pretty}-{n_shots}-QA.csv"

# Ensure the directory exists before saving
os.makedirs(os.path.dirname(output_file_name), exist_ok=True)

questions_df.to_csv(output_file_name, index=False)

# Clean up the pickle file
try:
    if os.path.exists(pickl_file):
        os.remove(pickl_file)
        print("Intermediate state pickle file deleted successfully.")
except Exception as e:
    print(f"Error deleting pickle file: {e}")
    traceback.print_exc()

print(f"Results saved to {output_file_name}")


# %%
import pandas as pd
import json


def csv_to_json(csv_filepath, json_filepath):
    # Step 1: Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_filepath)

    # Transform the DataFrame into a list of dictionaries, one per question
    questions_list = df.to_dict(orient="records")

    # Initialize the structure of the JSON file
    json_structure = {"questions": []}

    # Step 2: Transform the DataFrame into the desired JSON structure
    for item in questions_list:
        print("---" * 10)

        # Process snippets to ensure they have the correct format
        snippet_list = eval(item["snippets"])[:10]
        formatted_snippets = []
        for snippet in snippet_list:
            if isinstance(snippet, dict) and "text" in snippet:
                # Format the snippet according to the required structure
                formatted_snippet = {
                    "document": snippet.get("document", ""),
                    "text": snippet.get("text", ""),
                    "offsetInBeginSection": snippet.get("offsetInBeginSection", 0),
                    "offsetInEndSection": snippet.get("offsetInEndSection", 0),
                    "beginSection": snippet.get("beginSection", "sections.0"),
                    "endSection": snippet.get("endSection", "sections.0"),
                }
                formatted_snippets.append(formatted_snippet)
            elif isinstance(snippet, str):
                # If snippet is just a string, create a minimal structure
                formatted_snippet = {
                    "document": "",
                    "text": snippet,
                    "offsetInBeginSection": 0,
                    "offsetInEndSection": len(snippet),
                    "beginSection": "sections.0",
                    "endSection": "sections.0",
                }
                formatted_snippets.append(formatted_snippet)

        # Create the base question dictionary
        question_dict = {
            "documents": eval(item["documents"])[:10],
            "snippets": formatted_snippets,
            "body": item["body"],
            "type": item["type"],
            "id": item["id"],
            "ideal_answer": item["ideal_answer"],
        }

        # Process exact_answer based on question type
        if item["type"] == "yesno":
            yesno_answer = item["exact_answer"].replace(" ", "")
            yesno_answer = yesno_answer.lower()

            if yesno_answer not in ["yes", "no"]:
                yesno_answer = "no"
            question_dict["exact_answer"] = yesno_answer

        elif item["type"] == "factoid":

            # For factoid questions, exact_answer should be a list of lists
            factoid_answers = eval(item["exact_answer"])[:5]
            # Ensure each item is in a nested list
            formatted_answers = [
                [item] if not isinstance(item, list) else item
                for item in factoid_answers
            ]
            question_dict["exact_answer"] = formatted_answers

        elif item["type"] == "list":

            # For list questions, exact_answer should be a list
            list_items = eval(item["exact_answer"])[:100]
            question_dict["exact_answer"] = list_items

        json_structure["questions"].append(question_dict)

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(json_filepath), exist_ok=True)

    # Step 3: Write the JSON structure to a file
    with open(json_filepath, "w", encoding="utf-8") as json_file:
        json.dump(json_structure, json_file, ensure_ascii=False, indent=4)

    return json_structure


# Example usage
csv_filepath = output_file_name  # Update this path to your actual CSV file path
json_filepath = f"./Results/bioasq_results.json"  # Standard output file name
json_structure = csv_to_json(csv_filepath, json_filepath)


# %%
def print_json_structure(json_structure):
    # The results.json should keep the exact same structure as the input
    output_json = json_structure

    # Define output path in Results directory
    results_path = f"./Results/results_{timestamp}.json"

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, ensure_ascii=False, indent=4)

    print(f"Results saved to {results_path}")

    return output_json["questions"]


print_json_structure(json_structure)

# %%
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity

# # Sample biomedical question
# question = "Is Hirschsprung disease a mendelian or a multifactorial disorder?"

# # Sample snippets - these would typically come from your dataset
# snippets = [
#     {
#         'text': 'Hirschsprung disease (HSCR) is a multifactorial, non-mendelian disorder in which rare high-penetrance coding sequence mutations in the receptor tyrosine kinase RET contribute to risk in combination with mutations at other genes.',
#         'document': 'http://www.ncbi.nlm.nih.gov/pubmed/15829955'
#     },
#     {
#         'text': 'The majority of the identified genes are related to Mendelian syndromic forms of Hirschsprung\'s disease.',
#         'document': 'http://www.ncbi.nlm.nih.gov/pubmed/15617541'
#     },
#     {
#         'text': 'The inheritance of Hirschsprung disease is generally consistent with sex-modified multifactorial inheritance with a lower threshold of expression in males.',
#         'document': 'http://www.ncbi.nlm.nih.gov/pubmed/6650562'
#     },
#     {
#         'text': 'For almost all of the identified HSCR genes incomplete penetrance of the HSCR phenotype has been reported, probably due to modifier loci.',
#         'document': 'http://www.ncbi.nlm.nih.gov/pubmed/12239580'
#     },
#     {
#         'text': 'Chromosomal and related Mendelian syndromes associated with Hirschsprung\'s disease.',
#         'document': 'http://www.ncbi.nlm.nih.gov/pubmed/23001136'
#     },
#     {
#         'text': 'In the etiology of Hirschsprung disease various genes play a role; these are: RET, EDNRB, GDNF, EDN3 and SOX10, NTN3, ECE1, Mutations in these genes may result in dominant, recessive or multifactorial patterns of inheritance.',
#         'document': 'http://www.ncbi.nlm.nih.gov/pubmed/15858239'
#     },
# ]

# # Extract just the text from the snippets for embedding
# snippet_texts = extract_snippets(snippets)

# # Use find_similar_snippets directly to get the similarity scores
# top_k = 3
# similar_snippets = find_similar_snippets(
#     question=question,
#     snippets=snippet_texts,
#     top_k=top_k
# )

# # Print the results with similarity scores
# print(f"Question: {question}\n")
# print(f"Top {top_k} most relevant snippets with similarity scores:\n")

# # Create a list to store relevant snippets for QA pipeline
# relevant_snippets = []

# for i, (similarity, text, idx) in enumerate(similar_snippets):
#     # Get the corresponding snippet with metadata
#     snippet = snippets[idx]
#     relevant_snippets.append(snippet)

#     # Print the snippet with its similarity score
#     print(f"{i+1}. Similarity score: {similarity:.4f}")
#     print(f"   Text: {snippet['text']}")
#     print(f"   Source: {snippet['document']}\n")

# # Optional: Show how this would be used in a QA pipeline
# print("Example usage in a question answering pipeline:")
# print("-" * 60)
# # Format snippets for model input
# snippet_text_for_model = "\n\n".join([s['text'] for s in relevant_snippets])

# model_input = f"""
# Based on the following information:
# {snippet_text_for_model}

# Question: {question}
# Answer:
# """
# print(model_input)


# %%
# Test function to directly test generate_exact_answer with logging
def test_generate_exact_answer():
    """
    Test the generate_exact_answer function with a sample question
    and print out the prompt and responses.
    """
    print("Running test for generate_exact_answer...")

    # Load a test question from the dataset
    try:
        with open(
            "./BioASQ-task13bPhaseB-testset1.json", encoding="utf-8"
        ) as input_file:
            data = json.load(input_file)
            # Choose a question of each type for testing
            test_questions = []
            type_counts = {"factoid": 0, "list": 0, "yesno": 0}

            for q in data["questions"]:
                q_type = q.get("type")
                if q_type in type_counts and type_counts[q_type] < 1:
                    test_questions.append(q)
                    type_counts[q_type] += 1

                if sum(type_counts.values()) >= 3:
                    break

            if not test_questions:
                # Fallback to first question if no suitable questions found
                test_questions = [data["questions"][0]]
    except Exception as e:
        print(f"Error loading test questions: {e}")
        # Create a dummy question as fallback
        test_questions = [
            {
                "id": "test-id",
                "type": "factoid",
                "body": "What proteins are involved in the EGF receptor signaling pathway?",
                "snippets": [
                    {
                        "text": "The EGF receptor signaling pathway involves EGFR, Grb2, SOS, Ras, Raf, MEK, and ERK proteins."
                    },
                    {
                        "text": "Key proteins in EGFR signaling include PI3K, AKT, and mTOR, which activate cell survival pathways."
                    },
                ],
            }
        ]

    # Load examples for few-shot learning
    try:
        example_file = "./example.json"
        examples_data = read_json_file(example_file)
        example_list = examples_data.get("questions", [])

        # Pre-filter examples by type
        examples_by_type_dict = {
            "yesno": [],
            "list": [],
            "factoid": [],
        }
        for ex in example_list:
            q_type = ex.get("type")
            if q_type in examples_by_type_dict:
                examples_by_type_dict[q_type].append(ex)
    except Exception as e:
        print(f"Error loading examples: {e}")
        examples_by_type_dict = {"yesno": [], "list": [], "factoid": []}

    # Set parameters
    n_shots = 1
    model_name = "qwen/qwen3-235b-a22b"  # Adjust as needed

    # Process each test question
    for test_question in test_questions:
        q_type = test_question.get("type")
        print(f"\n\n{'='*40}")
        print(f"Testing question type: {q_type}")
        print(f"Question: {test_question['body']}")
        print(f"{'='*40}\n")

        # Get snippets
        snippets = test_question.get("snippets", [])

        # Get type-specific examples
        type_specific_examples = examples_by_type_dict.get(q_type, [])

        # Call the function
        try:
            result = generate_exact_answer(
                question=test_question,
                snippets=snippets,
                n_shots=n_shots,
                examples=type_specific_examples,
                model_name=model_name,
            )

            print("\n======= FINAL RESULT =======")
            if isinstance(result, list):
                print(json.dumps(result, indent=2))
            else:
                print(result)
            print("===========================\n")
        except Exception as e:
            print(f"Error running generate_exact_answer: {e}")
            import traceback

            traceback.print_exc()


# Uncomment the next line to run the test directly when this file is executed
# %%
# test_generate_exact_answer()

# %%
