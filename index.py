import os
import time
import pandas as pd
import random
import speech_recognition as sr
import pyttsx3
from transformers import RobertaTokenizer, RobertaForSequenceClassification, GPT2Tokenizer, GPT2LMHeadModel
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import f1_score, precision_score, recall_score
import threading
import keyboard

# Disable oneDNN optimizations in TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize models and tokenizers
classification_model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=1)
classification_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()

def play_beep():
    # For Windows
    import winsound
    frequency = 1000  # Set Frequency To 1000 Hertz
    duration = 500  # Set Duration To 500 ms == 0.5 second
    winsound.Beep(frequency, duration)

    # For Unix-based systems (comment out the above line and uncomment below)
    # os.system('echo -e "\a"')  # Use a bell character for beep

# Cross-encoder similarity function using Sentence-BERT
def cross_encoder_similarity_sbert(question, candidate_answer, model):
    embeddings = model.encode([question, candidate_answer])
    score = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return score.item()

# Provide feedback based on the score
def provide_feedback(score, threshold_good=0.8, threshold_average=0.5):
    feedback_messages = {
        'good': "Great answer! You've covered all the key points.",
        'average': "Decent answer, but there's room for improvement.",
        'poor': "The answer is lacking. Try to be more specific and cover the main points."
    }

    if score >= threshold_good:
        return feedback_messages['good']
    elif score >= threshold_average:
        return feedback_messages['average']
    else:
        return feedback_messages['poor']

# Ensure the answer meets minimum word count
def meets_min_word_count(answer, min_word_count=10):
    return len(answer.split()) >= min_word_count

# Load data based on the category
def load_data(category):
    file_paths = {
        "frontend": r'C:\Users\achib\OneDrive\Desktop\sih\DataSets\frontend_difficulty.csv',
        "backend": r'C:\Users\achib\OneDrive\Desktop\sih\DataSets\java_difficulty.csv'
    }
    
    file_path = file_paths.get(category)
    
    if not file_path or not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file for category '{category}' not found.")
    
    return pd.read_csv(file_path)

# Get questions by difficulty level
def get_questions_by_difficulty(data, difficulty):
    if 'difficulty' not in data.columns:
        raise ValueError("The column 'difficulty' is not present in the DataFrame.")
    return data[data['difficulty'].str.lower() == difficulty.lower()]

# Evaluate the model using multiple metrics
def multi_metric_evaluation(y_true, y_pred):
    if not y_true or not y_pred:
        return 0, 0, 0

    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    return f1, precision, recall

# Voice recognition setup
def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening...")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print(f"Voice input: {text}")
        return text
    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
        return ""
    except sr.RequestError:
        print("Could not request results; check your network connection.")
        return ""

# Function to read the question aloud
def speak_text(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

# Function to handle answering
def answer_question(timeout=60):
    answer = []
    print(f"Press 'A' to start answering and 'S' to submit. You have {timeout} seconds.")
    
    def record_answer():
        nonlocal answer
        while True:
            if keyboard.is_pressed('A'):
                print("Start speaking now...")
                ans = get_voice_input()
                answer.append(ans)
            if keyboard.is_pressed('S'):
                print("Answer submitted.")
                break
            time.sleep(0.1)

    answer_thread = threading.Thread(target=record_answer)
    answer_thread.start()
    
    answer_thread.join(timeout)
    if answer_thread.is_alive():
        print("Time's up!")
        return ' '.join(answer)
    return ' '.join(answer)

# Conduct an interview with immediate feedback
def conduct_interview_immediate_feedback(category, model, tokenizer, sbert_model, num_questions=3, random_state=None):
    try:
        data = load_data(category)
    except Exception as e:
        print(f"Failed to load data: {e}")
        return 0  # Return 0 if data loading fails

    try:
        easy_questions = get_questions_by_difficulty(data, 'easy')
        medium_questions = get_questions_by_difficulty(data, 'medium')
        hard_questions = get_questions_by_difficulty(data, 'hard')
    except Exception as e:
        print(f"Error in filtering questions by difficulty: {e}")
        return 0  # Return 0 if there is an error in filtering questions

    if easy_questions.empty and medium_questions.empty and hard_questions.empty:
        print("No questions available for the selected category and difficulty levels.")
        return 0  # Return 0 if no questions are available

    total_score = 0
    y_true = []
    y_pred = []

    difficulty_rounds = {'easy': 1, 'medium': 1, 'hard': 1}  # Total 3 questions

    for difficulty in ['easy', 'medium', 'hard']:
        questions = easy_questions if difficulty == 'easy' else medium_questions if difficulty == 'medium' else hard_questions
        for _ in range(difficulty_rounds[difficulty]):
            if questions.empty:
                break
            question = questions.sample(n=1).iloc[0]
            question_text = question['Question']
            expected_answer = question['Answer']

            # Print the question in the terminal
            print(f"Question: {question_text}")

            # Read the question aloud
            speak_text(question_text)
            
            # Provide a brief pause to allow the user to prepare
            time.sleep(2)  # Adjust this pause duration as needed

            candidate_answer = answer_question(timeout=60)

            if not meets_min_word_count(candidate_answer):
                print("Answer does not meet the minimum word count requirement. Score: 0")
                y_true.append(1 if meets_min_word_count(expected_answer) else 0)
                y_pred.append(0)
                continue

            score = cross_encoder_similarity_sbert(question_text, candidate_answer, sbert_model)
            scaled_score = score * (5 if difficulty == 'easy' else 5 if difficulty == 'medium' else 5)  # Each question has a maximum of 5 marks
            total_score += scaled_score

            feedback = provide_feedback(score)
            print(f"Feedback: {feedback}")

            y_true.append(1 if meets_min_word_count(expected_answer) else 0)
            y_pred.append(1 if meets_min_word_count(candidate_answer) else 0)

    f1, precision, recall = multi_metric_evaluation(y_true, y_pred)
    print(f"F1 Score: {f1:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

    return total_score

# Function to start the interview with immediate feedback
def start_interview_immediate_feedback(model, tokenizer, sbert_model, gpt2_model, gpt2_tokenizer, random_state=None):
    print("Please enter the job description:")
    job_description = input("Job Description: ")

    # Read the job description aloud
    speak_text("You entered the following job description.")
    speak_text(job_description)

    print("Please enter your resume skills:")
    resume_skills = input("Resume Skills: ")

    # Read the resume skills aloud
    speak_text("You entered the following resume skills.")
    speak_text(resume_skills)

    if "frontend" in job_description.lower() or "frontend" in resume_skills.lower():
        category = "frontend"
    elif "backend" in job_description.lower() or "backend" in resume_skills.lower():
        category = "backend"
    else:
        category = None

    if category:
        try:
            total_score = conduct_interview_immediate_feedback(category, model, tokenizer, sbert_model, num_questions=3, random_state=random_state)
            print(f"Total Score: {total_score:.2f}")
        except Exception as e:
            print(f"An error occurred during the interview: {e}")
    else:
        print("Category not identified from the job description or resume skills.")

# Start the interview
start_interview_immediate_feedback(classification_model, classification_tokenizer, sbert_model, gpt2_model, gpt2_tokenizer)
