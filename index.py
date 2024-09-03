import os
import pandas as pd
import random
import speech_recognition as sr
import pyttsx3
from transformers import RobertaTokenizer, RobertaForSequenceClassification, GPT2Tokenizer, GPTNeoForCausalLM
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import f1_score, precision_score, recall_score
from textblob import TextBlob
import keyboard  # For grammar and spelling checks

# Disable oneDNN optimizations in TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize models and tokenizers
classification_model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=1)
classification_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
gptneo_model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-2.7B')
gptneo_tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-2.7B')
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()

# Function to read the question aloud
def speak_text(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

# Voice recognition setup
def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please answer the question after the beep...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)

    try:
        answer = recognizer.recognize_google(audio)
        print(f"You said: {answer}")
        return answer
    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
        return ""
    except sr.RequestError:
        print("Could not request results; check your network connection.")
        return ""

# Cross-encoder similarity function using Sentence-BERT
def cross_encoder_similarity_sbert(question, candidate_answer, model):
    embeddings = model.encode([question, candidate_answer])
    score = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return score.item()

# Detect if the answer is a rephrased version of the question
def is_paraphrase(question, candidate_answer, model):
    similarity_score = cross_encoder_similarity_sbert(question, candidate_answer, model)
    # If the similarity score is very high (e.g., >= 0.9), treat it as a paraphrase
    return similarity_score >= 0.9

# Provide feedback for each question and assign a score based on feedback
def provide_question_feedback(score, difficulty, is_exact_match, is_paraphrase, is_content_correct):
    feedback_messages = {
        'excellent': "Excellent answer!",
        'good': "Good answer, but there's room for improvement.",
        'average': "Your answer was average. Consider revising the key concepts.",
        'poor': "Your answer was  poor. You may need to review the material.",
        'exact_match': "Your answer was identical to the question, which is not acceptable. Please try to provide a more precise and accurate answer.",
        'paraphrase': "Your answer is too similar to the question. so do not give question as answer to me. Please provide a more detailed and informative answer.",
    }

    if is_exact_match:
        feedback = feedback_messages['exact_match']
        assigned_score = 0
    elif is_paraphrase:
        feedback = feedback_messages['paraphrase']
        assigned_score = 0
    elif score >= 0.8 and is_content_correct:
        feedback = feedback_messages['excellent']
        assigned_score = random.uniform(8, 10)
    elif score >= 0.6:
        feedback = feedback_messages['good']
        assigned_score = random.uniform(3, 8)
    elif score >= 0.4:
        feedback = feedback_messages['average']
        assigned_score = random.uniform(1, 3)
    else:
        feedback = feedback_messages['poor']
        assigned_score = random.uniform(0, 1)

    speak_text(feedback)
    print(f"Feedback for {difficulty} question: {feedback}")

    return assigned_score

# Provide feedback based on the overall score at the end
def provide_final_feedback(total_score, max_score):
    feedback_messages = {
        'excellent': "Excellent performance! You've demonstrated a strong understanding.",
        'good': "Good performance! There are some areas for improvement, but overall well done.",
        'average': "Average performance. Consider revising key concepts.",
        'poor': "Below average performance. You may need to review the material thoroughly."
    }

    percentage_score = (total_score / max_score) * 100

    if percentage_score >= 80:
        return feedback_messages['excellent']
    elif percentage_score >= 60:
        return feedback_messages['good']
    elif percentage_score >= 40:
        return feedback_messages['average']
    else:
        return feedback_messages['poor']

# Ensure the answer meets minimum word count
def meets_min_word_count(answer, min_word_count=10):
    return len(answer.split()) >= min_word_count

# Check content correctness
def validate_content(expected_answer, candidate_answer):
    return expected_answer.strip().lower() in candidate_answer.strip().lower()

# Load data based on the category
def load_data(category):
    file_paths = {
        "frontend": r'D:\Aditya msr\.project\Interview\frontend_difficulty.csv',
        "backend": r'D:\Aditya msr\.project\Interview\java_difficulty.csv'
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

# Conduct an interview with feedback for each question and final feedback
def conduct_interview_with_feedback(category, model, tokenizer, sbert_model, num_questions=15, random_state=None):
    try:
        data = load_data(category)
    except Exception as e:
        print(f"Failed to load data: {e}")
        return "Error loading data."

    # Prepare question pools
    try:
        easy_questions = get_questions_by_difficulty(data, 'easy')
        medium_questions = get_questions_by_difficulty(data, 'medium')
        hard_questions = get_questions_by_difficulty(data, 'hard')
    except Exception as e:
        print(f"Error in filtering questions by difficulty: {e}")
        return "Error in filtering questions."

    if easy_questions.empty and medium_questions.empty and hard_questions.empty:
        print("No questions available for the selected category and difficulty levels.")
        return "No questions available."

    total_score = 0

    max_score = 0
    y_true = []
    y_pred = []

    # Performance tracking
    difficulty_rounds = {'easy': 2, 'medium': 2, 'hard': 2}  # Total 15 questions

    for difficulty in ['easy', 'medium', 'hard']:
        questions = easy_questions if difficulty == 'easy' else medium_questions if difficulty == 'medium' else hard_questions
        for _ in range(difficulty_rounds[difficulty]):
            if questions.empty:
                break
            question = questions.sample(n=1).iloc[0]
            question_text = question['Question']
            expected_answer = question['Answer']

            # Read the question aloud
            speak_text(f"This is a {difficulty} level question: {question_text}")

            print(f"Question ({difficulty}): {question_text}")
            print("Press 'S' to submit your response.")
            while True:
                if keyboard.is_pressed('s'):
                    candidate_answer = get_voice_input()
                    if candidate_answer:
                        break

            # Check for exact match with the question text
            is_exact_match = candidate_answer.strip().lower() == question_text.strip().lower()

            # Check if the answer is a paraphrase of the question
            is_paraphrase_result = is_paraphrase(question_text, candidate_answer, sbert_model)

            if is_exact_match:
                print("Answer is identical to the question, which is not considered a valid response. Score: 0")
                y_true.append(1 if meets_min_word_count(expected_answer) else 0)
                y_pred.append(0)
                continue

            if not meets_min_word_count(candidate_answer):
                print("Answer does not meet the minimum word count requirement. Score: 0")
                y_true.append(1 if meets_min_word_count(expected_answer) else 0)
                y_pred.append(0)
                continue

            # Check content correctness
            content_correct = validate_content(expected_answer, candidate_answer)

            # Calculate initial similarity score
            score = cross_encoder_similarity_sbert(question_text, candidate_answer, sbert_model)

            # Get the assigned score based on feedback
            assigned_score = provide_question_feedback(score, difficulty, is_exact_match, is_paraphrase_result, content_correct)

            # Adjust total and maximum scores
            total_score += assigned_score
            max_score += 10  # Maximum possible score per question

            # Track true and predicted values for later analysis
            y_true.append(1 if content_correct else 0)
            y_pred.append(1 if score >= 0.5 else 0)

            print(f"Candidate Answer: {candidate_answer}")
            print(f"Expected Answer: {expected_answer}")
            print(f"Assigned Score: {assigned_score:.2f}")

    final_feedback = provide_final_feedback(total_score, max_score)
    print(f"Final Feedback: {final_feedback}")
    return final_feedback

# Example execution
category = "frontend"  # Could be 'frontend' or 'backend'
final_feedback = conduct_interview_with_feedback(category, classification_model, classification_tokenizer, sbert_model)
print("Interview completed. Final Feedback:", final_feedback)
