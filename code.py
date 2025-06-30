import csv
import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import io
import string
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to preprocess answers
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]  # Lemmatization
    return ' '.join(words)

# Function to compute Jaccard similarity between two sets of words
def jaccard_similarity(answer1, answer2):
    set1 = set(word_tokenize(answer1.lower()))
    set2 = set(word_tokenize(answer2.lower()))
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0

# Function to compute Cosine similarity
def cosine_similarity_score(answer1, answer2):
    vectorizer = CountVectorizer(stop_words='english').fit_transform([answer1, answer2])
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors)[0, 1]

# Function to load questions and answers from a CSV file
def load_questions_answers(file):
    """Load questions and answers from a CSV file."""
    try:
        file_content = io.StringIO(file.getvalue().decode("utf-8"))
        reader = csv.reader(file_content)
        questions_answers = [{'question': row[0], 'answer': row[1]} for row in reader if len(row) >= 2]
        
        if not questions_answers:
            raise ValueError("CSV file is empty or malformed.")
        
        return questions_answers
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return []

# Function to evaluate student answers
def evaluate_answers(questions_answers, student_answers):
    results = []
    total_score = 0
    # Set standard thresholds for similarity
    exact_threshold = 0.9  # Standard threshold for exact match
    partial_threshold = 0.5  # Standard threshold for partial correctness
    jaccard_scores = []
    cosine_scores = []
    scores = []  # To store scores for bar chart coloring
    
    for question, student_answer in zip(questions_answers, student_answers):
        correct_answer = preprocess_text(question['answer'])
        student_answer = preprocess_text(student_answer)

        # Handle empty or malformed answers
        if not student_answer:
            feedback = f"Q: {question['question']}\nYour answer: [No answer]\n❌ Incorrect.\nCorrect answer: {question['answer']}"
            scores.append(0)  # Incorrect answer
        else:
            jaccard_sim = jaccard_similarity(correct_answer, student_answer)
            cosine_sim = cosine_similarity_score(correct_answer, student_answer)
            combined_similarity = (jaccard_sim + cosine_sim) / 2
            jaccard_scores.append(jaccard_sim)
            cosine_scores.append(cosine_sim)

            # Provide feedback based on similarity scores
            if combined_similarity >= exact_threshold:
                feedback = f"Q: {question['question']}\nYour answer: {student_answer}\n✅ Correct!"
                total_score += 1
                scores.append(1)  # Correct answer
            elif combined_similarity >= partial_threshold:
                feedback = f"Q: {question['question']}\nYour answer: {student_answer}\n⚠️ Partially correct.\nCorrect answer: {question['answer']}"
                total_score += 0.5
                scores.append(0.5)  # Partially correct answer
            else:
                feedback = f"Q: {question['question']}\nYour answer: {student_answer}\n❌ Incorrect.\nCorrect answer: {question['answer']}"
                scores.append(0)  # Incorrect answer

        results.append(feedback)

    return total_score, results, scores

# Function to save results to a CSV file
def save_results_to_csv(results, score, total_questions, output_file="results.csv"):
    data = [{"Feedback": result} for result in results]
    df = pd.DataFrame(data)
    df['Score'] = score
    df['Percentage'] = (score / total_questions) * 100
    df.to_csv(output_file, index=False)
    return output_file

# Main Streamlit app
def main():
    st.title("📝 AI Text Assessment Assistant")

    # File upload section
    st.subheader("Upload Questions and Answers CSV")
    questions_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if questions_file:
        questions_answers = load_questions_answers(questions_file)

        if questions_answers:
            student_answers = []
            for i, question in enumerate(questions_answers, 1):
                st.write(f"Q{i}: {question['question']}")
                student_answer = st.text_area(f"Your answer to Q{i}:", height=100)
                student_answers.append(student_answer)

            if st.button("Submit"):
                total_score, results, scores = evaluate_answers(questions_answers, student_answers)
                total_questions = len(questions_answers)
                st.write(f"Your total score: {total_score}/{total_questions} ({(total_score / total_questions) * 100:.2f}%)")

                st.subheader("Feedback")
                for feedback in results:
                    st.write(feedback)

                # Show a bar chart comparing the scores for each question
                st.subheader("Scores Comparison")
                st.write("Bar chart for student answers and scores.")

                fig, ax = plt.subplots()
                # Color logic: Green for correct, Blue for partial, Red for incorrect
                bar_colors = ['green' if score == 1 else ('blue' if score == 0.5 else 'red') for score in scores]
                # Bar height for partial scores is halved
                bar_heights = [score if score != 0.5 else score / 2 for score in scores]

                ax.bar(range(1, len(scores) + 1), bar_heights, color=bar_colors, alpha=0.6)
                ax.set_xlabel("Questions")
                ax.set_ylabel("Score")
                ax.set_title("Score per Question ")
                st.pyplot(fig)

                # Save results to a CSV file
                output_file = save_results_to_csv(results, total_score, total_questions)
                with open(output_file, "rb") as file:
                    st.download_button("Download Feedback", file, file_name="results.csv")

if __name__ == "__main__":
    main()
