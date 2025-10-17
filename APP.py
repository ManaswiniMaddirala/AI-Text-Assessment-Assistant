import csv
import streamlit as st
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import io
import string
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from heapq import nlargest
import matplotlib.pyplot as plt
import time

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    """Cleans, tokenizes, removes stop words, and lemmatizes the text for scoring."""
    text = text.lower() 
    
    text = ''.join([char for char in text if char not in string.punctuation])
    words = word_tokenize(text)

    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english') and word.isalnum()]
    return ' '.join(words)

def jaccard_similarity(answer1, answer2):
    """Computes Jaccard similarity between two sets of words."""
    set1 = set(word_tokenize(answer1.lower()))
    set2 = set(word_tokenize(answer2.lower()))
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0

def cosine_similarity_score(answer1, answer2):
    """Computes Cosine similarity using Count Vectorization."""
    vectorizer = CountVectorizer(stop_words='english').fit_transform([answer1, answer2])
    vectors = vectorizer.toarray()
    # Check for empty vectors (can happen if both answers are very short/stopwords only)
    if vectors.shape[0] < 2 or vectors.sum() == 0:
        return 0.0
    return cosine_similarity(vectors)[0, 1]

def summarize_document(document, num_sentences=3):
    """
    Generates an extractive summary of a document using sentence scoring based on word frequency.
    This fulfills the 'summarization module' requirement.
    """
    if not document:
        return ""
    
    sentences = sent_tokenize(document)
    if len(sentences) <= num_sentences:
        return document # Return original text if too short to summarize

    # Calculate word frequency
    words = word_tokenize(document.lower())
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.isalnum() and word not in stop_words]
    word_freq = FreqDist(words)
    
    # Assign scores to sentences
    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_freq:
                if len(sentence.split(' ')) < 40: 
                    sentence_scores[sentence] = sentence_scores.get(sentence, 0) + word_freq[word]
    
    try:
        summary_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
        summary = ' '.join(summary_sentences)
    except ValueError:
        summary = "Could not generate summary."

    return summary


def load_questions_answers(file):
    """Load questions and answers (Q/A pairs) from a CSV file."""
    try:

        file_content = io.StringIO(file.getvalue().decode("utf-8"))

        reader = csv.reader(file_content)
        
        try:
            next(reader) 
        except StopIteration:
            raise ValueError("CSV file is empty or malformed.")
            
        questions_answers = [{'question': row[0], 'answer': row[1]} for row in reader if len(row) >= 2]
        
        if not questions_answers:
            raise ValueError("CSV file contains no valid Q/A pairs.")
            
        return questions_answers
    except Exception as e:
        st.error(f"Error reading CSV file. Ensure it has at least two columns (Question, Answer): {e}")
        return []

def evaluate_answers(questions_answers, student_raw_answers):
    """Evaluates student answers against correct answers using similarity metrics."""
    results_data = []
    total_score = 0
    scores = []
    
    exact_threshold = 0.9  
    partial_threshold = 0.5  
    
    SUMMARY_WORD_COUNT_THRESHOLD = 30
    
    for i, question in enumerate(questions_answers):
        student_raw_answer = student_raw_answers[i]
        
        correct_answer_processed = preprocess_text(question['answer'])
        student_answer_processed = preprocess_text(student_raw_answer)

        score_value = 0
        feedback = ""
        summary = ""
        
        if not student_answer_processed:
            feedback = f"❌ Incorrect (No answer submitted or answer contained only filler words)."
        else:
            jaccard_sim = jaccard_similarity(correct_answer_processed, student_answer_processed)
            cosine_sim = cosine_similarity_score(correct_answer_processed, student_answer_processed)
            combined_similarity = (jaccard_sim + cosine_sim) / 2
            
            if combined_similarity >= exact_threshold:
                feedback = f"✅ Correct! (Similarity: {combined_similarity:.2f})"
                score_value = 1
            elif combined_similarity >= partial_threshold:
                feedback = f"⚠️ Partially correct. (Similarity: {combined_similarity:.2f})"
                score_value = 0.5
            else:
                feedback = f"❌ Incorrect. (Similarity: {combined_similarity:.2f})"
                score_value = 0
        
        total_score += score_value
        scores.append(score_value)
        
        if len(student_raw_answer.split()) > SUMMARY_WORD_COUNT_THRESHOLD:
            summary = summarize_document(student_raw_answer, num_sentences=3)
      
        results_data.append({
            'Question': question['question'],
            'Correct Answer (Expected)': question['answer'],
            'Student Answer (Raw)': student_raw_answer,
            'Score': score_value,
            'Feedback': feedback,
            'Summary (if > 30 words)': summary if summary else "N/A"
        })

    return total_score, results_data, scores

def save_results_to_csv(results_data):
    """Saves the results DataFrame to a CSV string in memory for download."""
    df = pd.DataFrame(results_data)
    
    total_questions = len(df)
    total_score = df['Score'].sum()
    percentage = (total_score / total_questions) * 100
    
    summary_rows = pd.DataFrame([
        {'Question': '--- Summary ---', 'Correct Answer (Expected)': '', 'Student Answer (Raw)': '', 'Score': '', 'Feedback': ''},
        {'Question': 'Total Score', 'Correct Answer (Expected)': '', 'Student Answer (Raw)': '', 'Score': total_score, 'Feedback': ''},
        {'Question': 'Total Questions', 'Correct Answer (Expected)': '', 'Student Answer (Raw)': '', 'Score': total_questions, 'Feedback': ''},
        {'Question': 'Percentage Score', 'Correct Answer (Expected)': '', 'Student Answer (Raw)': '', 'Score': f"{percentage:.2f}%", 'Feedback': ''}
    ])
    
    df = pd.concat([df, summary_rows], ignore_index=True)
    
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    return csv_buffer.getvalue().encode('utf-8')


def main():
    st.set_page_config(
        page_title="AI Assessment Assistant (NLP & Summarization)", 
        page_icon="📝", 
        layout="wide"
    )
    st.title("📝 AI Assessment & Summarization App")
    st.markdown("Use NLP to automatically grade free-text answers and summarize long responses.")

    st.subheader("1. Upload Questions and Answers CSV")
    st.info("Your CSV should have at least two columns: `Question` (Column 1) and `Answer` (Column 2).")
    questions_file = st.file_uploader("Upload your Q/A CSV file", type=["csv"])

    if questions_file:
        questions_answers = load_questions_answers(questions_file)

        if questions_answers:
            st.subheader(f"2. Input Student Answers ({len(questions_answers)} Questions Found)")
            student_raw_answers = []
            
            cols = st.columns(2)
            
            for i, question in enumerate(questions_answers):
                with cols[i % 2]:
        
                    st.markdown(f"**Q{i+1}:** *{question['question']}*")
                    
                    student_answer = st.text_area(
                        f"Answer to Q{i+1} (Correct Answer Preview: {question['answer'][:40]}...)", 
                        key=f"answer_{i}", 
                        height=120,
                        placeholder="Enter student response here..."
                    )
                    student_raw_answers.append(student_answer)
                    st.markdown("---")


            if st.button("3. 🚀 Grade and Generate Feedback", use_container_width=True, type="primary"):
                with st.spinner('Grading answers and summarizing long responses...'):
                    start_time = time.time()
                    total_score, results_data, scores = evaluate_answers(questions_answers, student_raw_answers)
                    total_questions = len(questions_answers)
                    end_time = time.time()

                st.subheader("4. Evaluation Results")

                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Total Score", f"{total_score} / {total_questions}")
                col_m2.metric("Percentage Score", f"{(total_score / total_questions) * 100:.2f}%")
                col_m3.metric("Time Taken", f"{(end_time - start_time):.2f} seconds")

                st.markdown("### Detailed Feedback and Summaries")
                df_results = pd.DataFrame(results_data)
                
                st.dataframe(
                    df_results[['Question', 'Student Answer (Raw)', 'Summary (if > 30 words)', 'Score', 'Feedback']],
                    use_container_width=True,
                    hide_index=True
                )
                
                st.markdown("### Score Distribution per Question")
                
                df_plot = pd.DataFrame({
                    'Question': [f"Q{i+1}" for i in range(total_questions)],
                    'Score': scores,
                    'Color': ['green' if s == 1 else ('orange' if s == 0.5 else 'red') for s in scores]
                })

                fig, ax = plt.subplots(figsize=(10, 4))
                
                ax.bar(df_plot['Question'], df_plot['Score'], color=df_plot['Color'], alpha=0.7)
                
                ax.set_ylim(0, 1.1)
                ax.set_ylabel("Score (0.0 to 1.0)")
                ax.set_title("Score per Question")
                
                legend_elements = [
                    plt.Rectangle((0, 0), 1, 1, fc='green', alpha=0.7, label='1.0 (Correct)'),
                    plt.Rectangle((0, 0), 1, 1, fc='orange', alpha=0.7, label='0.5 (Partial)'),
                    plt.Rectangle((0, 0), 1, 1, fc='red', alpha=0.7, label='0.0 (Incorrect)')
                ]
                ax.legend(handles=legend_elements, loc='upper right')
                
                st.pyplot(fig)
                
                csv_bytes = save_results_to_csv(results_data)
                st.download_button(
                    "Download Full Feedback Report (CSV)", 
                    csv_bytes, 
                    file_name="AI_Grading_Report.csv", 
                    mime="text/csv",
                    use_container_width=True
                )


if __name__ == "__main__":
    main()
