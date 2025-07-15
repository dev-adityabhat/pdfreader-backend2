import os
from flask import Flask, request, jsonify
from flask_cors import CORS
# from googletrans import Translator
import nltk
from nltk.corpus import wordnet
import fitz
import google.generativeai as genai
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Download required NLTK data
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = 'temp_uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize services
# translator = Translator()

# Configure Gemini AI
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
generation_config = {
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]
model = genai.GenerativeModel(model_name="gemini-2.0-flash",
                            generation_config=generation_config,
                            safety_settings=safety_settings)

def extract_text_from_pdf(file_path):
    """Extract text from PDF with proper paragraph separation."""
    doc = fitz.open(file_path)
    paragraphs = []
    
    for page in doc:
        text = page.get_text()
        if text.strip():
            # Split text into paragraphs and clean them
            page_paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            paragraphs.extend(page_paragraphs)
    
    doc.close()
    return paragraphs

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': True, 'message': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': True, 'message': 'No selected file'})
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            paragraphs = extract_text_from_pdf(filepath)
            # Clean up the temporary file
            os.remove(filepath)
            
            return jsonify({
                'error': False,
                'content': paragraphs
            })
        except Exception as e:
            return jsonify({'error': True, 'message': str(e)})

# @app.route('/translate', methods=['POST'])
# def translate_text():
#     try:
#         data = request.get_json()
#         text = data.get('text', '')
#         target_lang = data.get('target_lang', 'en')
        
#         if not text:
#             return jsonify({'error': True, 'message': 'No text provided'})
        
#         translation = translator.translate(text, dest=target_lang)
#         return jsonify({
#             'translation': translation.text,
#             'error': False
#         })
#     except Exception as e:
#         return jsonify({'error': True, 'message': str(e)})

@app.route('/define', methods=['POST'])
def define_word():
    try:
        data = request.get_json()
        word = data.get('word', '').strip().lower()
        
        if not word:
            return jsonify({'error': True, 'message': 'No word provided'})
        
        # Get word definition from WordNet
        synsets = wordnet.synsets(word)
        if not synsets:
            return jsonify({'definition': f'No definition found for "{word}"'})
        
        # Get the first definition and examples
        definition = synsets[0].definition()
        examples = synsets[0].examples()
        
        response = {
            'definition': definition,
            'examples': examples if examples else [],
            'error': False
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': True, 'message': str(e)})

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        selected_text = data.get('selected_text', '')
        book_context = data.get('book_context', '')
        message = data.get('message', '')
        
        # Construct prompt with context
        prompt = f"""
        Context from the book:
        {book_context}

        Selected text:
        "{selected_text}"

        User question/message:
        {message}

        Please provide a helpful response about the selected text, taking into account the surrounding context from the book.
        If the user is asking about the meaning or interpretation, provide a clear explanation.
        If they're asking about facts or claims, verify them against the context.
        If they're asking about connections or implications, analyze them based on the provided context.
        if user is asking about the author, provide a brief overview of the author's background and relevance to the text.
        if is aking for translation, provide the translation in Marathi language.
        if user is asking for a summary, provide a concise summary of the selected text.
        if user is asking for a definition, provide the definition of the word in English.
        if user is asking for examples, provide relevant examples from the text.
        if user is asking for a comparison, provide a comparison with another text or concept.
    
        """
        
        response = model.generate_content(prompt)
        
        if not response.text:
            return jsonify({
                'error': True,
                'message': 'No response generated'
            })
            
        return jsonify({
            'message': response.text,
            'error': False
        })
    except Exception as e:
        print(f"Chat error: {str(e)}")  # Debug print
        return jsonify({'error': True, 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)