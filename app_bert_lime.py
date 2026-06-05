from flask import Flask, render_template, request, jsonify
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import string

app = Flask(__name__)

MODEL_NAME = "unitary/toxic-bert"
CLASS_NAMES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
SPECIAL_TOKENS = ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]']

# Force CPU usage
torch.set_num_threads(1)

# Define punctuation to filter out
PUNCTUATION = set(string.punctuation)
SYMBOLS = set(['...', '``', "''", '--', '…', '’', '‘', '”', '“'])

# Category Definitions for display under words
CATEGORY_DEFINITIONS = {
    'toxic': {
        'short': 'Rude/disrespectful content',
        'full': 'Content that is rude, disrespectful, or likely to make someone leave a discussion.'
    },
    'severe_toxic': {
        'short': 'Extremely hateful/violent content',
        'full': 'Extremely hateful, aggressive, or violent content that goes beyond basic toxicity.'
    },
    'obscene': {
        'short': 'Vulgar/sexually explicit language',
        'full': 'Content containing vulgar, lewd, or sexually explicit language.'
    },
    'threat': {
        'short': 'Violent intent or harm',
        'full': 'Statements that suggest intent to inflict harm, violence, or negative consequences.'
    },
    'insult': {
        'short': 'Personal attack/name-calling',
        'full': 'Personal attacks, name-calling, or demeaning statements targeting an individual or group.'
    },
    'identity_hate': {
        'short': 'Discrimination based on identity',
        'full': 'Hateful content targeting people based on race, ethnicity, gender, sexual orientation, religion, or other identity factors.'
    }
}

def is_punctuation_or_symbol(token):
    """Check if a token is punctuation or a symbol"""
    if len(token) == 1 and token in PUNCTUATION:
        return True
    if token in SYMBOLS:
        return True
    if all(c in PUNCTUATION for c in token):
        return True
    return False

# Load model
print("Loading toxicity detection model...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Trying with local cache...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=False)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, local_files_only=False)
    model.eval()
    print("Model loaded successfully!")

def predict_with_attention(text):
    """Get prediction AND attention weights from the model"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        logits = outputs.logits
        probs = torch.sigmoid(logits).numpy()[0]
        
        # Get attention from the last layer
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            last_layer_attention = outputs.attentions[-1]
            avg_attention = last_layer_attention.mean(dim=1).squeeze()
            token_importance = avg_attention.mean(dim=0).numpy()
        else:
            token_importance = np.ones(len(inputs['input_ids'][0]))
        
    return probs, token_importance, inputs['input_ids'][0]

def get_word_importance(text):
    """Map token-level attention back to words, filtering out special tokens and punctuation"""
    probs, token_importance, input_ids = predict_with_attention(text)
    
    # Decode tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    # Group subwords into words, filtering out special tokens and punctuation
    word_importance = {}
    word_details = {}  # Store label and specific definition for each word
    current_word = ""
    current_importance = 0.0
    
    for token, importance in zip(tokens, token_importance):
        # Skip special tokens
        if token in SPECIAL_TOKENS:
            if current_word:
                word_importance[current_word] = float(current_importance)
                current_word = ""
                current_importance = 0.0
            continue
        
        # Skip punctuation and symbols
        if is_punctuation_or_symbol(token):
            if current_word:
                word_importance[current_word] = float(current_importance)
                current_word = ""
                current_importance = 0.0
            continue
        
        if token.startswith("##"):
            # Continuation of previous word
            current_word += token[2:]
            current_importance += float(importance)
        else:
            # New word starts
            if current_word:
                word_importance[current_word] = float(current_importance)
            current_word = token
            current_importance = float(importance)
    
    # Add last word
    if current_word:
        word_importance[current_word] = float(current_importance)
    
    # Fallback if no words found
    if not word_importance:
        for token, importance in zip(tokens, token_importance):
            if token not in SPECIAL_TOKENS and not token.startswith("##") and not is_punctuation_or_symbol(token):
                word_importance[token] = float(importance)
    
    # Normalize importance scores to 0-1 range
    if word_importance:
        max_imp = max(word_importance.values())
        if max_imp > 0:
            word_importance = {w: i/max_imp for w, i in word_importance.items()}
    
    return probs, word_importance, word_details

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    print("\n" + "="*60)
    print("ANALYZE REQUEST RECEIVED")
    print("="*60)
    
    try:
        text = request.form.get('text', '').strip()
        print(f"Original text: '{text}'")
        
        if len(text) < 3:
            return jsonify({'error': 'Please enter at least 3 characters'}), 400
        
        # Run BERT toxicity analysis directly (no sentiment check)
        print("\n--- Running BERT toxicity analysis ---")
        probs, word_importance, word_details = get_word_importance(text)
        
        # Convert probs to list of floats
        probs_list = [float(p) for p in probs]
        toxic_prob = probs_list[0]  # 'toxic' is first in CLASS_NAMES
        
        print(f"Toxic probability: {toxic_prob:.3f}")
        
        # Determine toxicity based on standard threshold
        if toxic_prob > 0.5:
            decision = 'TOXIC'
            is_toxic = True
            primary_name = 'toxic'
            primary_prob = toxic_prob
            message = f'⚠️ TOXIC - Toxicity probability {toxic_prob:.2%} (> 50%)'
            print(f"Result: Toxic (prob={toxic_prob:.3f} > 0.5)")
        else:
            decision = 'NON_TOXIC'
            is_toxic = False
            primary_name = 'non-toxic'
            primary_prob = 0.0
            message = f'✅ NON_TOXIC - Toxicity probability {toxic_prob:.2%} (< 50%)'
            print(f"Result: Non-toxic (prob={toxic_prob:.3f} <= 0.5)")
        
        # Get top predictions
        predictions = []
        detected_categories = set()
        
        for i, name in enumerate(CLASS_NAMES):
            prob = probs_list[i]
            
            # Only mark as detected if decision is TOXIC and probability > 0.5
            if decision == 'TOXIC' and prob > 0.5:
                is_detected = True
                detected_categories.add(name)
            else:
                is_detected = False
            
            predictions.append({
                'name': name,
                'probability': round(prob, 3),
                'percentage': round(prob * 100, 2),
                'is_predicted': is_detected
            })
        
        # Sort predictions by probability
        predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        # Prepare word importance for display with definitions (only if toxic)
        important_words = []
        if decision == 'TOXIC':
            print("Extracting important words for toxic content")
            sorted_words = sorted(word_importance.items(), key=lambda x: x[1], reverse=True)
            
            for word, importance in sorted_words[:10]:
                # Filter out very short words that aren't meaningful
                if len(word) < 2 and word.lower() not in ['i', 'a', 'we', 'he', 'she', 'it', 'be', 'to', 'do', 'so', 'go']:
                    continue
                
                # Get word details if available
                word_info = word_details.get(word, {})
                label = word_info.get('label', 'unknown')
                specific_def = word_info.get('specific', '')
                
                # Get the full category definition if label exists
                category_def = CATEGORY_DEFINITIONS.get(label, {}).get('full', '')
                
                important_words.append({
                    'word': word,
                    'importance': round(float(importance), 3),
                    'label': label,
                    'specific_definition': specific_def,
                    'category_definition': category_def
                })
        
        # Prepare response
        response_data = {
            'success': True,
            'text': text,
            'decision': decision,
            'decision_message': message,
            'toxic_probability': round(toxic_prob, 3),
            'primary_prediction': {
                'name': primary_name,
                'probability': primary_prob,
                'percentage': round(primary_prob * 100, 2),
                'is_predicted': is_toxic
            },
            'all_predictions': predictions,
            'detected_categories': list(detected_categories),
            'important_words': important_words
        }
        
        print("Sending successful response")
        print("="*60 + "\n")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"\n!!! ERROR in analyze: {e}")
        import traceback
        traceback.print_exc()
        print("="*60 + "\n")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("STARTING FLASK SERVER")
    print("="*60)
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=False)