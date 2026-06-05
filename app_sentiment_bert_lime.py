from flask import Flask, render_template, request, jsonify
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from textblob import TextBlob
import re
import string
import traceback

app = Flask(__name__)

MODEL_NAME = "unitary/toxic-bert"
CLASS_NAMES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
SPECIAL_TOKENS = ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]']

# Force CPU usage
torch.set_num_threads(1)

# Punctuation filter
PUNCTUATION = set(string.punctuation)
SYMBOLS = set(['...', '``', "''", '--', '…', '’', '‘', '”', '“'])

# Category Definitions
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

def remove_punctuation(text):
    """Remove all punctuation from text for sentiment analysis"""
    text_without_punct = re.sub(r'[^\w\s]', '', text)
    text_without_punct = re.sub(r'\s+', ' ', text_without_punct).strip()
    return text_without_punct

def check_sentiment(text):
    """Sentiment analysis using TextBlob"""
    clean_text = remove_punctuation(text)
    print(f"Clean text for sentiment: {clean_text[:50]}...")
    
    blob = TextBlob(clean_text)
    polarity = blob.sentiment.polarity
    
    if polarity > 0.1:
        return 'POSITIVE', abs(polarity)
    elif polarity < -0.1:
        return 'NEGATIVE', abs(polarity)
    else:
        return 'NEUTRAL', abs(polarity)

def is_punctuation_or_symbol(token):
    if len(token) == 1 and token in PUNCTUATION:
        return True
    if token in SYMBOLS:
        return True
    if all(c in PUNCTUATION for c in token):
        return True
    return False

# Load toxicity model
print("Loading toxicity detection model...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    print("Toxicity model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    traceback.print_exc()
    raise

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
    """Map token-level attention back to words"""
    probs, token_importance, input_ids = predict_with_attention(text)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    word_importance = {}
    word_details = {}
    current_word = ""
    current_importance = 0.0
    
    for token, importance in zip(tokens, token_importance):
        # Skip special tokens
        if token in SPECIAL_TOKENS:
            if current_word:
                word_importance[current_word] = float(current_importance)
                word_lower = current_word.lower()
                current_word = ""
                current_importance = 0.0
            continue
        
        # Skip punctuation
        if is_punctuation_or_symbol(token):
            if current_word:
                word_importance[current_word] = float(current_importance)
                word_lower = current_word.lower()
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
                word_lower = current_word.lower()
            current_word = token
            current_importance = float(importance)
    
    # Add last word
    if current_word:
        word_importance[current_word] = float(current_importance)
        word_lower = current_word.lower()
    
    # Normalize importance scores
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
        
        # Get sentiment
        print("\n--- Running sentiment analysis ---")
        sentiment_label, sentiment_confidence = check_sentiment(text)
        print(f"Sentiment: {sentiment_label} (confidence: {sentiment_confidence:.3f})")
        
        # CASE 1: Positive sentiment -> Directly Non-Toxic (skip BERT)
        if sentiment_label == 'POSITIVE':
            print("Case 1: Positive sentiment detected -> Non-Toxic (skipping BERT)")
            
            predictions = []
            for name in CLASS_NAMES:
                predictions.append({
                    'name': name,
                    'probability': 0.0,
                    'percentage': 0.0,
                    'is_predicted': False
                })
            
            return jsonify({
                'success': True,
                'text': text,
                'sentiment': {
                    'label': 'POSITIVE',
                    'confidence': round(sentiment_confidence, 3)
                },
                'case': 1,
                'decision': 'NON_TOXIC',
                'decision_message': '✅ NOT TOXIC - Positive sentiment detected',
                'toxic_probability': 0.0,
                'primary_prediction': {
                    'name': 'non-toxic',
                    'probability': 0.0,
                    'percentage': 0,
                    'is_predicted': False
                },
                'all_predictions': predictions,
                'detected_categories': [],
                'important_words': []
            })
        
        # CASE 2 & 3: Neutral OR Negative sentiment -> Run BERT
        print(f"Case {2 if sentiment_label == 'NEUTRAL' else 3}: {sentiment_label} sentiment -> Running BERT analysis")
        print("\n--- Running BERT toxicity analysis ---")
        probs, word_importance, word_details = get_word_importance(text)
        probs_list = [float(p) for p in probs]
        toxic_prob = probs_list[0]  # 'toxic' is first in CLASS_NAMES
        
        print(f"Toxic probability: {toxic_prob:.3f}")
        
        # Apply rules based on sentiment and toxic probability
        if sentiment_label == 'NEUTRAL':
            # Case 2: Neutral sentiment
            if toxic_prob > 0.6:
                decision = 'TOXIC'
                is_toxic = True
                primary_name = 'toxic'
                primary_prob = toxic_prob
                message = f'⚠️ TOXIC - Neutral sentiment but toxicity probability {toxic_prob:.2%} (> 60%)'
                print(f"Result: Toxic (neutral sentiment, prob={toxic_prob:.3f} > 0.6)")
            else:
                decision = 'SARCASTIC'
                is_toxic = False
                primary_name = 'non-toxic'
                primary_prob = 0.0
                message = f'🤔 SARCASTIC - Neutral sentiment with toxicity probability {toxic_prob:.2%} (< 60%)'
                print(f"Result: Sarcastic (neutral sentiment, prob={toxic_prob:.3f} <= 0.6)")
        else:  # sentiment_label == 'NEGATIVE'
            # Case 3: Negative sentiment
            if toxic_prob > 0.6:
                decision = 'TOXIC'
                is_toxic = True
                primary_name = 'toxic'
                primary_prob = toxic_prob
                message = f'⚠️ TOXIC - Negative sentiment and toxicity probability {toxic_prob:.2%} (> 60%)'
                print(f"Result: Toxic (negative sentiment, prob={toxic_prob:.3f} > 0.6)")
            else:
                decision = 'SARCASTIC'
                is_toxic = False
                primary_name = 'non-toxic'
                primary_prob = 0.0
                message = f'🤔 SARCASTIC - Negative sentiment but low toxicity probability {toxic_prob:.2%} (< 60%)'
                print(f"Result: Sarcastic (negative sentiment, prob={toxic_prob:.3f} <= 0.6)")
        
        # Build predictions for all categories
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
        
        predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        # Get important words only if toxic
        important_words = []
        if decision == 'TOXIC':
            print("Extracting important words for toxic content")
            sorted_words = sorted(word_importance.items(), key=lambda x: x[1], reverse=True)
            
            for word, importance in sorted_words[:10]:
                if len(word) < 2 and word.lower() not in ['i', 'a', 'we', 'he', 'she', 'it', 'be', 'to', 'do', 'so', 'go']:
                    continue
                
                word_info = word_details.get(word, {})
                label = word_info.get('label', 'unknown')
                specific_def = word_info.get('specific', '')
                category_def = CATEGORY_DEFINITIONS.get(label, {}).get('full', '')
                
                important_words.append({
                    'word': word,
                    'importance': round(float(importance), 3),
                    'label': label,
                    'specific_definition': specific_def,
                    'category_definition': category_def
                })
        
        response_data = {
            'success': True,
            'text': text,
            'sentiment': {
                'label': sentiment_label,
                'confidence': round(sentiment_confidence, 3)
            },
            'case': 2 if sentiment_label == 'NEUTRAL' else 3,
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
        traceback.print_exc()
        print("="*60 + "\n")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("STARTING FLASK SERVER")
    print("="*60)
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=False)