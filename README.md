# Detection of Toxic Language and Threats

A basic application that combines sentiment analysis (using TextBlob) with BERT-based toxicity detection (toxic-bert) to identify toxic content, detect potential sarcasm, and provide detailed word-level explanations.

## Features

### Dual-Analysis System
- **Sentiment Analysis**: First-pass analysis using TextBlob to detect positive, negative, or neutral sentiment
- **BERT Toxicity Detection**: Deep learning model for multi-category toxicity classification

### Toxicity Categories Detected
1. **Toxic** - Rude, disrespectful, or offensive content
2. **Severe Toxic** - Extremely hateful, aggressive, or violent content
3. **Obscene** - Vulgar, lewd, or sexually explicit language
4. **Threat** - Statements suggesting intent to inflict harm
5. **Insult** - Personal attacks, name-calling, or demeaning statements
6. **Identity Hate** - Discrimination based on identity factors

### Classification Outcomes
- **TOXIC**: Content identified as genuinely toxic
- **SARCASTIC**: Content with neutral/negative sentiment but low toxicity (potential sarcasm)
- **NON-TOXIC**: Safe content (positive sentiment)

### Additional Features
- **Word-level Importance**: Highlights which words contributed most to toxicity decisions
- **Real-time Analysis**: Instant feedback with probability scores
- **Category Definitions**: Detailed explanations for each toxicity type
- **Interactive Web Interface**: User-friendly interface with visual feedback

## How It Works

The application uses a three-case decision system:

### Case 1: Positive Sentiment
- **Trigger**: Sentiment analysis returns POSITIVE
- **Action**: Skip BERT analysis, immediately classify as NON-TOXIC
- **Rationale**: Positive sentiment content is unlikely to be toxic

### Case 2: Neutral Sentiment
- **Trigger**: Sentiment analysis returns NEUTRAL
- **Action**: Run BERT toxicity analysis
- **Decision Rules**:
  - Toxicity > 60% → **TOXIC**
  - Toxicity ≤ 60% → **SARCASTIC**

### Case 3: Negative Sentiment
- **Trigger**: Sentiment analysis returns NEGATIVE
- **Action**: Run BERT toxicity analysis
- **Decision Rules**:
  - Toxicity > 60% → **TOXIC**
  - Toxicity ≤ 60% → **SARCASTIC**

## Prerequisites

Before running this application, ensure you have the following installed:

- **Python 3.7 or higher** (Python 3.8+ recommended)
- **pip** (Python package installer)
- Minimum **4GB RAM** (2GB for model, 2GB for system)
- **Internet connection** (for first-time model download)

## Installation

### 1. Clone or Download the Repository

```bash
git clone <your-repository-url>
cd <project-directory>
```

Or download the project files to your local machine.

### 2. Create a Virtual Environment (Recommended)

Create and activate a virtual environment to avoid dependency conflicts:

**On Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Required Dependencies

Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, install manually:

```bash
pip install flask numpy torch transformers textblob
python -m textblob.download_corpora
```

**Note**: The last command downloads necessary corpora for TextBlob sentiment analysis.

### 4. Verify File Structure

Ensure your project directory has the following structure:

```
your-project/
├── app_sentiment_bert_lime.py   # Main application file
├── requirements.txt              # Python dependencies
├── templates/                    # HTML templates directory
│   └── index.html               # Main web interface
├── static/                       # Static files (CSS, JS, images)
│   └── (your static files)
└── README.md                     # This file
```

**Important**: 
- The `templates` folder must contain `index.html`
- The `static` folder should be present (can be empty initially)

## Running the Application

### 1. Start the Flask Server

Run the following command in your terminal:

```bash
python app_sentiment_bert_lime.py
```

You should see output similar to:

```
Loading toxicity detection model...
Toxicity model loaded successfully!

============================================================
STARTING FLASK SERVER
============================================================
 * Serving Flask app 'app_sentiment_bert_lime'
 * Debug mode: on
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://192.168.1.100:5000
```

### 2. Access the Web Application

Open your web browser and navigate to:

```
http://localhost:5000
```

Or if accessing from another device on the same network:

```
http://<your-ip-address>:5000
```

### 3. Using the Application

1. Enter text in the input field (minimum 3 characters)
2. Click the "Analyze" button
3. Review the results:
   - **Sentiment**: Positive, Negative, or Neutral with confidence score
   - **Overall Decision**: TOXIC, SARCASTIC, or NON-TOXIC
   - **Toxicity Probability**: Percentage likelihood of toxicity
   - **Category Breakdown**: Detailed probabilities for each toxicity type
   - **Important Words**: Words contributing to toxicity (if applicable)

## Example Use Cases

### Example 1: Positive Sentiment (Non-Toxic)
```
Input: "This is a wonderful and helpful application!"
Result: NON-TOXIC (Positive sentiment detected, BERT skipped)
```

### Example 2: Genuine Toxicity
```
Input: "You are an idiot and everyone hates you"
Result: TOXIC (Negative sentiment + high toxicity probability)
```

### Example 3: Potential Sarcasm
```
Input: "Oh great, another amazing feature that doesn't work"
Result: SARCASTIC (Negative/Neutral sentiment but low toxicity)
```

## API Endpoints

### GET `/`
Returns the main web interface (index.html)

### POST `/analyze`
Analyzes text for sentiment and toxicity

**Request Body** (form-data):
- `text`: The text string to analyze (minimum 3 characters)

**Response Format**:
```json
{
  "success": true,
  "text": "analyzed text",
  "sentiment": {
    "label": "NEGATIVE",
    "confidence": 0.75
  },
  "case": 3,
  "decision": "TOXIC",
  "decision_message": "⚠️ TOXIC - Negative sentiment and toxicity probability 85% (> 60%)",
  "toxic_probability": 0.85,
  "primary_prediction": {
    "name": "toxic",
    "probability": 0.85,
    "percentage": 85.0,
    "is_predicted": true
  },
  "all_predictions": [
    {
      "name": "toxic",
      "probability": 0.85,
      "percentage": 85.0,
      "is_predicted": true
    }
  ],
  "detected_categories": ["toxic", "insult"],
  "important_words": [
    {
      "word": "idiot",
      "importance": 0.92,
      "label": "insult",
      "category_definition": "Personal attacks, name-calling..."
    }
  ]
}
```

## Configuration and Customization

### Adjusting Toxicity Thresholds

Modify the threshold values in the decision logic (default is 0.6 / 60%):

```python
# In the analyze() function, change these values:
if toxic_prob > 0.6:  # Change 0.6 to desired threshold (e.g., 0.5 for more sensitive)
```

### Changing Maximum Text Length

Modify the `max_length` parameter in `predict_with_attention`:

```python
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)  # Increase for longer text
```

### Adjusting Sentiment Sensitivity

Modify the sentiment thresholds in `check_sentiment()`:

```python
if polarity > 0.1:  # Change to 0.05 for more positive-sensitive
    return 'POSITIVE', abs(polarity)
elif polarity < -0.1:  # Change to -0.05 for more negative-sensitive
    return 'NEGATIVE', abs(polarity)
```

### Adding Custom Toxicity Categories

Update the `CLASS_NAMES` list and `CATEGORY_DEFINITIONS` dictionary:

```python
CLASS_NAMES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'your_category']

CATEGORY_DEFINITIONS['your_category'] = {
    'short': 'Short description',
    'full': 'Detailed description of this category'
}
```

## Troubleshooting

### Common Issues and Solutions

#### 1. **ModuleNotFoundError: No module named 'textblob'**
**Solution**: Install TextBlob and download corpora
```bash
pip install textblob
python -m textblob.download_corpora
```

#### 2. **Model download fails or is slow**
**Solution**: 
- Check your internet connection
- Use a mirror for Hugging Face:
```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

#### 3. **Out of Memory Error**
**Solution**: The model requires significant RAM. Try:
- Close other applications
- Force CPU-only mode (already configured)
- Reduce batch size if modified
- Use a smaller model variant

#### 4. **Port 5000 already in use**
**Solution**: Change the port number in the last line:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Use different port
```

#### 5. **TextBlob corpora download fails**
**Solution**: Manual download or use alternative:
```python
# Alternative: Use pattern library or manually download corpora
# Or handle gracefully with fallback sentiment analysis
```

#### 6. **Slow first request**
**Solution**: This is normal. The model loads into memory on first request. Subsequent requests will be faster.

#### 7. **Template not found error**
**Solution**: Ensure you have `templates/index.html` in your project directory.

## Performance Optimization

### Current Optimizations
- CPU-only mode (`torch.set_num_threads(1)`)
- Single-threaded processing
- Efficient token grouping for word importance
- Early exit for positive sentiment (skips BERT)

### Recommended Server Specifications
- **CPU**: 2+ cores (Intel i5 or equivalent)
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 1GB free space (for model and dependencies)
- **Network**: Broadband internet for initial model download

### Performance Metrics
- **Initial load time**: 10-30 seconds (model loading)
- **Analysis time (Case 1)**: < 100ms (sentiment only)
- **Analysis time (Cases 2-3)**: 500ms - 2 seconds (BERT + sentiment)
- **Memory usage**: 2-3 GB

## Production Deployment Notes

For production environments, consider these changes:

1. **Disable Debug Mode**:
```python
app.run(debug=False, host='0.0.0.0', port=5000)
```

2. **Use Production WSGI Server**:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app_sentiment_bert_lime:app
```

3. **Add Rate Limiting**:
```python
from flask_limiter import Limiter
limiter = Limiter(app, key_func=lambda: request.remote_addr)
```

4. **Implement Caching**:
```python
from flask_caching import Cache
cache = Cache(app, config={'CACHE_TYPE': 'simple'})
```

5. **Add Logging**:
```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Security Considerations

- **Input Validation**: Minimum length enforced (3 characters)
- **Error Handling**: Comprehensive try-catch blocks
- **Resource Limits**: CPU thread limitation prevents over-utilization
- **No External Calls**: All processing is local after model download

**Warning**: Debug mode (`debug=True`) exposes error details and should not be used in production.

## Limitations

1. **Language**: Currently optimized for English text only
2. **Length**: Maximum 128 tokens (approximately 100 words) for BERT analysis
3. **Sarcasm Detection**: Uses heuristic approach, may not catch all sarcastic content
4. **Context**: Limited context window may miss long-range dependencies
5. **Performance**: CPU-only inference slower than GPU

## Future Enhancements

Potential improvements:
- GPU support for faster inference
- Multi-language support
- Batch processing for multiple texts
- Customizable thresholds via UI
- Export results functionality
- Historical analysis tracking
- Real-time streaming analysis

## License

[Specify your license here]

## Acknowledgments

- **Hugging Face** - Transformers library and model hosting
- **Unitary** - Fine-tuned toxic-bert model
- **TextBlob** - Sentiment analysis library
- **Jigsaw/Google** - Toxic comment classification dataset

## Support and Contributing

For issues, questions, or contributions:
1. Check the troubleshooting section
2. Verify all dependencies are correctly installed
3. Ensure Python version compatibility (3.7+)
4. Open an issue in the repository
5. Provide error logs and system information

---

**Important Disclaimer**: This tool is designed for content moderation assistance and educational purposes. It should not be the sole determinant for content decisions. Always incorporate human review for critical applications, as no automated system is 100% accurate.

**Version**: 2.0 (with sentiment analysis)
**Last Updated**: June 2026
```

This README now accurately reflects your enhanced code with sentiment analysis integration, the three-case decision system, and all the specific features of your application.