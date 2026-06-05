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
git clone https://github.com/Coding-Powers/efficient-toxicity-detection-sentiment-first
cd efficient-toxicity-detection-sentiment-first/
```

### 2. Install Required Dependencies

Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, install manually:

```bash
pip install flask numpy torch transformers textblob
python -m textblob.download_corpora
```

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
```

### 2. Access the Web Application

Open your web browser and navigate to:

```
http://localhost:5000
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

## Acknowledgments

- **Hugging Face** - Transformers library and model hosting
- **Unitary** - Fine-tuned toxic-bert model
- **TextBlob** - Sentiment analysis library
- **Jigsaw/Google** - Toxic comment classification dataset
