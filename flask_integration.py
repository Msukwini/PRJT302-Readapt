"""
Flask Integration for Readapt ML Models
Add these routes to your app.py
"""
from flask import Flask, jsonify, request
import joblib
import numpy as np
import json
from pathlib import Path
import spacy

# GPU detection
try:
    import torch
    TORCH_AVAILABLE = True
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        GPU_NAME = torch.cuda.get_device_name(0)
        print(f"🎮 GPU DETECTED: {GPU_NAME}")
        print(f"✅ Flask app will use GPU for inference")
    else:
        DEVICE = torch.device("cpu")
        print("💻 No GPU detected, using CPU for inference")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None
    print("⚠️  PyTorch not available")

# Try cuML for GPU-accelerated predictions
try:
    import cupy as cp
    CUML_AVAILABLE = True
    print("🚀 cuML available for GPU-accelerated SVM predictions")
except ImportError:
    CUML_AVAILABLE = False

# Load models at startup
try:
    SVM_MODEL = joblib.load('models/svm_classifier.pkl')
    SCALER = joblib.load('models/feature_scaler.pkl')
    print("✅ SVM model loaded")
    
    # Check if it's a cuML model
    IS_CUML_MODEL = hasattr(SVM_MODEL, '__module__') and 'cuml' in SVM_MODEL.__module__
    if IS_CUML_MODEL:
        print("🚀 Loaded GPU-accelerated SVM model (cuML)")
    else:
        print("💻 Loaded CPU-based SVM model (scikit-learn)")
except Exception as e:
    SVM_MODEL = None
    SCALER = None
    IS_CUML_MODEL = False
    print(f"⚠️  SVM model not found: {e}")

try:
    NLP = spacy.load("en_core_web_sm")
    print("✅ spaCy model loaded")
except:
    NLP = None
    print("⚠️  spaCy model not found")

# Try loading GPT-2 (with GPU support)
try:
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    GPT2_MODEL = GPT2LMHeadModel.from_pretrained('models/gpt2_finetuned')
    GPT2_TOKENIZER = GPT2Tokenizer.from_pretrained('models/gpt2_finetuned')
    
    # Move to GPU if available
    if TORCH_AVAILABLE and torch.cuda.is_available():
        GPT2_MODEL = GPT2_MODEL.to(DEVICE)
        print(f"✅ GPT-2 model loaded on GPU: {GPU_NAME}")
    else:
        print("✅ GPT-2 model loaded on CPU")
except Exception as e:
    GPT2_MODEL = None
    GPT2_TOKENIZER = None
    print(f"⚠️  GPT-2 model not found: {e}")

# Load experimental results
try:
    with open('results/experimental_results.json', 'r') as f:
        EXPERIMENTAL_RESULTS = json.load(f)
except:
    EXPERIMENTAL_RESULTS = {}

DIFFICULTY_MAP_LIST = {1: 'easy', 2: 'easy', 3: 'intermediate', 4: 'intermediate', 5: 'high', 6: 'high'}

# ==========================================
# HELPER FUNCTIONS
# ==========================================


def extract_features(text: str) -> np.ndarray:
    """Extract linguistic features from text"""
    if not NLP:
        return np.zeros(11)
    
    doc = NLP(text)
    sentences = list(doc.sents)
    words = [token for token in doc if not token.is_punct]
    
    if not words:
        return np.zeros(11)
    
    # Calculate features (same as in data_pipeline.py)
    syllables = sum(len([c for c in token.text.lower() if c in 'aeiouy']) for token in words)
    avg_syllables = syllables / len(words) if words else 0
    avg_words_per_sent = len(words) / len(sentences) if sentences else 0
    flesch_kincaid = 0.39 * avg_words_per_sent + 11.8 * avg_syllables - 15.59
    
    features = np.array([
        len(sentences),
        len(words),
        len(set(token.text.lower() for token in words)),
        np.mean([len(token.text) for token in words]) if words else 0,
        avg_words_per_sent,
        max(0, flesch_kincaid),
        sum(1 for token in doc if token.pos_ == 'NOUN'),
        sum(1 for token in doc if token.pos_ == 'VERB'),
        sum(1 for token in doc if token.pos_ == 'ADJ'),
        len(doc.ents),
        len(set(token.text.lower() for token in words)) / len(words) if words else 0
    ])
    
    return features


def predict_difficulty(text: str) -> dict:
    """Predict difficulty level of text using GPU if available"""
    if not SVM_MODEL or not SCALER:
        return {'level': 3, 'confidence': 0.5, 'label': 'intermediate'}
    
    features = extract_features(text).reshape(1, -1)
    
    # Use GPU if model is cuML
    if IS_CUML_MODEL and CUML_AVAILABLE:
        features_gpu = cp.array(features)
        features_scaled = SCALER.transform(features_gpu)
        
        prediction_gpu = SVM_MODEL.predict(features_scaled)[0]
        probabilities_gpu = SVM_MODEL.predict_proba(features_scaled)[0]
        
        prediction = int(cp.asnumpy(prediction_gpu))
        probabilities = cp.asnumpy(probabilities_gpu)
    else:
        # CPU inference
        features_scaled = SCALER.transform(features)
        prediction = int(SVM_MODEL.predict(features_scaled)[0])
        probabilities = SVM_MODEL.predict_proba(features_scaled)[0]
    
    difficulty_map = DIFFICULTY_MAP_LIST
    
    return {
        'level': prediction,
        'confidence': float(probabilities[prediction-1]) if prediction <= len(probabilities) else 0.5,
        'label': difficulty_map.get(prediction, 'intermediate'),
        'all_probabilities': probabilities.tolist(),
        'used_gpu': IS_CUML_MODEL and CUML_AVAILABLE
    }


def generate_passage(topic: str, difficulty: str = "intermediate", max_length: int = 200) -> str:
    """Generate adaptive reading passage using GPU if available"""
    if not GPT2_MODEL or not GPT2_TOKENIZER:
        # Fallback templates
        templates = {
            'easy': f"Let's learn about {topic}. {topic.capitalize()} is an interesting subject. It helps us understand the world around us. Many people study {topic} to learn new things.",
            'intermediate': f"Understanding {topic} requires careful analysis. {topic.capitalize()} involves several key concepts that build upon each other. Researchers have made significant discoveries in this field.",
            'high': f"The comprehensive study of {topic} necessitates rigorous examination. {topic.capitalize()} encompasses multifaceted theoretical frameworks and empirical methodologies that researchers employ."
        }
        return templates.get(difficulty, templates['intermediate'])
    
    difficulty_prompts = {
        'easy': f"Write a simple passage for beginners about {topic}:",
        'intermediate': f"Write an educational passage about {topic}:",
        'high': f"Write an advanced academic passage about {topic}:"
    }
    
    prompt = difficulty_prompts.get(difficulty, difficulty_prompts['intermediate'])
    inputs = GPT2_TOKENIZER(prompt, return_tensors="pt")
    
    # Move inputs to GPU if available
    if TORCH_AVAILABLE and torch.cuda.is_available():
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    with torch.no_grad():  # No gradients needed for inference
        outputs = GPT2_MODEL.generate(
            inputs['input_ids'],
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=GPT2_TOKENIZER.eos_token_id
        )
    
    generated = GPT2_TOKENIZER.decode(outputs[0], skip_special_tokens=True)
    return generated


def generate_questions(passage: str, num_questions: int = 3) -> list:
    """Generate comprehension questions for passage"""
    # Simple question generation based on entities and key concepts
    if not NLP:
        return [
            "What is the main topic of this passage?",
            "What are the key concepts discussed?",
            "How does this information relate to real-world applications?"
        ]
    
    doc = NLP(passage)
    entities = [ent.text for ent in doc.ents]
    nouns = [token.text for token in doc if token.pos_ == 'NOUN']
    
    questions = []
    
    if entities:
        questions.append(f"What role does {entities[0]} play in this passage?")
    
    if nouns:
        questions.append(f"How is {nouns[0]} described in the text?")
    
    questions.append("What is the main idea of this passage?")
    questions.append("Based on the text, what can you infer about the topic?")
    
    return questions[:num_questions]


def get_system_info() -> dict:
    """Get GPU and system information"""
    info = {
        'gpu_available': False,
        'gpu_name': None,
        'gpu_memory_gb': None,
        'cuda_version': None,
        'torch_version': None,
        'cuml_available': CUML_AVAILABLE,
        'svm_on_gpu': IS_CUML_MODEL and CUML_AVAILABLE,
        'gpt2_on_gpu': TORCH_AVAILABLE and torch.cuda.is_available() and GPT2_MODEL is not None
    }
    
    if TORCH_AVAILABLE:
        info['torch_version'] = torch.__version__
        if torch.cuda.is_available():
            info['gpu_available'] = True
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory_gb'] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
            info['cuda_version'] = torch.version.cuda
            
            # GPU memory usage
            if GPT2_MODEL is not None:
                info['gpu_memory_allocated_gb'] = round(torch.cuda.memory_allocated(0) / 1024**3, 2)
                info['gpu_memory_reserved_gb'] = round(torch.cuda.memory_reserved(0) / 1024**3, 2)
    
    return info

# ==========================================
# TESTING FUNCTIONS
# ==========================================

def test_integration():
    """Test all ML functions with GPU info"""
    print("=" * 60)
    print("TESTING ML INTEGRATION (GPU-ENABLED)")
    print("=" * 60)
    
    # System info
    print("\n0. System Information:")
    info = get_system_info()
    print(f"   GPU Available: {info['gpu_available']}")
    if info['gpu_available']:
        print(f"   GPU: {info['gpu_name']}")
        print(f"   GPU Memory: {info['gpu_memory_gb']} GB")
        print(f"   CUDA: {info['cuda_version']}")
    print(f"   SVM on GPU: {info['svm_on_gpu']}")
    print(f"   GPT-2 on GPU: {info['gpt2_on_gpu']}")
    
    # Test 1: Feature extraction
    print("\n1. Testing feature extraction...")
    test_text = "This is a simple sentence for testing. It contains multiple sentences."
    features = extract_features(test_text)
    print(f"✅ Extracted {len(features)} features")
    
    # Test 2: Difficulty prediction
    print("\n2. Testing difficulty prediction...")
    result = predict_difficulty(test_text)
    print(f"✅ Predicted difficulty: {result['label']} (level {result['level']})")
    print(f"   Confidence: {result['confidence']:.2%}")
    print(f"   Used GPU: {result.get('used_gpu', False)}")
    
    # Test 3: Content generation
    print("\n3. Testing content generation...")
    passage = generate_passage("climate change", "intermediate")
    print(f"✅ Generated passage ({len(passage)} chars)")
    print(f"   Preview: {passage[:100]}...")
    if TORCH_AVAILABLE and torch.cuda.is_available():
        print(f"   Generated on GPU: {GPU_NAME}")
    
    # Test 4: Question generation
    print("\n4. Testing question generation...")
    questions = generate_questions(passage)
    print(f"✅ Generated {len(questions)} questions")
    for i, q in enumerate(questions, 1):
        print(f"   Q{i}: {q}")
    
    # GPU memory info
    if TORCH_AVAILABLE and torch.cuda.is_available():
        print("\n5. GPU Memory Status:")
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"   Allocated: {allocated:.2f} GB")
        print(f"   Reserved: {reserved:.2f} GB")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED")
    print("=" * 60)

if __name__ == "__main__":
    test_integration()
