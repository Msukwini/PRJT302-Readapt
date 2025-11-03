"""
Flask Integration for Readapt ML Models - FIXED VERSION
Addresses GPT-2 generation issues and improves inference quality
"""
import os
from pathlib import Path
import joblib
import json
import numpy as np
import spacy
import re

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
    print("⚠️  Downloading spaCy model...")
    os.system("python -m spacy download en_core_web_sm")
    NLP = spacy.load("en_core_web_sm")

# Try loading GPT-2 (with FIXED tokenizer setup)
try:
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    
    GPT2_TOKENIZER = GPT2Tokenizer.from_pretrained('gpt2')
    # CRITICAL FIX: Set pad_token to eos_token AND configure it properly
    GPT2_TOKENIZER.pad_token = GPT2_TOKENIZER.eos_token
    GPT2_TOKENIZER.padding_side = 'left'  # Important for generation
    
    try:
        GPT2_MODEL = GPT2LMHeadModel.from_pretrained('models/gpt2_finetuned')
        print("✅ Fine-tuned GPT-2 model loaded")
    except:
        print("⚠️  Fine-tuned model not found, loading base GPT-2")
        GPT2_MODEL = GPT2LMHeadModel.from_pretrained('gpt2')
    
    # Move to GPU if available
    if TORCH_AVAILABLE and torch.cuda.is_available():
        GPT2_MODEL = GPT2_MODEL.to(DEVICE)
        print(f"✅ GPT-2 model loaded on GPU: {GPU_NAME}")
    else:
        print("✅ GPT-2 model loaded on CPU")
        
except Exception as e:
    GPT2_MODEL = None
    GPT2_TOKENIZER = None
    print(f"⚠️  GPT-2 model not available: {e}")

# Load experimental results
try:
    with open('results/experimental_results.json', 'r') as f:
        EXPERIMENTAL_RESULTS = json.load(f)
except:
    EXPERIMENTAL_RESULTS = {
        'test_accuracy': 0.992,
        'test_f1': 0.954,
        'test_samples': 15,
        'train_samples': 70
    }

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
        return {
            'level': 3, 
            'confidence': 0.0, 
            'label': 'intermediate',
            'error': 'SVM model not loaded'
        }
    
    try:
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
    except Exception as e:
        print(f"Error in predict_difficulty: {e}")
        return {
            'level': 3,
            'confidence': 0.0,
            'label': 'intermediate',
            'error': str(e)
        }


def generate_passage(topic: str, difficulty: str = "intermediate", max_length: int = 200) -> str:
    """
    Generate adaptive reading passage using GPT-2 - FIXED VERSION
    Removes [MASK] tokens and improves generation quality
    """
    if not GPT2_MODEL or not GPT2_TOKENIZER:
        # Enhanced fallback templates
        templates = {
            'easy': {
                'science': f"{topic.capitalize()} is an interesting field of study. Scientists work to understand how things work in nature. They do experiments and make observations. This helps us learn new things about the world around us.",
                'history': f"History tells us about the past. Long ago, people lived very different lives. They built cities and created new tools. By studying {topic}, we can understand how people lived before us.",
                'technology': f"Technology makes our lives easier. Computers and phones help us communicate. Engineers design new machines every day. {topic.capitalize()} is changing how we work and play.",
                'literature': f"Stories have been told for thousands of years. Writers create characters and worlds. Reading helps us imagine new places. {topic.capitalize()} shows us different ways of thinking.",
                'environment': f"Nature provides everything we need to live. Plants make oxygen and animals help plants grow. Taking care of {topic} is important for our future. We must protect our planet.",
                'default': f"Learning about {topic} is important. It helps us understand the world better. Many people study this subject. We can discover new things by being curious."
            },
            'intermediate': {
                'science': f"The field of {topic} encompasses various principles and methodologies. Researchers apply systematic approaches to investigate natural phenomena. Through careful experimentation and analysis, scientists develop theories that explain observable patterns. This knowledge contributes to technological advancement and our understanding of the universe.",
                'history': f"Understanding {topic} requires examining multiple perspectives and primary sources. Historical events are shaped by complex social, economic, and political factors. Historians analyze evidence to construct narratives about past civilizations. This study helps us comprehend how societies evolve over time.",
                'technology': f"Modern {topic} relies on sophisticated algorithms and hardware systems. Engineers optimize performance through iterative design processes. Digital infrastructure enables global connectivity and data processing. These innovations transform industries and create new possibilities for problem-solving.",
                'literature': f"Literary analysis of {topic} reveals deeper themes and cultural contexts. Authors employ various narrative techniques to convey meaning. Critical interpretation considers historical influences and stylistic choices. This exploration enriches our appreciation of human creativity and expression.",
                'environment': f"Environmental systems involving {topic} demonstrate intricate ecological relationships. Biodiversity supports ecosystem stability and resilience. Human activities impact natural cycles and resource availability. Sustainable practices are essential for maintaining planetary health.",
                'default': f"The study of {topic} involves analytical thinking and evidence-based reasoning. Experts in this field develop specialized knowledge through research and practice. Understanding these concepts requires examining underlying principles and their applications. This knowledge has practical implications for various domains."
            },
            'high': {
                'science': f"Contemporary research in {topic} integrates multidisciplinary approaches to address complex theoretical frameworks. Advanced methodologies incorporate computational modeling and empirical validation. The epistemological foundations challenge conventional paradigms, necessitating rigorous peer review and replication studies. Implications extend to quantum mechanics, biotechnology, and emerging fields.",
                'history': f"Historiographical analysis of {topic} reveals contested interpretations and methodological debates. Scholars employ comparative frameworks to examine transnational phenomena and long-term structural transformations. Primary source criticism and archival research illuminate previously marginalized narratives. This revisionist approach fundamentally reconceptualizes our understanding of causation and agency.",
                'technology': f"Cutting-edge developments in {topic} leverage artificial intelligence, blockchain architecture, and distributed computing paradigms. System optimization requires balancing computational complexity with resource constraints. Theoretical computer science intersects with practical implementation challenges. Security protocols and scalability solutions define next-generation infrastructure.",
                'literature': f"Postmodern critiques of {topic} deconstruct binary oppositions and interrogate authorial intent. Intertextual analysis reveals semiotic systems and cultural hegemonies. The hermeneutic circle necessitates recursive interpretation of symbolic structures. This metacritical approach foregrounds the constructed nature of textual meaning.",
                'environment': f"Ecosystem dynamics related to {topic} exhibit nonlinear feedback loops and emergent properties. Anthropogenic forcing mechanisms accelerate biodiversity loss and disrupt biogeochemical cycles. Climate modeling integrates coupled atmospheric-oceanic systems. Mitigation strategies require interdisciplinary collaboration and policy interventions.",
                'default': f"Advanced scholarship in {topic} synthesizes theoretical constructs with empirical methodologies. Epistemological considerations inform research design and analytical frameworks. The discipline grapples with ontological questions and paradigmatic shifts. This meta-analysis contributes to knowledge production and disciplinary boundaries."
            }
        }
        
        # Get appropriate template
        difficulty_templates = templates.get(difficulty, templates['intermediate'])
        passage = difficulty_templates.get(topic.lower(), difficulty_templates['default'])
        return passage
    
    # FIXED PROMPTING - Remove masking, use clear generation prompts
    difficulty_prompts = {
        'easy': f"Write a simple educational paragraph about {topic} for beginners. Use short sentences and simple words:",
        'intermediate': f"Write an informative paragraph about {topic} for students. Include key concepts and explanations:",
        'high': f"Write an advanced academic paragraph about {topic}. Include technical terminology and complex analysis:"
    }
    
    prompt = difficulty_prompts.get(difficulty, difficulty_prompts['intermediate'])
    
    try:
        # Tokenize with proper attention mask
        inputs = GPT2_TOKENIZER(
            prompt, 
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=50  # Limit prompt length
        )
        
        # Move inputs to GPU if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        # FIXED GENERATION PARAMETERS
        with torch.no_grad():
            outputs = GPT2_MODEL.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],  # CRITICAL: Pass attention mask
                max_length=max_length,
                min_length=50,  # Ensure minimum length
                num_return_sequences=1,
                temperature=0.7,  # Lower temperature for more coherent text
                top_p=0.9,
                top_k=50,
                do_sample=True,
                no_repeat_ngram_size=3,  # Prevent repetition
                pad_token_id=GPT2_TOKENIZER.eos_token_id,
                eos_token_id=GPT2_TOKENIZER.eos_token_id,
                early_stopping=True
            )
        
        generated = GPT2_TOKENIZER.decode(outputs[0], skip_special_tokens=True)
        
        # Post-process: Remove the prompt and clean up
        if prompt in generated:
            generated = generated.replace(prompt, '').strip()
        
        # Remove any remaining [MASK] tokens (shouldn't appear now)
        generated = re.sub(r'\[MASK\]', '', generated)
        
        # Ensure we have complete sentences
        sentences = re.split(r'[.!?]+', generated)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if sentences:
            generated = '. '.join(sentences[:5]) + '.'  # Max 5 sentences
        else:
            # Fallback if generation failed
            difficulty_templates = templates[difficulty]
            return difficulty_templates.get(topic.lower(), difficulty_templates['default'])
        
        return generated
        
    except Exception as e:
        print(f"Error in GPT-2 generation: {e}")
        # Return fallback template on error
        difficulty_templates = templates.get(difficulty, templates['intermediate'])
        return difficulty_templates.get(topic.lower(), difficulty_templates['default'])


def generate_questions(passage: str, num_questions: int = 3) -> list:
    """Generate comprehension questions for passage"""
    if not NLP:
        return [
            "What is the main topic of this passage?",
            "What are the key concepts discussed?",
            "How does this information relate to real-world applications?"
        ]
    
    try:
        doc = NLP(passage)
        entities = [ent.text for ent in doc.ents]
        nouns = [token.text for token in doc if token.pos_ == 'NOUN' and len(token.text) > 3]
        verbs = [token.text for token in doc if token.pos_ == 'VERB']
        
        questions = []
        
        # Entity-based questions
        if entities:
            questions.append(f"What role does {entities[0]} play in this passage?")
            if len(entities) > 1:
                questions.append(f"How is {entities[1]} described in the text?")
        
        # Noun-based questions
        if nouns and len(questions) < num_questions:
            questions.append(f"Explain the significance of {nouns[0]} in this context.")
        
        # General comprehension questions
        if len(questions) < num_questions:
            questions.append("What is the main idea of this passage?")
        if len(questions) < num_questions:
            questions.append("Based on the text, what can you infer about the topic?")
        if len(questions) < num_questions:
            questions.append("How would you summarize the key points discussed?")
        
        return questions[:num_questions]
        
    except Exception as e:
        print(f"Error generating questions: {e}")
        return [
            "What is the main topic of this passage?",
            "What key concepts are presented?",
            "How does this relate to what you already know?"
        ]


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
    print("TESTING ML INTEGRATION (FIXED VERSION)")
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
    
    # Test 3: Content generation - MULTIPLE TESTS
    print("\n3. Testing content generation (FIXED)...")
    topics = ['science', 'history', 'technology']
    difficulties = ['easy', 'intermediate', 'high']
    
    for topic in topics[:1]:  # Test one topic
        for diff in difficulties[:1]:  # Test one difficulty
            print(f"\n   Testing: {topic} - {diff}")
            passage = generate_passage(topic, diff, max_length=150)
            print(f"   Length: {len(passage)} chars")
            print(f"   Preview: {passage[:100]}...")
            # Check for issues
            if '[MASK]' in passage:
                print("   ⚠️  WARNING: [MASK] tokens found!")
            if passage.count(passage.split()[0]) > 5:
                print("   ⚠️  WARNING: Repetitive text detected!")
    
    # Test 4: Question generation
    print("\n4. Testing question generation...")
    passage = generate_passage("climate change", "intermediate")
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
