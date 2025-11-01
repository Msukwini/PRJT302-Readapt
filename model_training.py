"""
Readapt Adaptive Learning System - Model Training & Evaluation (GPU-Enabled)
Implements GPT-2 unsupervised post-training and SVM classification with GPU support
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from datetime import datetime

# ML Libraries
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import joblib

# Deep Learning (if available)
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer
    TORCH_AVAILABLE = True
    
    # GPU Detection and Setup
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        GPU_NAME = torch.cuda.get_device_name(0)
        GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🎮 GPU DETECTED: {GPU_NAME}")
        print(f"💾 GPU Memory: {GPU_MEMORY:.2f} GB")
        print(f"✅ Using CUDA device: {DEVICE}")
    else:
        DEVICE = torch.device("cpu")
        print("⚠️  No GPU detected, using CPU")
        print("   For faster training, consider using Google Colab or a GPU-enabled machine")
    
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None
    print("⚠️  PyTorch/Transformers not available. Will train SVM only.")


# Try cuML for GPU-accelerated SVM (RAPIDS)
try:
    from cuml.svm import SVC as cuSVC
    from cuml.preprocessing import StandardScaler as cuStandardScaler
    from cuml.ensemble import RandomForestClassifier as cuRandomForestClassifier
    import cupy as cp
    CUML_AVAILABLE = True
    print("🚀 cuML (RAPIDS) detected - GPU-accelerated SVM available!")
except ImportError:
    CUML_AVAILABLE = False
    print("ℹ️  cuML not available. Using scikit-learn (CPU) for SVM")
    print("   Install RAPIDS for GPU-accelerated SVM: conda install -c rapidsai -c conda-forge cuml")


# ===========================
# GPU UTILITY FUNCTIONS
# ===========================

def print_gpu_memory_usage():
    """Print current GPU memory usage"""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"📊 GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")


def clear_gpu_cache():
    """Clear GPU cache to free up memory"""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("🧹 GPU cache cleared")


# ===========================
# 1. GPT-2 UNSUPERVISED POST-TRAINING (GPU-ENABLED)
# ===========================

class MaskedLanguageModelingDataset(Dataset):
    """Dataset for GPT-2 unsupervised post-training with masked tokens"""
    
    def __init__(self, data_file: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load processed data
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data)} samples for training")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Use masked text for training (implements noun-entity-aware masking)
        masked_text = item['masked_text']
        
        # Tokenize
        encoding = self.tokenizer(
            masked_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()  # For language modeling
        }


class GPT2PostTrainer:
    """Implements unsupervised post-training for GPT-2 with GPU support"""
    
    def __init__(self, model_name: str = "gpt2", output_dir: str = "models"):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        if not TORCH_AVAILABLE:
            print("⚠️  Skipping GPT-2 training (PyTorch not available)")
            return
        
        # Initialize tokenizer and model
        print(f"📥 Loading {model_name}...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Move model to GPU
        self.model = self.model.to(DEVICE)
        
        param_count = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"✅ Model loaded: {param_count:.1f}M parameters")
        print(f"📍 Model device: {next(self.model.parameters()).device}")
        
        if torch.cuda.is_available():
            print_gpu_memory_usage()
    
    def train(self, train_file: str, val_file: str, 
              epochs: int = 3, batch_size: int = 4, learning_rate: float = 5e-5):
        """Train GPT-2 with unsupervised post-training on GPU"""
        
        if not TORCH_AVAILABLE:
            return None
        
        print("\n" + "=" * 60)
        print("🚀 Starting GPT-2 Unsupervised Post-Training")
        if torch.cuda.is_available():
            print(f"🎮 Training on GPU: {GPU_NAME}")
        print("=" * 60)
        
        # Adjust batch size based on GPU memory
        if torch.cuda.is_available():
            if GPU_MEMORY >= 16:
                batch_size = 8
                print(f"🔥 Large GPU detected, increasing batch size to {batch_size}")
            elif GPU_MEMORY >= 8:
                batch_size = 4
                print(f"💪 Using batch size: {batch_size}")
            else:
                batch_size = 2
                print(f"⚠️  Limited GPU memory, reducing batch size to {batch_size}")
        
        # Create datasets
        train_dataset = MaskedLanguageModelingDataset(train_file, self.tokenizer)
        val_dataset = MaskedLanguageModelingDataset(val_file, self.tokenizer)
        
        # Training arguments with GPU optimizations
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / "gpt2_checkpoints"),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            save_total_limit=2,
            report_to="none",
            # GPU optimizations
            fp16=torch.cuda.is_available(),  # Mixed precision training on GPU
            dataloader_num_workers=4 if torch.cuda.is_available() else 0,
            gradient_accumulation_steps=2 if not torch.cuda.is_available() else 1,
        )
        
        print(f"\n⚙️  Training Configuration:")
        print(f"   • Epochs: {epochs}")
        print(f"   • Batch size: {batch_size}")
        print(f"   • Learning rate: {learning_rate}")
        print(f"   • Mixed precision (FP16): {training_args.fp16}")
        print(f"   • Device: {DEVICE}")
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )
        
        # Train
        print("\n📚 Training started...")
        if torch.cuda.is_available():
            print_gpu_memory_usage()
        
        train_result = trainer.train()
        
        if torch.cuda.is_available():
            print_gpu_memory_usage()
            clear_gpu_cache()
        
        # Save model
        save_path = self.output_dir / "gpt2_finetuned"
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"✅ Model saved to: {save_path}")
        
        # Get training metrics
        metrics = {
            'train_loss': train_result.training_loss,
            'epochs': epochs,
            'train_samples': len(train_dataset),
            'device': str(DEVICE),
            'gpu_name': GPU_NAME if torch.cuda.is_available() else 'CPU',
            'fp16_enabled': training_args.fp16
        }
        
        return metrics
    
    def generate_passage(self, prompt: str, max_length: int = 200, difficulty: str = "intermediate"):
        """Generate adaptive reading passage on GPU"""
        
        if not TORCH_AVAILABLE:
            return "GPT-2 generation not available (PyTorch not installed)"
        
        # Add difficulty context to prompt
        difficulty_prompts = {
            'easy': "Write a simple passage about ",
            'intermediate': "Write an educational passage about ",
            'high': "Write an advanced academic passage about "
        }
        
        full_prompt = difficulty_prompts.get(difficulty, "") + prompt
        
        inputs = self.tokenizer(full_prompt, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}  # Move to GPU
        
        with torch.no_grad():  # No gradient calculation for inference
            outputs = self.model.generate(
                inputs['input_ids'],
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text


# ===========================
# 2. SVM DIFFICULTY CLASSIFICATION (GPU-ENABLED)
# ===========================

class SVMClassifier:
    """SVM for difficulty level prediction with optional GPU acceleration"""
    
    def __init__(self, output_dir: str = "models", use_gpu: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.use_gpu = use_gpu and CUML_AVAILABLE
        
        if self.use_gpu:
            print("🚀 Initializing GPU-accelerated SVM (cuML)")
            self.scaler = cuStandardScaler()
            self.svm_model = cuSVC(kernel='rbf', C=10, gamma='scale', probability=True)
            self.rf_model = cuRandomForestClassifier(n_estimators=100, random_state=42)
        else:
            print("💻 Initializing CPU-based SVM (scikit-learn)")
            self.scaler = StandardScaler()
            self.svm_model = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
            self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Train SVM classifier with GPU acceleration if available"""
        
        print("\n" + "=" * 60)
        print("🎯 Training SVM Difficulty Classifier")
        if self.use_gpu:
            print("🚀 Using GPU-accelerated training (RAPIDS cuML)")
        else:
            print("💻 Using CPU training (scikit-learn)")
        print("=" * 60)
        
        # Convert to GPU arrays if using cuML
        if self.use_gpu:
            X_train = cp.array(X_train)
            y_train = cp.array(y_train)
            X_val = cp.array(X_val)
            y_val = cp.array(y_val)
        
        # Scale features
        print("\n🔄 Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train SVM
        print("🔄 Training SVM (RBF kernel)...")
        self.svm_model.fit(X_train_scaled, y_train)
        
        # Train Random Forest for comparison
        print("🔄 Training Random Forest...")
        self.rf_model.fit(X_train_scaled, y_train)
        
        # Evaluate both models
        svm_train_pred = self.svm_model.predict(X_train_scaled)
        svm_val_pred = self.svm_model.predict(X_val_scaled)
        
        rf_train_pred = self.rf_model.predict(X_train_scaled)
        rf_val_pred = self.rf_model.predict(X_val_scaled)
        
        # Convert back to numpy for metrics if using cuML
        if self.use_gpu:
            y_train = cp.asnumpy(y_train)
            y_val = cp.asnumpy(y_val)
            svm_train_pred = cp.asnumpy(svm_train_pred)
            svm_val_pred = cp.asnumpy(svm_val_pred)
            rf_train_pred = cp.asnumpy(rf_train_pred)
            rf_val_pred = cp.asnumpy(rf_val_pred)
        
        # Calculate metrics
        metrics = {
            'svm': {
                'train_accuracy': accuracy_score(y_train, svm_train_pred),
                'val_accuracy': accuracy_score(y_val, svm_val_pred),
                'train_f1': f1_score(y_train, svm_train_pred, average='weighted'),
                'val_f1': f1_score(y_val, svm_val_pred, average='weighted'),
                'train_precision': precision_score(y_train, svm_train_pred, average='weighted', zero_division=0),
                'val_precision': precision_score(y_val, svm_val_pred, average='weighted', zero_division=0),
                'train_recall': recall_score(y_train, svm_train_pred, average='weighted', zero_division=0),
                'val_recall': recall_score(y_val, svm_val_pred, average='weighted', zero_division=0),
                'using_gpu': self.use_gpu
            },
            'random_forest': {
                'train_accuracy': accuracy_score(y_train, rf_train_pred),
                'val_accuracy': accuracy_score(y_val, rf_val_pred),
                'train_f1': f1_score(y_train, rf_train_pred, average='weighted'),
                'val_f1': f1_score(y_val, rf_val_pred, average='weighted')
            }
        }
        
        # Cross-validation (CPU only - cuML doesn't support cross_val_score)
        if not self.use_gpu:
            print("\n🔄 Performing 5-fold cross-validation...")
            cv_scores = cross_val_score(self.svm_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
            metrics['svm']['cv_accuracy'] = cv_scores.mean()
            metrics['svm']['cv_std'] = cv_scores.std()
        
        # Save models
        joblib.dump(self.svm_model, self.output_dir / 'svm_classifier.pkl')
        joblib.dump(self.rf_model, self.output_dir / 'rf_classifier.pkl')
        joblib.dump(self.scaler, self.output_dir / 'feature_scaler.pkl')
        
        print("\n✅ Models saved successfully")
        
        return metrics
    
    def predict_difficulty(self, features: np.ndarray) -> int:
        """Predict difficulty level for new passage"""
        if self.use_gpu:
            features = cp.array(features)
        
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        prediction = self.svm_model.predict(features_scaled)[0]
        
        if self.use_gpu:
            prediction = cp.asnumpy(prediction)
        
        return int(prediction)
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Get probability distribution over difficulty levels"""
        if self.use_gpu:
            features = cp.array(features)
        
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        proba = self.svm_model.predict_proba(features_scaled)[0]
        
        if self.use_gpu:
            proba = cp.asnumpy(proba)
        
        return proba


# ===========================
# 3. COMPREHENSIVE EVALUATION
# ===========================

class ModelEvaluator:
    """Comprehensive evaluation for conference paper results"""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def evaluate_svm(self, svm_model, scaler, X_test, y_test, use_gpu: bool = False) -> Tuple[Dict, np.ndarray]:
        """Evaluate SVM on test set"""
        
        print("\n" + "=" * 60)
        print("📊 EVALUATING SVM ON TEST SET")
        print("=" * 60)
        
        if use_gpu and CUML_AVAILABLE:
            X_test = cp.array(X_test)
            y_test_gpu = cp.array(y_test)
            
            X_test_scaled = scaler.transform(X_test)
            y_pred_gpu = svm_model.predict(X_test_scaled)
            y_proba_gpu = svm_model.predict_proba(X_test_scaled)
            
            y_pred = cp.asnumpy(y_pred_gpu)
            y_proba = cp.asnumpy(y_proba_gpu)
        else:
            X_test_scaled = scaler.transform(X_test)
            y_pred = svm_model.predict(X_test_scaled)
            y_proba = svm_model.predict_proba(X_test_scaled)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'test_samples': len(y_test)
        }
        
        # Classification report
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        return metrics, cm
    
    def generate_visualizations(self, metrics: Dict, svm_cm: np.ndarray):
        """Generate visualizations for paper"""
        
        print("\n📈 Generating visualizations...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Readapt Model Performance - GPU-Accelerated Training', fontsize=16, fontweight='bold')
        
        # 1. Model Accuracy Comparison
        ax1 = axes[0, 0]
        models = ['SVM\n(GPU)' if metrics['svm'].get('using_gpu') else 'SVM\n(CPU)', 
                  'Random Forest', 'Deep Learning\n(Target)']
        accuracies = [
            metrics['svm']['val_accuracy'] * 100,
            metrics.get('random_forest', {}).get('val_accuracy', 0.85) * 100,
            96.87  # Target from paper
        ]
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        bars = ax1.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
        ax1.set_ylim([0, 105])
        ax1.axhline(y=90, color='gray', linestyle='--', alpha=0.5, label='90% Baseline')
        ax1.grid(axis='y', alpha=0.3)
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax1.legend()
        
        # 2. Performance Metrics Radar
        ax2 = axes[0, 1]
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        svm_values = [
            metrics['svm']['val_accuracy'] * 100,
            metrics['svm']['val_precision'] * 100,
            metrics['svm']['val_recall'] * 100,
            metrics['svm']['val_f1'] * 100
        ]
        x_pos = np.arange(len(metric_names))
        bars = ax2.bar(x_pos, svm_values, color='#3498db', alpha=0.7, edgecolor='black', linewidth=1.5)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(metric_names, rotation=45, ha='right')
        ax2.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
        ax2.set_title('SVM Classifier Metrics', fontsize=13, fontweight='bold')
        ax2.set_ylim([0, 105])
        ax2.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, svm_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 3. Confusion Matrix
        ax3 = axes[1, 0]
        if svm_cm.size > 0:
            im = ax3.imshow(svm_cm, interpolation='nearest', cmap='Blues')
            ax3.figure.colorbar(im, ax=ax3)
            ax3.set_title('Confusion Matrix - SVM', fontsize=13, fontweight='bold')
            ax3.set_xlabel('Predicted Difficulty', fontsize=11)
            ax3.set_ylabel('True Difficulty', fontsize=11)
            
            # Add text annotations
            thresh = svm_cm.max() / 2.
            for i in range(svm_cm.shape[0]):
                for j in range(svm_cm.shape[1]):
                    ax3.text(j, i, format(svm_cm[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if svm_cm[i, j] > thresh else "black",
                            fontweight='bold')
        
        # 4. Hardware Acceleration Info
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Display system info
        info_text = "🚀 GPU-ACCELERATED TRAINING\n\n"
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            info_text += f"GPU: {GPU_NAME}\n"
            info_text += f"Memory: {GPU_MEMORY:.2f} GB\n"
            info_text += f"CUDA: {torch.version.cuda}\n\n"
        else:
            info_text += "Device: CPU\n\n"
        
        if metrics['svm'].get('using_gpu'):
            info_text += "✅ SVM: GPU-accelerated (cuML)\n"
        else:
            info_text += "💻 SVM: CPU (scikit-learn)\n"
        
        if metrics.get('gpt2', {}).get('fp16_enabled'):
            info_text += "✅ GPT-2: Mixed precision (FP16)\n"
        
        info_text += f"\nTraining Device: {metrics.get('gpt2', {}).get('device', 'N/A')}"
        
        ax4.text(0.5, 0.5, info_text, ha='center', va='center', 
                fontsize=12, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        # Save figure
        fig_path = self.output_dir / 'model_performance_results.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"✅ Visualization saved: {fig_path}")
        
        plt.close()
    
    def generate_results_report(self, all_metrics: Dict):
        """Generate comprehensive results report for paper"""
        
        print("\n" + "=" * 60)
        print("📄 GENERATING RESULTS REPORT FOR CONFERENCE PAPER")
        print("=" * 60)
        
        report = []
        report.append("=" * 60)
        report.append("READAPT ADAPTIVE LEARNING SYSTEM (GPU-ACCELERATED)")
        report.append("Experimental Results - Conference Paper")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 60)
        
        # Hardware Info
        report.append("\n## HARDWARE ACCELERATION")
        report.append("-" * 60)
        if TORCH_AVAILABLE and torch.cuda.is_available():
            report.append(f"GPU: {GPU_NAME}")
            report.append(f"GPU Memory: {GPU_MEMORY:.2f} GB")
            report.append(f"CUDA Version: {torch.version.cuda}")
        else:
            report.append("Device: CPU")
        
        if all_metrics['svm'].get('using_gpu'):
            report.append("SVM Training: GPU-accelerated (RAPIDS cuML)")
        else:
            report.append("SVM Training: CPU (scikit-learn)")
        
        if all_metrics.get('gpt2', {}).get('fp16_enabled'):
            report.append("GPT-2 Training: Mixed precision (FP16)")
        
        # SVM Results
        report.append("\n## 1. SVM DIFFICULTY CLASSIFIER RESULTS")
        report.append("-" * 60)
        svm_metrics = all_metrics['svm']
        report.append(f"Training Accuracy:   {svm_metrics['train_accuracy']*100:.2f}%")
        report.append(f"Validation Accuracy: {svm_metrics['val_accuracy']*100:.2f}%")
        report.append(f"Test Accuracy:       {all_metrics['test_accuracy']*100:.2f}% ← PAPER RESULT")
        report.append(f"F1-Score (weighted): {all_metrics['test_f1']*100:.2f}%")
        report.append(f"Precision:           {all_metrics['test_precision']*100:.2f}%")
        report.append(f"Recall:              {all_metrics['test_recall']*100:.2f}%")
        if 'cv_accuracy' in svm_metrics:
            report.append(f"CV Accuracy (5-fold): {svm_metrics['cv_accuracy']*100:.2f}% ± {svm_metrics['cv_std']*100:.2f}%")
        
        # Comparison with Paper Targets
        report.append("\n## 2. COMPARISON WITH PAPER TARGETS")
        report.append("-" * 60)
        report.append(f"Target SVM Accuracy:        99.00% (from paper)")
        report.append(f"Achieved SVM Accuracy:      {all_metrics['test_accuracy']*100:.2f}%")
        report.append(f"Target Deep Learning Acc:   96.87% (from paper)")
        report.append(f"Target F1-Score:            95.39% (from paper)")
        
        # Data Distribution
        report.append("\n## 3. DATA DISTRIBUTION")
        report.append("-" * 60)
        report.append(f"Training Samples:    {all_metrics.get('train_samples', 'N/A')}")
        report.append(f"Validation Samples:  {all_metrics.get('val_samples', 'N/A')}")
        report.append(f"Test Samples:        {all_metrics['test_samples']}")
        report.append(f"Historical Data:     85.71% (from paper)")
        report.append(f"Real-time Metrics:   42.86% (from paper)")
        
        # Methodology Summary
        report.append("\n## 4. METHODOLOGY IMPLEMENTED")
        report.append("-" * 60)
        report.append("✓ Unsupervised Learning Framework")
        report.append("✓ Noun-Entity-Aware Masking (15% ratio)")
        report.append("✓ Linguistic Feature Extraction (11 features)")
        report.append("✓ SVM with RBF Kernel (C=10)")
        report.append("✓ 70/15/15 Train/Val/Test Split")
        if TORCH_AVAILABLE:
            report.append("✓ GPT-2 Unsupervised Post-Training")
        if all_metrics['svm'].get('using_gpu'):
            report.append("✓ GPU-Accelerated SVM Training")
        if all_metrics.get('gpt2', {}).get('fp16_enabled'):
            report.append("✓ Mixed Precision Training (FP16)")
        
        report.append("\n" + "=" * 60)
        report.append("END OF REPORT")
        report.append("=" * 60)
        
        # Save report
        report_text = "\n".join(report)
        report_path = self.output_dir / 'experimental_results.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        # Also save as JSON
        json_path = self.output_dir / 'experimental_results.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_metrics, f, indent=2)
        
        print(report_text)
        print(f"\n✅ Report saved: {report_path}")
        print(f"✅ JSON metrics saved: {json_path}")


# ===========================
# 4. MAIN TRAINING PIPELINE
# ===========================

def main():
    print("=" * 60)
    print("READAPT MODEL TRAINING PIPELINE (GPU-ACCELERATED)")
    print("Action Steps 2 & 3: Training and Results Generation")
    print("=" * 60)
    
    # Display GPU info
    if TORCH_AVAILABLE and torch.cuda.is_available():
        print(f"\n🎮 GPU: {GPU_NAME}")
        print(f"💾 GPU Memory: {GPU_MEMORY:.2f} GB")
        print(f"🔧 CUDA Version: {torch.version.cuda}")
        print(f"🔧 PyTorch Version: {torch.__version__}")
    else:
        print("\n💻 Running on CPU")
    
    if CUML_AVAILABLE:
        print("🚀 RAPIDS cuML available for GPU-accelerated SVM")
    
    # Load preprocessed data
    print("\n[STEP 1] Loading Preprocessed Data")
    X_train = np.load('processed_data/svm_features_train.npy')
    y_train = np.load('processed_data/svm_labels_train.npy')
    X_val = np.load('processed_data/svm_features_val.npy')
    y_val = np.load('processed_data/svm_labels_val.npy')
    X_test = np.load('processed_data/svm_features_test.npy')
    y_test = np.load('processed_data/svm_labels_test.npy')
    
    print(f"✅ Data loaded:")
    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Val:   {X_val.shape[0]} samples")
    print(f"   Test:  {X_test.shape[0]} samples")
    
    # Train SVM Classifier (with GPU if available)
    print("\n[STEP 2] Training SVM Classifier")
    svm_classifier = SVMClassifier(use_gpu=CUML_AVAILABLE)
    svm_metrics = svm_classifier.train(X_train, y_train, X_val, y_val)
    
    # Train GPT-2 (if available)
    gpt2_metrics = None
    if TORCH_AVAILABLE:
        print("\n[STEP 3] Training GPT-2 with Unsupervised Post-Training")
        if torch.cuda.is_available():
            print("🚀 Using GPU for GPT-2 training")
        gpt2_trainer = GPT2PostTrainer()
        gpt2_metrics = gpt2_trainer.train(
            'processed_data/unsupervised_train_data.json',
            'processed_data/unsupervised_val_data.json',
            epochs=3,
            batch_size=4
        )
    
    # Evaluate on Test Set
    print("\n[STEP 4] Final Evaluation on Test Set")
    evaluator = ModelEvaluator()
    test_metrics, svm_cm = evaluator.evaluate_svm(
        svm_classifier.svm_model,
        svm_classifier.scaler,
        X_test,
        y_test,
        use_gpu=svm_classifier.use_gpu
    )
    
    # Compile all metrics
    all_metrics = {
        'svm': svm_metrics['svm'],
        'random_forest': svm_metrics.get('random_forest', {}),
        'test_accuracy': test_metrics['accuracy'],
        'test_f1': test_metrics['f1_score'],
        'test_precision': test_metrics['precision'],
        'test_recall': test_metrics['recall'],
        'test_samples': test_metrics['test_samples'],
        'train_samples': X_train.shape[0],
        'val_samples': X_val.shape[0]
    }
    
    if gpt2_metrics:
        all_metrics['gpt2'] = gpt2_metrics
    
    # Generate visualizations and report
    print("\n[STEP 5] Generating Results for Conference Paper")
    evaluator.generate_visualizations(all_metrics, svm_cm)
    evaluator.generate_results_report(all_metrics)
    
    # Print summary for paper
    print("\n" + "=" * 60)
    print("🎯 KEY RESULTS FOR CONFERENCE PAPER (Section 6)")
    print("=" * 60)
    print(f"✓ SVM Accuracy:        {test_metrics['accuracy']*100:.2f}%")
    print(f"✓ SVM F1-Score:        {test_metrics['f1_score']*100:.2f}%")
    print(f"✓ Training Samples:    {X_train.shape[0]}")
    print(f"✓ Test Samples:        {X_test.shape[0]}")
    print(f"✓ Feature Dimensions:  {X_train.shape[1]}")
    
    if svm_classifier.use_gpu:
        print(f"✓ GPU-Accelerated:     Yes (RAPIDS cuML)")
    if gpt2_metrics and gpt2_metrics.get('fp16_enabled'):
        print(f"✓ Mixed Precision:     Yes (FP16)")
    
    print("\n📊 Compare with paper targets:")
    print(f"   Paper SVM Target:   99.00%")
    print(f"   Paper DL Target:    96.87% accuracy, 95.39% F1")
    print("\n✅ All results saved to results/ directory")
    print("✅ Models saved to models/ directory")
    
    # GPU memory cleanup
    if TORCH_AVAILABLE and torch.cuda.is_available():
        print("\n🧹 Cleaning up GPU memory...")
        clear_gpu_cache()
        print_gpu_memory_usage()


if __name__ == "__main__":
    main()