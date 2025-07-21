#!/usr/bin/env python3
"""
HINGLISH CHATBOT SYSTEM
Production-ready chatbot with dataset loading from hinglish.txt
Simple and clean implementation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
import math
import numpy as np
from collections import Counter
import re
import warnings
import os
import time
import gc
from typing import List, Optional, Dict, Tuple, Any
from tqdm.auto import tqdm
from dataclasses import dataclass
import unicodedata
import random
import pickle
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

# ==========================================
# DEVICE SETUP
# ==========================================

def setup_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        gpu_props = torch.cuda.get_device_properties(0)
        print(f"🔥 GPU: {gpu_props.name} ({gpu_props.total_memory / 1e9:.1f} GB)")
    else:
        print("⚠️ Using CPU")
    
    return device

device = setup_device()

# ==========================================
# CONFIGURATIONS
# ==========================================

@dataclass
class ModelConfig:
    vocab_size: int = 3000
    d_model: int = 512
    n_heads: int = 16
    n_layers: int = 8
    d_ff: int = 2048
    max_seq_len: int = 128
    dropout: float = 0.1

@dataclass 
class TrainingConfig:
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_epochs: int = 50
    early_stopping_patience: int = 10
    use_mixed_precision: bool = True

@dataclass
class TokenizerConfig:
    min_freq: int = 2
    max_vocab: int = 3000
    special_tokens: Dict[str, int] = None
    
    def __post_init__(self):
        if self.special_tokens is None:
            self.special_tokens = {
                '<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3
            }

@dataclass
class GenerationConfig:
    max_length: int = 30
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.9
    repetition_penalty: float = 1.1

# ==========================================
# DATASET LOADER
# ==========================================

def load_dataset_from_file(filepath: str = "hinglish.txt") -> List[str]:
    """Load dataset from hinglish.txt file"""
    if not os.path.exists(filepath):
        print(f"❌ File {filepath} not found!")
        print("Please create hinglish.txt with your Hinglish sentences (one per line)")
        return []
    
    texts = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and len(line.split()) >= 3:  # Minimum 3 words
                    texts.append(line)
        
        print(f"📚 Loaded {len(texts)} sentences from {filepath}")
        return texts
    
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return []

# ==========================================
# TOKENIZER
# ==========================================

class HinglishTokenizer:
    def __init__(self, config: TokenizerConfig):
        self.config = config
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        self.special_tokens = config.special_tokens.copy()
        self.word_freq = Counter()
        
    def preprocess_text(self, text: str) -> str:
        if not text:
            return ""
        
        # Unicode normalization
        text = unicodedata.normalize('NFKC', text.lower())
        
        # Add spaces around punctuation
        text = re.sub(r'([।॥])', r' \1 ', text)
        text = re.sub(r'([.!?,:;])', r' \1 ', text)
        
        # Keep Hindi, English, punctuation
        text = re.sub(r'[^\u0900-\u097F\w\s.,!?।॥\-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        processed = self.preprocess_text(text)
        if not processed:
            return []
        return [token for token in processed.split() if token.strip()]
    
    def build_vocabulary(self, texts: List[str]):
        print("🔤 Building vocabulary...")
        
        # Collect word frequencies
        all_words = []
        for text in texts:
            words = self.tokenize(text)
            all_words.extend(words)
            self.word_freq.update(words)
        
        # Start with special tokens
        self.word_to_id = self.special_tokens.copy()
        self.id_to_word = {v: k for k, v in self.special_tokens.items()}
        
        # Add frequent words
        current_id = len(self.special_tokens)
        for word, freq in self.word_freq.most_common():
            if freq >= self.config.min_freq and current_id < self.config.max_vocab:
                if word not in self.word_to_id:
                    self.word_to_id[word] = current_id
                    self.id_to_word[current_id] = word
                    current_id += 1
        
        self.vocab_size = len(self.word_to_id)
        print(f"✅ Vocabulary built: {self.vocab_size} words")
        
        # Show some stats
        hindi_words = sum(1 for w in self.word_to_id.keys() 
                         if any('\u0900' <= c <= '\u097F' for c in w))
        english_words = sum(1 for w in self.word_to_id.keys() 
                           if w.isalpha() and not any('\u0900' <= c <= '\u097F' for c in w))
        
        print(f"📊 Hindi words: {hindi_words}, English words: {english_words}")
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = self.tokenize(text)
        token_ids = []
        
        if add_special_tokens:
            token_ids.append(self.word_to_id['<BOS>'])
        
        for token in tokens:
            token_id = self.word_to_id.get(token, self.word_to_id['<UNK>'])
            token_ids.append(token_id)
        
        if add_special_tokens:
            token_ids.append(self.word_to_id['<EOS>'])
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        words = []
        special_ids = set(self.special_tokens.values())
        
        for token_id in token_ids:
            if skip_special_tokens and token_id in special_ids:
                continue
            
            word = self.id_to_word.get(token_id, '<UNK>')
            if word != '<UNK>':
                words.append(word)
        
        text = ' '.join(words)
        
        # Clean up spacing
        text = re.sub(r' ([.!?,:;।॥])', r'\1', text)
        return text.strip()

# ==========================================
# DATASET CLASS
# ==========================================

class HinglishDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer: HinglishTokenizer, max_length: int = 64):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        print(f"📊 Processing {len(texts)} texts...")
        
        for text in tqdm(texts, desc="Processing"):
            tokens = tokenizer.encode(text)
            
            if len(tokens) >= 4:  # Minimum length
                padded = self._pad_sequence(tokens)
                self.data.append(padded)
        
        print(f"✅ Dataset ready: {len(self.data)} samples")
    
    def _pad_sequence(self, tokens: List[int]) -> List[int]:
        pad_id = self.tokenizer.word_to_id['<PAD>']
        eos_id = self.tokenizer.word_to_id['<EOS>']
        
        if len(tokens) > self.max_length:
            return tokens[:self.max_length-1] + [eos_id]
        else:
            return tokens + [pad_id] * (self.max_length - len(tokens))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = self.data[idx]
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        targets = torch.tensor(sequence[1:], dtype=torch.long)
        return input_ids, targets

# ==========================================
# TRANSFORMER MODEL
# ==========================================

class SimpleTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        
        # Output
        self.ln_final = nn.LayerNorm(config.d_model)
        self.output_projection = nn.Linear(config.d_model, config.vocab_size)
        
        # Initialize
        self.apply(self._init_weights)
        
        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"🧠 Model: {param_count:,} parameters")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def create_causal_mask(self, seq_len: int, device: torch.device):
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)
        return mask
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Position IDs
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_emb = self.token_embedding(input_ids) * math.sqrt(self.config.d_model)
        pos_emb = self.position_embedding(positions)
        x = self.dropout(token_emb + pos_emb)
        
        # Causal mask
        mask = self.create_causal_mask(seq_len, device)
        
        # Transformer
        x = self.transformer(x, mask=mask, is_causal=True)
        
        # Output
        x = self.ln_final(x)
        logits = self.output_projection(x)
        
        return logits

# ==========================================
# TEXT GENERATOR
# ==========================================

class TextGenerator:
    def __init__(self, model: SimpleTransformer, tokenizer: HinglishTokenizer, config: GenerationConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
    
    def generate(self, prompt: str, **kwargs) -> str:
        # Update config with kwargs
        gen_config = GenerationConfig(**{**self.config.__dict__, **kwargs})
        
        self.model.eval()
        device = next(self.model.parameters()).device
        
        with torch.no_grad():
            # Encode prompt
            input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
            
            generated_tokens = input_ids.copy()
            
            for _ in range(gen_config.max_length):
                if input_tensor.size(1) >= self.model.config.max_seq_len:
                    break
                
                # Forward pass
                logits = self.model(input_tensor)
                next_token_logits = logits[0, -1, :].float()
                
                # Apply repetition penalty
                if gen_config.repetition_penalty != 1.0:
                    for token_id in set(generated_tokens):
                        if next_token_logits[token_id] < 0:
                            next_token_logits[token_id] *= gen_config.repetition_penalty
                        else:
                            next_token_logits[token_id] /= gen_config.repetition_penalty
                
                # Temperature
                if gen_config.temperature != 1.0:
                    next_token_logits = next_token_logits / gen_config.temperature
                
                # Top-k filtering
                if gen_config.top_k > 0:
                    k = min(gen_config.top_k, next_token_logits.size(-1))
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, k)
                    filtered_logits = torch.full_like(next_token_logits, float('-inf'))
                    filtered_logits[top_k_indices] = top_k_logits
                    next_token_logits = filtered_logits
                
                # Top-p filtering
                if gen_config.top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > gen_config.top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                probs[self.tokenizer.word_to_id['<PAD>']] = 0.0
                probs = probs / probs.sum()
                
                next_token = torch.multinomial(probs, num_samples=1).item()
                
                # Check EOS
                if next_token == self.tokenizer.word_to_id['<EOS>']:
                    break
                
                # Add token
                generated_tokens.append(next_token)
                next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
                input_tensor = torch.cat([input_tensor, next_token_tensor], dim=1)
            
            return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

# ==========================================
# TRAINING SYSTEM
# ==========================================

class Trainer:
    def __init__(self, model: SimpleTransformer, tokenizer: HinglishTokenizer, config: TrainingConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Setup scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.max_epochs, eta_min=1e-6
        )
        
        # Setup loss
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.word_to_id['<PAD>'])
        
        # Setup mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.use_mixed_precision else None
        
        # Training state
        self.best_loss = float('inf')
        self.patience = 0
        
        print("🚀 Trainer initialized")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        print(f"🔥 Starting training for {self.config.max_epochs} epochs")
        
        for epoch in range(self.config.max_epochs):
            # Train
            train_loss = self._train_epoch(train_loader, epoch)
            
            # Validate
            val_loss = self._validate_epoch(val_loader, epoch)
            
            # Schedule
            self.scheduler.step()
            lr = self.optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch+1}/{self.config.max_epochs} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {lr:.6f}")
            
            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.patience = 0
                self.save_model('best_hinglish_chatbot.pth')
                print(f"💾 Best model saved! Loss: {val_loss:.4f}")
            else:
                self.patience += 1
            
            # Early stopping
            if self.patience >= self.config.early_stopping_patience:
                print(f"⏹️ Early stopping at epoch {epoch+1}")
                break
            
            # Sample generation
            if (epoch + 1) % 5 == 0:
                self._generate_sample()
        
        print("✅ Training completed!")
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for input_ids, targets in pbar:
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=(self.scaler is not None)):
                logits = self.model(input_ids)
                loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(train_loader)
    
    def _validate_epoch(self, val_loader: DataLoader, epoch: int) -> float:
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
            
            for input_ids, targets in pbar:
                input_ids = input_ids.to(device)
                targets = targets.to(device)
                
                with torch.cuda.amp.autocast(enabled=(self.scaler is not None)):
                    logits = self.model(input_ids)
                    loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(val_loader)
    
    def _generate_sample(self):
        generator = TextGenerator(self.model, self.tokenizer, GenerationConfig())
        
        test_prompts = ["yaar", "kya haal", "college mein"]
        
        for prompt in test_prompts:
            try:
                generated = generator.generate(prompt, max_length=15, temperature=0.8)
                print(f"📝 '{prompt}': {generated}")
            except:
                pass
    
    def save_model(self, filepath: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'tokenizer_word_to_id': self.tokenizer.word_to_id,
            'tokenizer_id_to_word': self.tokenizer.id_to_word,
            'tokenizer_vocab_size': self.tokenizer.vocab_size,
            'model_config': self.model.config.__dict__
        }, filepath)

# ==========================================
# CHATBOT CLASS
# ==========================================

class HinglishChatbot:
    def __init__(self, model_path: str = "best_hinglish_chatbot.pth"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.generator = None
        
        self.load_model()
    
    def load_model(self):
        if not os.path.exists(self.model_path):
            print(f"❌ Model file {self.model_path} not found!")
            print("Please train the model first by running: python chatbot.py --train")
            return False
        
        try:
            print("📚 Loading trained model...")
            checkpoint = torch.load(self.model_path, map_location=device)
            
            # Rebuild tokenizer
            tokenizer_config = TokenizerConfig()
            self.tokenizer = HinglishTokenizer(tokenizer_config)
            self.tokenizer.word_to_id = checkpoint['tokenizer_word_to_id']
            self.tokenizer.id_to_word = checkpoint['tokenizer_id_to_word']
            self.tokenizer.vocab_size = checkpoint['tokenizer_vocab_size']
            
            # Rebuild model
            model_config = ModelConfig(**checkpoint['model_config'])
            self.model = SimpleTransformer(model_config).to(device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Setup generator
            gen_config = GenerationConfig()
            self.generator = TextGenerator(self.model, self.tokenizer, gen_config)
            
            print("✅ Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def chat(self, user_input: str) -> str:
        if not self.generator:
            return "❌ Model not loaded. Please train first."
        
        try:
            # Generate response
            response = self.generator.generate(
                user_input,
                max_length=25,
                temperature=0.8,
                top_k=40,
                top_p=0.9
            )
            
            # Clean up response
            response = response.strip()
            if not response:
                return "Sorry, main kuch samajh nahi paya."
            
            return response
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def interactive_chat(self):
        print("\n" + "="*60)
        print("🤖 HINGLISH CHATBOT - Interactive Mode")
        print("="*60)
        print("Type 'quit', 'exit' or 'bye' to stop chatting")
        print("Type 'help' for usage tips")
        print("-"*60)
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    print("🤖 Bot: Bye! Take care yaar! 👋")
                    break
                
                if user_input.lower() == 'help':
                    print("🤖 Bot: Main Hinglish mein baat karta hun!")
                    print("      Try: 'kya haal hai', 'college kaisa hai', 'weekend plans'")
                    continue
                
                # Generate response
                response = self.chat(user_input)
                print(f"🤖 Bot: {response}")
                
            except KeyboardInterrupt:
                print("\n🤖 Bot: Bye! Take care! 👋")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

# ==========================================
# MAIN FUNCTIONS
# ==========================================

def train_model():
    """Train the chatbot model"""
    print("🚀 HINGLISH CHATBOT TRAINING")
    print("="*50)
    
    # Load dataset
    texts = load_dataset_from_file("hinglish.txt")
    if not texts:
        print("❌ No dataset found. Please create hinglish.txt file.")
        return
    
    if len(texts) < 10:
        print("⚠️ Dataset too small. Add more sentences to hinglish.txt")
        return
    
    # Initialize configs
    model_config = ModelConfig()
    train_config = TrainingConfig()
    tokenizer_config = TokenizerConfig()
    
    # Build tokenizer
    tokenizer = HinglishTokenizer(tokenizer_config)
    tokenizer.build_vocabulary(texts)
    
    # Update model config
    model_config.vocab_size = tokenizer.vocab_size
    
    # Create dataset
    dataset = HinglishDataset(texts, tokenizer, max_length=64)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=train_config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_config.batch_size, shuffle=False)
    
    print(f"📊 Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Initialize model
    model = SimpleTransformer(model_config).to(device)
    
    # Initialize trainer
    trainer = Trainer(model, tokenizer, train_config)
    
    # Train
    trainer.train(train_loader, val_loader)
    
    print("🎉 Training completed! You can now chat with the bot.")

def chat_mode():
    """Start interactive chat mode"""
    chatbot = HinglishChatbot()
    if chatbot.model:
        chatbot.interactive_chat()

def test_single_input():
    """Test single input mode"""
    chatbot = HinglishChatbot()
    if chatbot.model:
        user_input = input("Please Enter ur Query: ")
        response = chatbot.chat(user_input)
        print(f"Response: {response}")

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    import sys
    
    # Set seed
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    print("🚀 HINGLISH CHATBOT SYSTEM")
    print("="*40)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--train':
            train_model()
        elif sys.argv[1] == '--chat':
            chat_mode()
        elif sys.argv[1] == '--test':
            test_single_input()
        else:
            print("Usage:")
            print("  python chatbot.py --train    # Train the model")
            print("  python chatbot.py --chat     # Interactive chat")
            print("  python chatbot.py --test     # Single input test")
    else:
        # Default behavior - check if model exists
        if os.path.exists("best_hinglish_chatbot.pth"):
            print("Model found! Starting interactive chat...")
            chat_mode()
        else:
            print("No trained model found.")
            print("\nFirst, create 'hinglish.txt' with your Hinglish sentences.")
            print("Then run: python chatbot.py --train")
            print("After training: python chatbot.py --chat")

if __name__ == "__main__":
    main()
