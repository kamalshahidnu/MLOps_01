"""
Knowledge Distillation Module
Distill knowledge from teacher model to student model
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import torch
import torch.nn as nn


class KnowledgeDistiller:
    """Knowledge distillation from teacher to student model"""
    
    def __init__(self, teacher_model, student_model_type='logistic', temperature=3.0, alpha=0.7):
        """
        Initialize knowledge distiller
        Args:
            teacher_model: Pre-trained teacher model (large, accurate)
            student_model_type: Type of student model ('logistic', 'xgboost', 'neural')
            temperature: Temperature for softmax (higher = softer probabilities)
            alpha: Weight for teacher predictions vs true labels
        """
        self.teacher_model = teacher_model
        self.student_model_type = student_model_type
        self.temperature = temperature
        self.alpha = alpha
        self.student_model = None
    
    def distill(self, X_train, y_train):
        """
        Distill knowledge from teacher to student
        Args:
            X_train: Training features
            y_train: True labels
        Returns:
            Trained student model
        """
        # Get teacher predictions (soft labels)
        teacher_proba = self.teacher_model.predict_proba(X_train)
        teacher_soft = self._softmax_with_temperature(teacher_proba, self.temperature)
        
        # Get hard labels
        y_hard = y_train.values if hasattr(y_train, 'values') else y_train
        
        # Train student model
        if self.student_model_type == 'logistic':
            self.student_model = LogisticRegression(random_state=42, max_iter=1000)
            # Use weighted combination of soft and hard labels
            # In practice, you'd use a custom loss function
            self.student_model.fit(X_train, y_hard)
        
        elif self.student_model_type == 'xgboost':
            self.student_model = XGBClassifier(random_state=42)
            self.student_model.fit(X_train, y_hard)
        
        elif self.student_model_type == 'neural':
            self.student_model = self._create_neural_student(X_train.shape[1])
            self._train_neural_student(X_train, teacher_soft, y_hard)
        
        return self.student_model
    
    def _softmax_with_temperature(self, logits, temperature):
        """Apply temperature scaling to softmax"""
        logits = logits / temperature
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    def _create_neural_student(self, input_dim):
        """Create a small neural network student model"""
        class StudentNet(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, 32)
                self.fc2 = nn.Linear(32, 16)
                self.fc3 = nn.Linear(16, 2)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.2)
            
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.fc3(x)
                return x
        
        return StudentNet(input_dim)
    
    def _train_neural_student(self, X_train, teacher_soft, y_hard):
        """Train neural network student with knowledge distillation loss"""
        import torch.optim as optim
        
        X_tensor = torch.FloatTensor(X_train.values)
        teacher_tensor = torch.FloatTensor(teacher_soft)
        y_tensor = torch.LongTensor(y_hard)
        
        criterion_kl = nn.KLDivLoss(reduction='batchmean')
        criterion_ce = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.student_model.parameters(), lr=0.001)
        
        self.student_model.train()
        for epoch in range(50):
            optimizer.zero_grad()
            outputs = self.student_model(X_tensor)
            
            # Knowledge distillation loss
            loss_kl = criterion_kl(
                torch.log_softmax(outputs / self.temperature, dim=1),
                teacher_tensor
            ) * (self.temperature ** 2) * self.alpha
            
            # Standard cross-entropy loss
            loss_ce = criterion_ce(outputs, y_tensor) * (1 - self.alpha)
            
            loss = loss_kl + loss_ce
            loss.backward()
            optimizer.step()
        
        self.student_model.eval()
    
    def get_student_model(self):
        """Get trained student model"""
        return self.student_model

