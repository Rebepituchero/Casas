import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import time

class RedNeuronal(nn.Module):
    def __init__(self, entrada_dim: int, salida_dim: int):
        super(RedNeuronal, self).__init__()
        self.fc1 = nn.Linear(entrada_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, salida_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
   
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class GestorEntrenamiento:
    def __init__(self, configuracion: Dict[str, Any]):
        self.config = configuracion
        self.modelo = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.metricas_por_epoca = []
        self.tiempo_por_epoca = []
        self.X_val = None
        self.y_val = None
        self.matriz_confusion = None
        self.curva_roc = None
        self.importancia_features = None
        self.distribucion_errores = None
        self.predicciones_vs_reales = None
   
    def preparar_datos(self, df: pd.DataFrame):
        try:
            X = df[self.config['columnas_entrada']].copy()
            y = df[self.config['columna_objetivo']].copy()
            
            for col in X.columns:
                if X[col].dtype == 'object':
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
            
            X = X.values
            y = y.values
            
            X = self.scaler.fit_transform(X)
            
            if self.config['tipo_modelo'] == 'clasificacion' or self.config['tipo_modelo'] == 'red_neuronal':
                if y.dtype == 'object' or not np.issubdtype(y.dtype, np.number):
                    y = self.label_encoder.fit_transform(y.astype(str))
            
            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size=self.config['validacion_split'],
                random_state=42,
                stratify=y if self.config['tipo_modelo'] == 'clasificacion' else None
            )
            
            self.X_val = X_val
            self.y_val = y_val
            
            return X_train, X_val, y_train, y_val
        except Exception as e:
            raise Exception(f"Error en preparar_datos: {str(e)}")
   
    def entrenar_sklearn(self, X_train, X_val, y_train, y_val):
        try:
            if self.config['tipo_modelo'] == 'clasificacion':
                self.modelo = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            else:
                self.modelo = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            
            inicio = time.time()
            self.modelo.fit(X_train, y_train)
            tiempo_entrenamiento = time.time() - inicio
            
            pred_train = self.modelo.predict(X_train)
            pred_val = self.modelo.predict(X_val)
            
            if self.config['tipo_modelo'] == 'clasificacion':
                metricas = {
                    'precision_entrenamiento': float(accuracy_score(y_train, pred_train)),
                    'precision_validacion': float(accuracy_score(y_val, pred_val)),
                    'precision': float(accuracy_score(y_val, pred_val)),
                    'recall': float(recall_score(y_val, pred_val, average='weighted', zero_division=0)),
                    'f1_score': float(f1_score(y_val, pred_val, average='weighted', zero_division=0))
                }
                
                self.matriz_confusion = confusion_matrix(y_val, pred_val).tolist()
                
                if len(np.unique(y_val)) == 2:
                    try:
                        pred_proba = self.modelo.predict_proba(X_val)[:, 1]
                        fpr, tpr, _ = roc_curve(y_val, pred_proba)
                        roc_auc = auc(fpr, tpr)
                        self.curva_roc = {
                            'fpr': fpr.tolist(),
                            'tpr': tpr.tolist(),
                            'auc': float(roc_auc)
                        }
                    except:
                        pass
            else:
                metricas = {
                    'mse_entrenamiento': float(mean_squared_error(y_train, pred_train)),
                    'mse_validacion': float(mean_squared_error(y_val, pred_val)),
                    'mse': float(mean_squared_error(y_val, pred_val)),
                    'r2_score': float(r2_score(y_val, pred_val))
                }
                
                errores = y_val - pred_val
                self.distribucion_errores = errores.tolist()
                self.predicciones_vs_reales = [
                    {'real': float(r), 'prediccion': float(p)}
                    for r, p in zip(y_val, pred_val)
                ]
            
            if hasattr(self.modelo, 'feature_importances_'):
                self.importancia_features = [
                    {'feature': col, 'importancia': float(imp)}
                    for col, imp in zip(self.config['columnas_entrada'], self.modelo.feature_importances_)
                ]
                self.importancia_features.sort(key=lambda x: x['importancia'], reverse=True)
            
            metricas['tiempo_total'] = tiempo_entrenamiento
            metricas['perdida_final'] = metricas.get('mse', metricas.get('precision', 0))
            self.tiempo_por_epoca = [tiempo_entrenamiento]
            
            metrica_epoca = {
                'epoca': 1,
                'perdida_entrenamiento': metricas.get('mse_entrenamiento', 0),
                'perdida_validacion': metricas.get('mse_validacion', 0),
                'tiempo': tiempo_entrenamiento
            }
            
            if self.config['tipo_modelo'] == 'clasificacion':
                metrica_epoca['precision_entrenamiento'] = metricas['precision_entrenamiento']
                metrica_epoca['precision_validacion'] = metricas['precision_validacion']
            
            self.metricas_por_epoca = [metrica_epoca]
            
            return metricas
            
        except Exception as e:
            raise Exception(f"Error en entrenar_sklearn: {str(e)}")
   
    def entrenar_red_neuronal(self, X_train, X_val, y_train, y_val):
        try:
            entrada_dim = X_train.shape[1]
            if self.config['tipo_modelo'] == 'clasificacion' or len(np.unique(y_train)) < 20:
                salida_dim = len(np.unique(y_train))
                criterio = nn.CrossEntropyLoss()
                es_clasificacion = True
            else:
                salida_dim = 1
                criterio = nn.MSELoss()
                es_clasificacion = False
            
            self.modelo = RedNeuronal(entrada_dim, salida_dim)
            optimizador = optim.Adam(self.modelo.parameters(), lr=self.config['tasa_aprendizaje'])
            
            X_train_t = torch.FloatTensor(X_train)
            X_val_t = torch.FloatTensor(X_val)
            
            if es_clasificacion:
                y_train_t = torch.LongTensor(y_train.astype(int))
                y_val_t = torch.LongTensor(y_val.astype(int))
            else:
                y_train_t = torch.FloatTensor(y_train).reshape(-1, 1)
                y_val_t = torch.FloatTensor(y_val).reshape(-1, 1)
            
            tamano_lote = self.config['tamano_lote']
            n_muestras = len(X_train_t)
            
            for epoca in range(self.config['epocas']):
                tiempo_inicio = time.time()
                self.modelo.train()
                perdida_total = 0
                
                indices = torch.randperm(n_muestras)
                for i in range(0, n_muestras, tamano_lote):
                    batch_indices = indices[i:i+tamano_lote]
                    X_batch = X_train_t[batch_indices]
                    y_batch = y_train_t[batch_indices]
                    
                    optimizador.zero_grad()
                    salidas = self.modelo(X_batch)
                    perdida = criterio(salidas, y_batch)
                    perdida.backward()
                    optimizador.step()
                    
                    perdida_total += perdida.item()
                
                self.modelo.eval()
                with torch.no_grad():
                    salidas_train = self.modelo(X_train_t)
                    salidas_val = self.modelo(X_val_t)
                    
                    perdida_train = criterio(salidas_train, y_train_t).item()
                    perdida_val = criterio(salidas_val, y_val_t).item()
                
                tiempo_epoca = time.time() - tiempo_inicio
                self.tiempo_por_epoca.append(tiempo_epoca)
                
                metrica_epoca = {
                    'epoca': epoca + 1,
                    'perdida_entrenamiento': perdida_train,
                    'perdida_validacion': perdida_val,
                    'tiempo': tiempo_epoca
                }
                
                if es_clasificacion:
                    _, pred_train = torch.max(salidas_train, 1)
                    _, pred_val = torch.max(salidas_val, 1)
                    metrica_epoca['precision_entrenamiento'] = float((pred_train == y_train_t).sum().item() / len(y_train_t))
                    metrica_epoca['precision_validacion'] = float((pred_val == y_val_t).sum().item() / len(y_val_t))
                
                self.metricas_por_epoca.append(metrica_epoca)
            
            self.modelo.eval()
            with torch.no_grad():
                salidas_val = self.modelo(X_val_t)
                
                if es_clasificacion:
                    _, pred_val = torch.max(salidas_val, 1)
                    pred_val_np = pred_val.numpy()
                    y_val_np = y_val_t.numpy()
                    
                    metricas_finales = {
                        'precision': self.metricas_por_epoca[-1]['precision_validacion'],
                        'perdida_final': self.metricas_por_epoca[-1]['perdida_validacion'],
                        'recall': float(recall_score(y_val_np, pred_val_np, average='weighted', zero_division=0)),
                        'f1_score': float(f1_score(y_val_np, pred_val_np, average='weighted', zero_division=0))
                    }
                    
                    self.matriz_confusion = confusion_matrix(y_val_np, pred_val_np).tolist()
                    
                    if len(np.unique(y_val_np)) == 2:
                        try:
                            pred_proba = torch.softmax(salidas_val, dim=1)[:, 1].numpy()
                            fpr, tpr, _ = roc_curve(y_val_np, pred_proba)
                            roc_auc = auc(fpr, tpr)
                            self.curva_roc = {
                                'fpr': fpr.tolist(),
                                'tpr': tpr.tolist(),
                                'auc': float(roc_auc)
                            }
                        except:
                            pass
                else:
                    pred_val_np = salidas_val.numpy().flatten()
                    y_val_np = y_val_t.numpy().flatten()
                    
                    metricas_finales = {
                        'mse': self.metricas_por_epoca[-1]['perdida_validacion'],
                        'r2_score': float(r2_score(y_val_np, pred_val_np)),
                        'perdida_final': self.metricas_por_epoca[-1]['perdida_validacion']
                    }
                    
                    errores = y_val_np - pred_val_np
                    self.distribucion_errores = errores.tolist()
                    self.predicciones_vs_reales = [
                        {'real': float(r), 'prediccion': float(p)}
                        for r, p in zip(y_val_np, pred_val_np)
                    ]
            
            modelo_dict = {
                'state_dict': self.modelo.state_dict(),
                'output_dim': salida_dim,
                'es_clasificacion': es_clasificacion
            }
            self.modelo = modelo_dict
            
            return metricas_finales
            
        except Exception as e:
            raise Exception(f"Error en entrenar_red_neuronal: {str(e)}")
   
    def predecir(self, X):
        try:
            X_scaled = self.scaler.transform(X)
            
            if isinstance(self.modelo, (RandomForestClassifier, RandomForestRegressor)):
                prediccion = self.modelo.predict(X_scaled)
            else:
                modelo_restaurado = RedNeuronal(X_scaled.shape[1], self.modelo['output_dim'])
                modelo_restaurado.load_state_dict(self.modelo['state_dict'])
                modelo_restaurado.eval()
                
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X_scaled)
                    salida = modelo_restaurado(X_tensor)
                    
                    if salida.shape[1] > 1:
                        _, prediccion = torch.max(salida, 1)
                        prediccion = prediccion.numpy()
                    else:
                        prediccion = salida.numpy().flatten()
            
            return prediccion
            
        except Exception as e:
            raise Exception(f"Error en predecir: {str(e)}")
   
    def ejecutar(self, df: pd.DataFrame):
        try:
            X_train, X_val, y_train, y_val = self.preparar_datos(df)
            
            if self.config['tipo_modelo'] == 'red_neuronal':
                metricas = self.entrenar_red_neuronal(X_train, X_val, y_train, y_val)
            else:
                metricas = self.entrenar_sklearn(X_train, X_val, y_train, y_val)
            
            return metricas, self.metricas_por_epoca
        except Exception as e:
            raise Exception(f"Error en ejecutar: {str(e)}")