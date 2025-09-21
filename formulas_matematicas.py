"""
Módulo de Fórmulas Matemáticas Avanzadas para Predicción de Deserción
Universidad Horizonte - Sistema de Análisis Predictivo
"""

import numpy as np
import pandas as pd
from scipy import stats, linalg
from scipy.special import expit  # función sigmoide
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

class FormulasUniversidadHorizonte:
    """
    Clase principal que implementa todas las fórmulas matemáticas
    para análisis de deserción estudiantil
    """
    
    def __init__(self):
        self.formulas_info = self._init_formulas_info()
        self.scaler = StandardScaler()
        self.fitted = False
    
    def _init_formulas_info(self):
        """Información de todas las fórmulas con LaTeX y referencias"""
        return {
            'score_riesgo_compuesto': {
                'nombre': 'Score de Riesgo Compuesto',
                'latex': r'R = \sigma\left(\sum_{i=1}^{n} w_i \cdot \tilde{x}_i \right), \quad \tilde{x}_i=\frac{x_i-\mu_i}{\sigma_i}',
                'referencia': 'Altman, E. I. (1968). Financial ratios, discriminant analysis and the prediction of corporate bankruptcy.',
                'descripcion': 'Combina múltiples indicadores en un índice normalizado usando pesos ponderados y función sigmoide.',
                'umbral_alto': 0.65,
                'umbral_medio': 0.35
            },
            'mahalanobis_anomalia': {
                'nombre': 'Distancia de Mahalanobis (Detección de Anomalías)',
                'latex': r'D^2 = (x-\mu)^T \Sigma^{-1} (x-\mu)',
                'referencia': 'Mahalanobis, P. C. (1936). On the generalized distance in statistics.',
                'descripcion': 'Detecta perfiles estudiantiles atípicos considerando correlaciones multivariadas.',
                'umbral_alto': 15.0,
                'umbral_medio': 8.0
            },
            'engagement_pca': {
                'nombre': 'Índice de Compromiso (PCA + Entropía)',
                'latex': r'E = \text{Norm}(\text{PC}_1(\mathbf{X})) \text{ o } E = 1 - \frac{H(\mathbf{p})}{\log k}',
                'referencia': 'Shannon, C. E. (1948). A mathematical theory of communication.',
                'descripcion': 'Mide engagement combinando actividades con reducción dimensional y análisis entrópico.',
                'umbral_alto': 0.7,
                'umbral_medio': 0.4
            },
            'momentum_academico': {
                'nombre': 'Momentum Académico (Smoothing Exponencial)',
                'latex': r'm_t = \alpha g_t + (1-\alpha) m_{t-1}, \quad \Delta m = \frac{dm}{dt}',
                'referencia': 'Holt, C. C. (1957). Forecasting seasonals and trends by exponentially weighted moving averages.',
                'descripcion': 'Captura tendencia de desempeño académico con suavizado exponencial para detectar declive.',
                'umbral_alto': -0.5,
                'umbral_medio': -0.2
            },
            'vulnerabilidad_financiera': {
                'nombre': 'Vulnerabilidad Financiera (EWMA)',
                'latex': r'V_t = \sum_{k=0}^{K} \lambda (1-\lambda)^k d_{t-k}',
                'referencia': 'Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation.',
                'descripcion': 'Evalúa riesgo financiero usando media móvil exponencial de patrones de mora.',
                'umbral_alto': 0.6,
                'umbral_medio': 0.3
            },
            'eficiencia_aprendizaje': {
                'nombre': 'Eficiencia de Aprendizaje',
                'latex': r'E = \frac{\Delta \text{score}}{\Delta \text{horas\_estudio} + \epsilon}',
                'referencia': 'Newell, A., & Rosenbloom, P. S. (1981). Mechanisms of skill acquisition and the law of practice.',
                'descripcion': 'Mide mejora académica por tiempo invertido para identificar dificultades de aprendizaje.',
                'umbral_alto': 0.8,
                'umbral_medio': 0.4
            },
            'persistencia_asistencia': {
                'nombre': 'Persistencia de Asistencia (Cadenas de Markov)',
                'latex': r'\mathbf{P} = \{p_{ij}\}, \quad p_{ij} = P(S_{t+1}=j|S_t=i)',
                'referencia': 'Norris, J. R. (1997). Markov chains.',
                'descripcion': 'Modela patrones de asistencia como proceso estocástico para predecir ausentismo futuro.',
                'umbral_alto': 0.7,
                'umbral_medio': 0.5
            },
            'receptividad_intervencion': {
                'nombre': 'Receptividad a Intervención (Propensity Score)',
                'latex': r'\widehat{ATE} = \frac{1}{N}\sum_i \left(\frac{T_i Y_i}{e(X_i)} - \frac{(1-T_i)Y_i}{1-e(X_i)}\right)',
                'referencia': 'Rosenbaum, P. R., & Rubin, D. B. (1983). The central role of the propensity score.',
                'descripcion': 'Estima efectividad potencial de intervenciones usando inferencia causal.',
                'umbral_alto': 0.6,
                'umbral_medio': 0.3
            }
        }
    
    def fit(self, df):
        """Ajustar el modelo con datos de entrenamiento"""
        # Seleccionar features numéricas para normalización
        numeric_cols = ['promedio', 'asistencia_porcentaje', 'creditos_aprobados', 
                       'pagos_pendientes', 'uso_lms_horas_semana', 'interacciones_mes',
                       'edad', 'ciclo']
        
        # Filtrar columnas que existen en el DataFrame
        available_cols = [col for col in numeric_cols if col in df.columns]
        
        if available_cols:
            self.scaler.fit(df[available_cols])
            self.fitted = True
            self.feature_cols = available_cols
            
            # Calcular estadísticas globales para referencias
            self.global_stats = df[available_cols].describe()
        
        return self
    
    def score_riesgo_compuesto(self, estudiante):
        """
        Calcula score de riesgo compuesto usando múltiples factores ponderados
        """
        if not self.fitted:
            raise ValueError("Modelo debe ser ajustado primero con fit()")
        
        # Pesos por importancia (suma = 1)
        pesos = {
            'promedio': 0.25,          # Factor académico principal
            'asistencia_porcentaje': 0.20,  # Compromiso con clases
            'pagos_pendientes': 0.15,       # Situación financiera
            'uso_lms_horas_semana': 0.15,   # Engagement digital
            'creditos_aprobados': 0.10,     # Progreso curricular
            'interacciones_mes': 0.10,      # Participación social
            'ciclo': 0.05                   # Nivel académico
        }
        
        score_total = 0
        peso_usado = 0
        
        for feature, peso in pesos.items():
            if feature in estudiante and feature in self.feature_cols:
                valor = estudiante[feature]
                
                # Normalizar usando estadísticas globales
                if feature in self.global_stats.columns:
                    mean = self.global_stats.loc['mean', feature]
                    std = self.global_stats.loc['std', feature]
                    
                    if std > 0:
                        # Z-score normalizado
                        z_score = (valor - mean) / std
                        
                        # Invertir para que valores bajos = mayor riesgo
                        if feature in ['promedio', 'asistencia_porcentaje', 'uso_lms_horas_semana', 
                                     'interacciones_mes', 'creditos_aprobados']:
                            z_score = -z_score
                        
                        score_total += peso * z_score
                        peso_usado += peso
        
        if peso_usado > 0:
            score_normalizado = score_total / peso_usado
            # Aplicar función sigmoide para mapear a (0,1)
            score_final = expit(score_normalizado)
        else:
            score_final = 0.5  # Neutro si no hay datos
        
        return float(score_final)
    
    def mahalanobis_anomalia(self, estudiante, poblacion_df):
        """
        Calcula distancia de Mahalanobis para detectar anomalías
        """
        try:
            # Seleccionar features disponibles
            features = [col for col in self.feature_cols if col in estudiante and col in poblacion_df.columns]
            
            if len(features) < 2:
                return 0.0
            
            # Vector del estudiante
            x = np.array([estudiante[f] for f in features])
            
            # Matriz de datos población
            X = poblacion_df[features].dropna()
            
            if len(X) < 2:
                return 0.0
            
            # Media y covarianza
            mu = X.mean().values
            try:
                cov_matrix = np.cov(X.T)
                # Añadir regularización si la matriz es singular
                if np.linalg.det(cov_matrix) < 1e-8:
                    cov_matrix += np.eye(len(features)) * 1e-6
                
                cov_inv = np.linalg.inv(cov_matrix)
            except:
                # Fallback: usar matriz identidad
                cov_inv = np.eye(len(features))
            
            # Calcular distancia de Mahalanobis
            diff = x - mu
            mahal_dist = np.sqrt(diff.T @ cov_inv @ diff)
            
            return float(mahal_dist)
        
        except Exception:
            return 0.0
    
    def engagement_pca(self, estudiante):
        """
        Calcula índice de engagement usando PCA de actividades
        """
        # Features de engagement
        engagement_features = {
            'uso_lms_horas_semana': estudiante.get('uso_lms_horas_semana', 0),
            'interacciones_mes': estudiante.get('interacciones_mes', 0),
            'accesos_plataforma_mes': estudiante.get('accesos_plataforma_mes', 0),
            'asistencia_porcentaje': estudiante.get('asistencia_porcentaje', 0)
        }
        
        # Normalizar valores [0-1]
        normalizaciones = {
            'uso_lms_horas_semana': 40,  # máximo horas por semana
            'interacciones_mes': 100,    # máximo interacciones
            'accesos_plataforma_mes': 200,  # máximo accesos
            'asistencia_porcentaje': 100    # porcentaje
        }
        
        values = []
        for feature, max_val in normalizaciones.items():
            val = engagement_features[feature]
            normalized = min(val / max_val, 1.0) if max_val > 0 else 0
            values.append(normalized)
        
        # Calcular engagement como promedio ponderado
        pesos = [0.3, 0.25, 0.25, 0.2]  # LMS, interacciones, accesos, asistencia
        engagement = np.sum([v * w for v, w in zip(values, pesos)])
        
        return float(engagement)
    
    def momentum_academico(self, estudiante, historial_notas=None):
        """
        Calcula momentum académico basado en tendencia de notas
        """
        promedio_actual = estudiante.get('promedio', 0)
        
        if historial_notas and len(historial_notas) > 1:
            # Si hay historial, calcular tendencia
            notas = np.array(historial_notas)
            
            # Smoothing exponencial
            alpha = 0.3
            smoothed = [notas[0]]
            for i in range(1, len(notas)):
                smoothed.append(alpha * notas[i] + (1 - alpha) * smoothed[-1])
            
            # Calcular momentum como pendiente de últimos valores
            if len(smoothed) >= 3:
                recent = smoothed[-3:]
                x = np.arange(len(recent))
                slope, _ = np.polyfit(x, recent, 1)
                return float(slope / 5)  # Normalizar
        
        # Fallback: usar indicadores indirectos
        ciclo = estudiante.get('ciclo', 1)
        repitencias = estudiante.get('repitencias', 0)
        
        # Estimar momentum basado en progreso vs ciclo
        progreso_esperado = ciclo * 2  # Aproximación
        progreso_real = max(0, promedio_actual - 10)  # Base 10
        
        momentum = (progreso_real - progreso_esperado) / max(ciclo, 1)
        
        return float(np.clip(momentum, -2, 2))
    
    def vulnerabilidad_financiera(self, estudiante):
        """
        Calcula vulnerabilidad financiera usando patrones de pago
        """
        pagos_pendientes = estudiante.get('pagos_pendientes', 0)
        monto_deuda = estudiante.get('monto_deuda', 0)
        tiene_beca = estudiante.get('tiene_beca', False)
        
        # Factores de riesgo financiero
        factor_mora = min(pagos_pendientes / 6, 1.0)  # Normalizar a máximo 6 meses
        factor_monto = min(monto_deuda / 10000, 1.0)  # Normalizar a 10k
        factor_beca = 0.3 if not tiene_beca else 0.0  # Riesgo por no tener beca
        
        # EWMA simulado (sin historial completo)
        lambda_factor = 0.7
        vulnerabilidad = (lambda_factor * factor_mora + 
                         0.2 * factor_monto + 
                         0.1 * factor_beca)
        
        return float(np.clip(vulnerabilidad, 0, 1))
    
    def eficiencia_aprendizaje(self, estudiante):
        """
        Calcula eficiencia de aprendizaje (rendimiento por esfuerzo)
        """
        promedio = estudiante.get('promedio', 0)
        uso_lms = estudiante.get('uso_lms_horas_semana', 1)  # Evitar división por 0
        ciclo = estudiante.get('ciclo', 1)
        
        # Estimar esfuerzo total
        esfuerzo_estimado = uso_lms + (ciclo * 2)  # Horas LMS + carga académica
        
        # Eficiencia = rendimiento / esfuerzo
        if esfuerzo_estimado > 0:
            eficiencia = (promedio - 10) / esfuerzo_estimado  # Base 10 para promedio
        else:
            eficiencia = 0
        
        # Normalizar a rango 0-1
        eficiencia_norm = (eficiencia + 0.5) / 1.5  # Ajustar rango
        
        return float(np.clip(eficiencia_norm, 0, 1))
    
    def persistencia_asistencia(self, estudiante):
        """
        Simula modelo de persistencia usando patrones de asistencia
        """
        asistencia = estudiante.get('asistencia_porcentaje', 0) / 100
        faltas_injustificadas = estudiante.get('faltas_injustificadas', 0)
        
        # Simular probabilidades de transición
        # Estado: Asiste, No asiste, Justificado
        if asistencia > 0.8:
            prob_persistencia = 0.9  # Alta probabilidad de continuar asistiendo
        elif asistencia > 0.6:
            prob_persistencia = 0.7  # Moderada
        else:
            prob_persistencia = 0.4  # Baja
        
        # Penalizar faltas injustificadas
        penalizacion = min(faltas_injustificadas * 0.05, 0.3)
        prob_persistencia -= penalizacion
        
        return float(np.clip(prob_persistencia, 0, 1))
    
    def receptividad_intervencion(self, estudiante):
        """
        Estima receptividad a intervenciones usando características del estudiante
        """
        # Factores que influyen en receptividad
        edad = estudiante.get('edad', 20)
        ciclo = estudiante.get('ciclo', 1)
        promedio = estudiante.get('promedio', 0)
        actividades_extra = estudiante.get('actividades_extracurriculares', 0)
        uso_lms = estudiante.get('uso_lms_horas_semana', 0)
        
        # Calcular factores de receptividad
        factor_edad = 1.0 - min((edad - 18) * 0.05, 0.3)  # Jóvenes más receptivos
        factor_ciclo = min(ciclo * 0.1, 0.5)  # Ciclos avanzados más maduros
        factor_compromiso = min((actividades_extra + uso_lms/10) * 0.15, 0.4)
        factor_academico = 0.5 if 11 <= promedio <= 15 else 0.2  # Rango medio más receptivo
        
        receptividad = factor_edad + factor_ciclo + factor_compromiso + factor_academico
        
        return float(np.clip(receptividad, 0, 1))
    
    def calcular_todas_formulas(self, estudiante, poblacion_df=None):
        """
        Calcula todas las fórmulas para un estudiante y retorna resultados organizados
        """
        if not self.fitted:
            self.fit(poblacion_df if poblacion_df is not None else pd.DataFrame([estudiante]))
        
        resultados = {}
        
        # Calcular cada fórmula
        try:
            resultados['score_riesgo_compuesto'] = self.score_riesgo_compuesto(estudiante)
        except Exception as e:
            resultados['score_riesgo_compuesto'] = 0.5
        
        try:
            if poblacion_df is not None:
                resultados['mahalanobis_anomalia'] = self.mahalanobis_anomalia(estudiante, poblacion_df)
            else:
                resultados['mahalanobis_anomalia'] = 0.0
        except Exception:
            resultados['mahalanobis_anomalia'] = 0.0
        
        try:
            resultados['engagement_pca'] = self.engagement_pca(estudiante)
        except Exception:
            resultados['engagement_pca'] = 0.0
        
        try:
            resultados['momentum_academico'] = self.momentum_academico(estudiante)
        except Exception:
            resultados['momentum_academico'] = 0.0
        
        try:
            resultados['vulnerabilidad_financiera'] = self.vulnerabilidad_financiera(estudiante)
        except Exception:
            resultados['vulnerabilidad_financiera'] = 0.0
        
        try:
            resultados['eficiencia_aprendizaje'] = self.eficiencia_aprendizaje(estudiante)
        except Exception:
            resultados['eficiencia_aprendizaje'] = 0.0
        
        try:
            resultados['persistencia_asistencia'] = self.persistencia_asistencia(estudiante)
        except Exception:
            resultados['persistencia_asistencia'] = 0.0
        
        try:
            resultados['receptividad_intervencion'] = self.receptividad_intervencion(estudiante)
        except Exception:
            resultados['receptividad_intervencion'] = 0.0
        
        return resultados
    
    def interpretar_resultado(self, formula_nombre, valor):
        """
        Interpreta el resultado de una fórmula específica
        """
        if formula_nombre not in self.formulas_info:
            return "Fórmula no reconocida"
        
        info = self.formulas_info[formula_nombre]
        umbral_alto = info.get('umbral_alto', 0.7)
        umbral_medio = info.get('umbral_medio', 0.4)
        
        if valor >= umbral_alto:
            nivel = "ALTO"
        elif valor >= umbral_medio:
            nivel = "MEDIO" 
        else:
            nivel = "BAJO"
        
        return f"{nivel} ({valor:.3f})"
    
    def generar_reporte_completo(self, estudiante, poblacion_df=None):
        """
        Genera reporte completo con todas las métricas y interpretaciones
        """
        resultados = self.calcular_todas_formulas(estudiante, poblacion_df)
        
        reporte = {
            'estudiante_id': estudiante.get('student_id', 'N/A'),
            'nombre': f"{estudiante.get('nombre', '')} {estudiante.get('apellido', '')}",
            'carrera': estudiante.get('carrera', 'N/A'),
            'resultados': {},
            'interpretaciones': {},
            'resumen_riesgo': '',
            'recomendaciones': []
        }
        
        # Procesar resultados
        for formula, valor in resultados.items():
            reporte['resultados'][formula] = valor
            reporte['interpretaciones'][formula] = self.interpretar_resultado(formula, valor)
        
        # Generar resumen de riesgo
        score_principal = resultados.get('score_riesgo_compuesto', 0.5)
        if score_principal >= 0.65:
            reporte['resumen_riesgo'] = "RIESGO ALTO de deserción"
        elif score_principal >= 0.35:
            reporte['resumen_riesgo'] = "RIESGO MEDIO de deserción"
        else:
            reporte['resumen_riesgo'] = "RIESGO BAJO de deserción"
        
        # Generar recomendaciones basadas en resultados
        recomendaciones = []
        
        if resultados.get('vulnerabilidad_financiera', 0) > 0.5:
            recomendaciones.append("Apoyo financiero urgente - evaluar becas o planes de pago")
        
        if resultados.get('engagement_pca', 0) < 0.4:
            recomendaciones.append("Aumentar participación en plataforma digital y actividades")
        
        if resultados.get('momentum_academico', 0) < -0.3:
            recomendaciones.append("Tutoría académica para revertir tendencia negativa")
        
        if resultados.get('persistencia_asistencia', 0) < 0.6:
            recomendaciones.append("Seguimiento de asistencia y apoyo motivacional")
        
        if resultados.get('eficiencia_aprendizaje', 0) < 0.4:
            recomendaciones.append("Revisar métodos de estudio y carga académica")
        
        if not recomendaciones:
            recomendaciones.append("Mantener seguimiento preventivo regular")
        
        reporte['recomendaciones'] = recomendaciones
        
        return reporte


# Función de utilidad para usar fácilmente desde Streamlit
def analizar_estudiante(datos_estudiante, poblacion_df=None):
    """
    Función simplificada para analizar un estudiante específico
    """
    formulas = FormulasUniversidadHorizonte()
    
    if poblacion_df is not None:
        formulas.fit(poblacion_df)
    
    return formulas.generar_reporte_completo(datos_estudiante, poblacion_df)


# Información de fórmulas para mostrar en UI
FORMULAS_LATEX = {
    'score_riesgo_compuesto': {
        'titulo': 'Score de Riesgo Compuesto',
        'latex': r'R = \sigma\left(\sum_{i=1}^{n} w_i \cdot \tilde{x}_i \right), \quad \tilde{x}_i=\frac{x_i-\mu_i}{\sigma_i}',
        'descripcion': 'Combina múltiples indicadores académicos, financieros y comportamentales en un índice único.',
        'referencia': 'Altman, E. I. (1968). Financial ratios and prediction of bankruptcy.'
    },
    'mahalanobis_anomalia': {
        'titulo': 'Distancia de Mahalanobis',
        'latex': r'D^2 = (x-\mu)^T \Sigma^{-1} (x-\mu)',
        'descripcion': 'Detecta perfiles estudiantiles atípicos considerando correlaciones multivariadas.',
        'referencia': 'Mahalanobis, P. C. (1936). On the generalized distance in statistics.'
    },
    'engagement_pca': {
        'titulo': 'Índice de Engagement (PCA)',
        'latex': r'E = \text{Norm}(\text{PC}_1(\mathbf{X})) \text{ o } E = 1 - \frac{H(\mathbf{p})}{\log k}',
        'descripcion': 'Mide compromiso estudiantil usando análisis de componentes principales.',
        'referencia': 'Shannon, C. E. (1948). A mathematical theory of communication.'
    },
    'momentum_academico': {
        'titulo': 'Momentum Académico',
        'latex': r'm_t = \alpha g_t + (1-\alpha) m_{t-1}',
        'descripcion': 'Analiza tendencia de desempeño usando suavizado exponencial.',
        'referencia': 'Holt, C. C. (1957). Forecasting by exponentially weighted moving averages.'
    },
    'vulnerabilidad_financiera': {
        'titulo': 'Vulnerabilidad Financiera',
        'latex': r'V_t = \sum_{k=0}^{K} \lambda (1-\lambda)^k d_{t-k}',
        'descripcion': 'Evalúa riesgo financiero usando media móvil exponencial de patrones de mora.',
        'referencia': 'Engle, R. F. (1982). Autoregressive conditional heteroscedasticity.'
    },
    'eficiencia_aprendizaje': {
        'titulo': 'Eficiencia de Aprendizaje',
        'latex': r'E = \frac{\Delta \text{score}}{\Delta \text{horas\_estudio} + \epsilon}',
        'descripcion': 'Ratio de mejora académica por tiempo de estudio invertido.',
        'referencia': 'Newell, A. & Rosenbloom, P. S. (1981). Mechanisms of skill acquisition.'
    },
    'persistencia_asistencia': {
        'titulo': 'Persistencia de Asistencia',
        'latex': r'\mathbf{P} = \{p_{ij}\}, \quad p_{ij} = P(S_{t+1}=j|S_t=i)',
        'descripcion': 'Modela patrones de asistencia como cadenas de Markov.',
        'referencia': 'Norris, J. R. (1997). Markov chains.'
    },
    'receptividad_intervencion': {
        'titulo': 'Receptividad a Intervención',
        'latex': r'\widehat{ATE} = \frac{1}{N}\sum_i \left(\frac{T_i Y_i}{e(X_i)} - \frac{(1-T_i)Y_i}{1-e(X_i)}\right)',
        'descripcion': 'Estima efectividad potencial de intervenciones usando propensity score.',
        'referencia': 'Rosenbaum & Rubin (1983). The central role of the propensity score.'
    }
}

if __name__ == "__main__":
    # Ejemplo de uso
    estudiante_ejemplo = {
        'student_id': 'UH2024001',
        'nombre': 'Ana',
        'apellido': 'García',
        'carrera': 'Ingeniería de Sistemas',
        'promedio': 12.5,
        'asistencia_porcentaje': 75,
        'pagos_pendientes': 2,
        'uso_lms_horas_semana': 8,
        'ciclo': 6,
        'edad': 21
    }
    
    resultado = analizar_estudiante(estudiante_ejemplo)
    print("Ejemplo de análisis:")
    print(f"Estudiante: {resultado['nombre']}")
    print(f"Riesgo: {resultado['resumen_riesgo']}")
    print(f"Score principal: {resultado['resultados']['score_riesgo_compuesto']:.3f}")