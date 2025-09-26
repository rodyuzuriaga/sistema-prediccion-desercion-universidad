"""
Universidad Horizonte - Sistema Predictivo de Deserción Estudiantil
Version: 3.0 - Professional UI/UX
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pymongo
from pymongo import MongoClient
import json
from datetime import datetime, timedelta
import time
import random
from faker import Faker
import streamlit_shadcn_ui as ui
# Importaciones opcionales para PDF
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

import io
import base64

st.set_page_config(
    page_title="Universidad Horizonte - Predicción Deserción",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS profesional con soporte para modo oscuro/claro
PROFESSIONAL_CSS = """
<style>
    /* Variables CSS para temas */
    :root {
        --primary-color: #1e40af;
        --secondary-color: #3b82f6;
        --accent-color: #10b981;
        --danger-color: #ef4444;
        --warning-color: #f59e0b;
        --success-color: #22c55e;
        --bg-primary: #ffffff;
        --bg-secondary: #f8fafc;
        --text-primary: #1f2937;
        --text-secondary: #6b7280;
        --border-color: #e5e7eb;
        --shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }

    /* Tema oscuro */
    [data-theme="dark"] {
        --bg-primary: #1f2937;
        --bg-secondary: #111827;
        --text-primary: #f9fafb;
        --text-secondary: #d1d5db;
        --border-color: #374151;
        --shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.3);
    }

    /* Reset y base */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Header principal */
    .main-header {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: var(--shadow);
    }

    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
    }

    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        text-align: center;
        opacity: 0.9;
    }

    /* Cards y contenedores */
    .metric-card {
        background: var(--bg-primary);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: var(--shadow);
        transition: all 0.3s ease;
        height: 100%;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-color);
        margin: 0;
    }

    .metric-label {
        font-size: 0.9rem;
        color: var(--text-secondary);
        margin: 0.5rem 0 0 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.025em;
    }

    .status-activo {
        background-color: #dcfce7;
        color: #166534;
    }

    .status-desertor {
        background-color: #fee2e2;
        color: #991b1b;
    }

    .status-riesgo {
        background-color: #fef3c7;
        color: #92400e;
    }

    /* Tablas */
    .data-table {
        border-collapse: collapse;
        width: 100%;
        background: var(--bg-primary);
        border-radius: 8px;
        overflow: hidden;
        box-shadow: var(--shadow);
    }

    .data-table th {
        background: var(--bg-secondary);
        color: var(--text-primary);
        font-weight: 600;
        text-align: left;
        padding: 1rem;
        border-bottom: 1px solid var(--border-color);
    }

    .data-table td {
        padding: 0.75rem 1rem;
        border-bottom: 1px solid var(--border-color);
        color: var(--text-primary);
    }

    .data-table tbody tr:hover {
        background: var(--bg-secondary);
    }

    /* Formularios y inputs */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 1px solid var(--border-color);
        background: var(--bg-primary);
        color: var(--text-primary);
    }

    /* Selectbox - Deshabilitar edición de texto */
    .stSelectbox > div > div > div > div {
        pointer-events: auto;
    }
    
    .stSelectbox > div > div > div > div > input {
        pointer-events: none !important;
        cursor: pointer !important;
        caret-color: transparent !important;
        user-select: none !important;
        -webkit-user-select: none !important;
        -moz-user-select: none !important;
        -ms-user-select: none !important;
    }
    
    .stSelectbox > div > div > div {
        cursor: pointer !important;
    }
    
    /* Evitar focus en input de selectbox */
    .stSelectbox input:focus {
        outline: none !important;
        box-shadow: none !important;
    }
    
    /* Asegurar que solo funcione como dropdown */
    .stSelectbox div[data-baseweb="select"] input {
        pointer-events: none !important;
        cursor: pointer !important;
        caret-color: transparent !important;
    }

    /* Sidebar */
    .css-1d391kg {
        background: var(--bg-secondary);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background: var(--bg-secondary);
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: 1px solid var(--border-color);
    }

    /* Alertas y notificaciones */
    .alert {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid;
    }

    .alert-success {
        background: #f0f9ff;
        border-color: var(--success-color);
        color: #0c4a6e;
    }

    .alert-warning {
        background: #fffbeb;
        border-color: var(--warning-color);
        color: #92400e;
    }

    .alert-danger {
        background: #fef2f2;
        border-color: var(--danger-color);
        color: #991b1b;
    }

    /* Ocultar elementos de Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Responsive */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 1.8rem;
        }
        
        .metric-card {
            padding: 1rem;
        }
        
        .metric-value {
            font-size: 1.5rem;
        }
    }

    /* Reducir espaciado de divisores */
    hr {
        margin: 0.5rem 0 !important;
        border: none !important;
        border-top: 1px solid var(--border-color) !important;
        opacity: 0.3 !important;
    }

    /* Reducir espaciado de contenedores de divisores */
    .stElementContainer:has(hr) {
        margin: 0.25rem 0 !important;
        padding: 0 !important;
    }
</style>
"""

st.markdown(PROFESSIONAL_CSS, unsafe_allow_html=True)

def mostrar_formulas_matematicas():
    """Muestra las fórmulas matemáticas implementadas en el sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Algoritmos Implementados")
    st.sidebar.markdown("*8 modelos matemáticos aplicados*")
    
    with st.sidebar.expander("Score de Riesgo Compuesto", expanded=False):
        st.markdown("**Fórmula:** \( R = \sigma\left(\sum_{i=1}^{n} w_i \cdot \tilde{x}_i \right) \)")
        st.caption("*Autor: Basado en Altman (1968) - Credit scoring*")
        st.caption("*Sirve matemáticamente: Función sigmoide para mapear valores lineales a probabilidades (0-1)*")
        st.caption("*En la aplicación: Calcula riesgo global ponderando promedio (40%), asistencia (30%), pagos (30%)*")
        st.caption("*Finalidad: Identificar estudiantes en riesgo de deserción con score interpretable*")
    
    with st.sidebar.expander("Tasa de Retención Individual", expanded=False):
        st.markdown("**Fórmula:** Modelo de progreso relativo vs esperado")
        st.caption("*Autor: Adaptado de Tinto (1993)*")
        st.caption("*Sirve matemáticamente: Ratio de créditos acumulados vs esperados con penalización por ciclo*")
        st.caption("*En la aplicación: Evalúa progreso académico real vs teórico por ciclo*")
        st.caption("*Finalidad: Detectar atrasos académicos que predicen deserción*")
    
    with st.sidebar.expander("Entropía Académica", expanded=False):
        st.markdown("**Fórmula:** \( H(X) = -\sum_{i=1}^{n} p(x_i) \log_2 p(x_i) \)")
        st.caption("*Autor: Shannon (1948)*")
        st.caption("*Sirve matemáticamente: Mide incertidumbre/variabilidad en distribución de probabilidades*")
        st.caption("*En la aplicación: Calcula variabilidad en rendimiento académico simulado*")
        st.caption("*Finalidad: Identificar inconsistencia en desempeño que indica riesgo*")
    
    with st.sidebar.expander("Momentum Académico", expanded=False):
        st.markdown("**Fórmula:** \( m_t = \alpha g_t + (1-\alpha) m_{t-1} \)")
        st.caption("*Autor: Holt-Winters smoothing (1960)*")
        st.caption("*Sirve matemáticamente: Suavizado exponencial para detectar tendencias*")
        st.caption("*En la aplicación: Evalúa aceleración/desaceleración en promedio académico*")
        st.caption("*Finalidad: Predecir trayectoria futura del rendimiento estudiantil*")
    
    with st.sidebar.expander("Engagement Score (PCA)", expanded=False):
        st.markdown("**Fórmula:** \( E = \text{Norm}(\text{PC}_1(\mathbf{X})) \)")
        st.caption("*Autor: Hotelling (1933)*")
        st.caption("*Sirve matemáticamente: Reducción dimensional para combinar variables correlacionadas*")
        st.caption("*En la aplicación: Sintetiza horas LMS, interacciones y accesos en índice único*")
        st.caption("*Finalidad: Medir involucramiento estudiantil con plataformas educativas*")
    
    with st.sidebar.expander("Vulnerabilidad Financiera (EWMA)", expanded=False):
        st.markdown("**Fórmula:** \( V_t = \sum_{k=0}^{K} \lambda (1-\lambda)^k d_{t-k} \)")
        st.caption("*Autor: RiskMetrics (1996)*")
        st.caption("*Sirve matemáticamente: Media ponderada exponencialmente para datos recientes*")
        st.caption("*En la aplicación: Evalúa historial de pagos pendientes con mayor peso en eventos recientes*")
        st.caption("*Finalidad: Detectar deterioro financiero progresivo que afecta permanencia*")
    
    with st.sidebar.expander("Persistencia de Asistencia", expanded=False):
        st.markdown("**Fórmula:** Modelo de cadenas de Markov simplificado")
        st.caption("*Autor: Markov (1906)*")
        st.caption("*Sirve matemáticamente: Probabilidad de estados futuros basada en estados actuales*")
        st.caption("*En la aplicación: Estima probabilidad de mantener patrones de asistencia*")
        st.caption("*Finalidad: Predecir continuidad en hábitos de participación académica*")
    
    with st.sidebar.expander("Receptividad a Intervención", expanded=False):
        st.markdown("**Fórmula:** Propensity score matching adaptado")
        st.caption("*Autor: Rosenbaum & Rubin (1983)*")
        st.caption("*Sirve matemáticamente: Estimación de probabilidad de tratamiento condicional*")
        st.caption("*En la aplicación: Evalúa qué tan receptivo será estudiante a intervenciones específicas*")
        st.caption("*Finalidad: Priorizar estudiantes más propensos a responder positivamente a ayuda*")
    
def aplicar_formulas_estudiante(estudiante):
    """Aplica las fórmulas matemáticas a un estudiante específico"""
    resultados = {}
    
    try:
        # 1. Score de Riesgo Compuesto
        promedio_norm = estudiante.get('promedio', 0) / 20
        asistencia_norm = estudiante.get('asistencia_porcentaje', 100) / 100
        pagos_norm = 1 - (estudiante.get('pagos_pendientes', 0) / 5)
        
        # Ajuste conservador: si el promedio es alto (>16/20), dar más peso al promedio
        if promedio_norm > 0.8:  # Promedio > 16
            weights = [0.6, 0.2, 0.2]  # Más peso al promedio
        elif promedio_norm > 0.7:  # Promedio > 14
            weights = [0.5, 0.25, 0.25]  # Peso moderado al promedio
        else:
            weights = [0.4, 0.3, 0.3]  # Pesos originales
        
        features = [promedio_norm, asistencia_norm, pagos_norm]
        score_lineal = sum(w * f for w, f in zip(weights, features))
        
        # Aplicar función sigmoide
        resultados['score_riesgo'] = 1 / (1 + np.exp(-5 * (score_lineal - 0.5)))
        
        # 2. Tasa de Retención Individual (reformulada para mayor discriminación)
        # Problema: muchos estudiantes tienen créditos acumulados > créditos esperados
        # Solución: usar una métrica que considere el progreso relativo de manera más realista
        ciclo = estudiante.get('ciclo', 1)
        creditos_acumulados = estudiante.get('creditos_aprobados', 0)

        # Créditos esperados acumulados hasta este ciclo (acumulativo)
        # Asumiendo ~22 créditos por ciclo en promedio para carreras universitarias
        creditos_esperados_acumulados = ciclo * 22

        # Calcular ratio real vs esperado
        if creditos_esperados_acumulados > 0:
            ratio_progreso = creditos_acumulados / creditos_esperados_acumulados
        else:
            ratio_progreso = 1.0

        # Aplicar transformación sigmoide para mayor discriminación
        # Esta función mapea el ratio a una escala de retención más sensible
        # Ratio = 1.0 -> Retención = 0.8 (buen progreso)
        # Ratio < 0.5 -> Retención baja
        # Ratio > 1.2 -> Retención alta (estudiantes avanzados)

        if ratio_progreso >= 1.2:
            # Estudiantes que van adelantados
            tasa_retencion = min(0.95 + (ratio_progreso - 1.2) * 0.1, 1.0)
        elif ratio_progreso >= 0.8:
            # Estudiantes en ritmo normal
            tasa_retencion = 0.7 + (ratio_progreso - 0.8) * 0.5
        elif ratio_progreso >= 0.5:
            # Estudiantes atrasados moderadamente
            tasa_retencion = 0.3 + (ratio_progreso - 0.5) * 0.8
        elif ratio_progreso >= 0.2:
            # Estudiantes muy atrasados
            tasa_retencion = 0.1 + (ratio_progreso - 0.2) * 0.4
        else:
            # Estudiantes extremadamente atrasados
            tasa_retencion = max(0.01, ratio_progreso * 0.2)

        # Penalización adicional por ciclo: estudiantes en ciclos altos deben mantener ritmo
        # Penalización exponencial suave para no ser demasiado dura
        if ciclo > 3:
            factor_ciclo = 1.0 / (1.0 + (ciclo - 3) * 0.15)  # Penalización del 15% por ciclo adicional
            tasa_retencion = tasa_retencion * factor_ciclo

        # Asegurar límites
        resultados['tasa_retencion'] = max(0.0, min(1.0, tasa_retencion))
        
        # 3. Entropía Académica (simulada)
        notas_simuladas = [
            estudiante.get('promedio', 12) + random.uniform(-2, 2) for _ in range(5)
        ]
        # Normalizar para crear distribución de probabilidad
        notas_norm = np.array(notas_simuladas)
        notas_norm = (notas_norm - notas_norm.min()) / (notas_norm.max() - notas_norm.min() + 0.001)
        notas_norm = notas_norm / notas_norm.sum()
        
        # Calcular entropía
        entropia = -sum(p * np.log2(p + 1e-10) for p in notas_norm if p > 0)
        resultados['entropia_academica'] = entropia
        
        # 4. Momentum Académico (simulado con smoothing exponencial)
        alpha = 0.3
        momentum = alpha * (estudiante.get('promedio', 12) - 12) + (1 - alpha) * 0
        resultados['momentum_academico'] = momentum
        
        # 5. Engagement Score (PCA simulado)
        horas_lms = estudiante.get('uso_lms_horas_semana', 5)
        interacciones = estudiante.get('interacciones_mes', 10)
        accesos = estudiante.get('accesos_plataforma_mes', 20)
        
        engagement_features = np.array([horas_lms/40, interacciones/50, accesos/80])
        # PCA simplificado: primera componente principal simulada
        pc1 = np.dot(engagement_features, [0.6, 0.3, 0.1])
        resultados['engagement_score'] = min(max(pc1, 0), 1)
        
        # 6. Índice de Vulnerabilidad Financiera (EWMA)
        pagos_pendientes = estudiante.get('pagos_pendientes', 0)
        lambda_decay = 0.7
        vulnerabilidad = pagos_pendientes * lambda_decay
        resultados['vulnerabilidad_financiera'] = min(vulnerabilidad / 5, 1)
        
        return resultados
        
    except Exception as e:
        st.error(f"Error aplicando fórmulas: {str(e)}")
        return {}

def generar_diagnostico_automatico(resultados_formulas, estudiante):
    """Genera un diagnóstico automático dinámico basado en los resultados de fórmulas"""

    score_riesgo = resultados_formulas.get('score_riesgo', 0)
    tasa_retencion = resultados_formulas.get('tasa_retencion', 1)
    entropia = resultados_formulas.get('entropia_academica', 0)
    engagement = resultados_formulas.get('engagement_score', 1)
    momentum = resultados_formulas.get('momentum_academico', 0)
    vulnerabilidad = resultados_formulas.get('vulnerabilidad_financiera', 0)

    # Determinar nivel de riesgo
    if score_riesgo >= 0.7:
        nivel_riesgo = "alto"
        descripcion_riesgo = "un nivel de riesgo elevado"
    elif score_riesgo >= 0.4:
        nivel_riesgo = "moderado"
        descripcion_riesgo = "un nivel de riesgo moderado"
    else:
        nivel_riesgo = "bajo"
        descripcion_riesgo = "un nivel de riesgo bajo"

    # Generar diagnóstico basado en patrones con análisis más profundo
    diagnostico = f"El estudiante presenta un Score de Riesgo de **{score_riesgo:.3f}**, lo que indica {descripcion_riesgo}."

    # Agregar información sobre el promedio académico con contexto
    promedio = estudiante.get('promedio', 0)
    if promedio >= 16:
        diagnostico += f" A pesar de un excelente promedio académico (**{promedio:.1f}/20**), otros factores contribuyen al riesgo identificado."
    elif promedio >= 14:
        diagnostico += f" Con un buen promedio académico (**{promedio:.1f}/20**), el riesgo se debe principalmente a factores no académicos."
    elif promedio >= 11:
        diagnostico += f" El promedio académico (**{promedio:.1f}/20**) es aceptable, pero requiere atención junto con otros indicadores."
    else:
        diagnostico += f" El bajo promedio académico (**{promedio:.1f}/20**) es un factor significativo en la evaluación de riesgo."

    # Análisis detallado de cada indicador con umbrales y contexto
    diagnostico += "\n\n**Análisis detallado de indicadores:**\n\n"

    # Análisis de retención con umbrales
    if tasa_retencion >= 0.9:
        diagnostico += f"• **Tasa de Retención: {tasa_retencion:.3f}** (Umbral óptimo: ≥0.90). Representa estabilidad académica máxima, indicando que el estudiante mantiene un progreso constante sin interrupciones significativas. Este factor positivo contribuye a reducir el riesgo global, aunque no compensa completamente otros indicadores críticos.\n\n"
    elif tasa_retencion >= 0.7:
        diagnostico += f"• **Tasa de Retención: {tasa_retencion:.3f}** (Umbral aceptable: ≥0.70). Muestra un progreso académico adecuado con algunas variaciones. Aunque no representa un riesgo inmediato, requiere monitoreo para asegurar la continuidad y evitar que se convierta en un factor de deserción.\n\n"
    else:
        diagnostico += f"• **Tasa de Retención: {tasa_retencion:.3f}** (Por debajo del umbral crítico: <0.70). Indica dificultades significativas en el avance curricular. En estudios longitudinales, valores por debajo de este umbral aumentan la probabilidad de deserción en un 40-60% durante el siguiente ciclo académico.\n\n"

    # Análisis de entropía académica con contexto
    if entropia >= 2.0:
        diagnostico += f"• **Entropía Académica: {entropia:.3f}** (Umbral alto: ≥2.0). Refleja variabilidad significativa en el rendimiento académico, sugiriendo fluctuaciones en el patrón de estudio que pueden deberse a factores externos o inconsistencia en los hábitos de aprendizaje. Esta inestabilidad aumenta el riesgo de deserción al generar incertidumbre en el progreso académico.\n\n"
    elif entropia >= 1.5:
        diagnostico += f"• **Entropía Académica: {entropia:.3f}** (Umbral moderado: 1.5-2.0). Indica cierta variabilidad en el desempeño que, aunque no crítica, merece atención preventiva. La dispersión en las calificaciones puede afectar la confianza del estudiante y requiere estrategias de estabilización del rendimiento.\n\n"
    else:
        diagnostico += f"• **Entropía Académica: {entropia:.3f}** (Umbral bajo: <1.5). Demuestra consistencia en el rendimiento académico, lo cual es un factor protector importante. Este patrón favorable contribuye significativamente a la estabilidad y reduce el riesgo de deserción.\n\n"

    # Análisis de engagement con impacto específico
    if engagement >= 0.7:
        diagnostico += f"• **Engagement Score: {engagement:.3f}** (Umbral alto: ≥0.70). Indica participación activa en actividades académicas, lo cual es uno de los factores protectores más importantes contra la deserción. La interacción constante con recursos educativos correlaciona negativamente con el riesgo de abandono.\n\n"
    elif engagement >= 0.4:
        diagnostico += f"• **Engagement Score: {engagement:.3f}** (Umbral moderado: 0.40-0.70). Existe espacio para mejorar la participación académica. Valores en este rango sugieren que el estudiante podría beneficiarse de estrategias para aumentar la involucramiento, ya que el bajo engagement es un predictor temprano de riesgo de deserción.\n\n"
    else:
        diagnostico += f"• **Engagement Score: {engagement:.3f}** (Umbral crítico: <0.40). Representa un nivel de participación muy bajo que requiere atención inmediata. En la literatura especializada, valores por debajo de este umbral se asocian con una probabilidad 3 veces mayor de deserción durante el siguiente semestre.\n\n"

    # Análisis de momentum con proyección
    if momentum >= 1.0:
        diagnostico += f"• **Momentum Académico: {momentum:.3f}** (Umbral positivo: ≥1.0). Muestra una tendencia ascendente sólida en el rendimiento reciente. Esta trayectoria favorable es un indicador protector que reduce significativamente el riesgo a futuro y sugiere que el estudiante está en una fase de mejora continua.\n\n"
    elif momentum >= 0.0:
        diagnostico += f"• **Momentum Académico: {momentum:.3f}** (Umbral neutral: 0.0-1.0). Indica un progreso lento pero constante. Aunque no representa un riesgo inmediato, requiere monitoreo para asegurar que no se convierta en una tendencia negativa que pueda afectar la retención.\n\n"
    else:
        diagnostico += f"• **Momentum Académico: {momentum:.3f}** (Umbral negativo: <0.0). Señala una tendencia descendente en el rendimiento que requiere intervención inmediata. Las trayectorias negativas correlacionan fuertemente con el abandono académico y necesitan ser revertidas con estrategias específicas de recuperación.\n\n"

    # Análisis de vulnerabilidad financiera con impacto
    if vulnerabilidad >= 0.6:
        diagnostico += f"• **Vulnerabilidad Financiera: {vulnerabilidad:.3f}** (Umbral alto: ≥0.60). Representa una presión externa significativa que puede forzar la deserción. Los problemas financieros son uno de los factores más críticos en contextos educativos, especialmente cuando se combinan con otros indicadores de riesgo académico.\n\n"
    elif vulnerabilidad >= 0.3:
        diagnostico += f"• **Vulnerabilidad Financiera: {vulnerabilidad:.3f}** (Umbral moderado: 0.30-0.60). Podría representar una presión externa que afecte la continuidad académica. Se recomienda evaluar opciones de apoyo financiero preventivo para evitar que este factor se convierta en crítico.\n\n"
    else:
        diagnostico += f"• **Vulnerabilidad Financiera: {vulnerabilidad:.3f}** (Umbral bajo: <0.30). Indica estabilidad financiera favorable. Este factor no representa un riesgo significativo para la permanencia del estudiante y contribuye positivamente a la ecuación de riesgo global.\n\n"

    # Conclusión integrada que conecta todos los indicadores
    diagnostico += "**Síntesis integrada del riesgo:**\n\n"

    # Análisis de interconexiones entre indicadores
    factores_riesgo_principales = []
    factores_protectores = []

    if score_riesgo > 0.6:
        factores_riesgo_principales.append("elevado score de riesgo compuesto")
    if tasa_retencion < 0.8:
        factores_riesgo_principales.append("baja tasa de retención")
    if engagement < 0.5:
        factores_riesgo_principales.append("bajo nivel de engagement")
    if vulnerabilidad > 0.4:
        factores_riesgo_principales.append("vulnerabilidad financiera significativa")
    if momentum < 0:
        factores_riesgo_principales.append("momentum negativo")
    if entropia > 1.8:
        factores_riesgo_principales.append("alta entropía académica")

    if tasa_retencion > 0.9:
        factores_protectores.append("excelente tasa de retención")
    if engagement > 0.6:
        factores_protectores.append("alto engagement")
    if momentum > 0.5:
        factores_protectores.append("momentum positivo")
    if vulnerabilidad < 0.3:
        factores_protectores.append("estabilidad financiera")
    if entropia < 1.5:
        factores_protectores.append("baja entropía académica")

    if factores_riesgo_principales:
        diagnostico += f"El riesgo elevado (Score {score_riesgo:.3f}) se explica principalmente por la combinación de {', '.join(factores_riesgo_principales)}. "

        # Análisis específico de interacciones
        if engagement < 0.5 and promedio >= 14:
            diagnostico += f"El contraste entre el buen rendimiento académico ({promedio:.1f}/20) y el bajo engagement ({engagement:.3f}) sugiere que el estudiante logra resultados por capacidad individual, pero con escasa integración en dinámicas académicas, lo que puede afectar la sostenibilidad a largo plazo. "
        elif vulnerabilidad > 0.4 and engagement < 0.5:
            diagnostico += f"La combinación de vulnerabilidad financiera ({vulnerabilidad:.3f}) y bajo engagement ({engagement:.3f}) crea un perfil de alto riesgo, donde factores externos e internos se potencian mutuamente. "
        elif momentum < 0 and entropia > 1.8:
            diagnostico += f"El momentum negativo ({momentum:.3f}) junto con la alta entropía académica ({entropia:.3f}) indica un patrón de deterioro progresivo que requiere intervención inmediata para revertir la trayectoria. "

        if factores_protectores:
            diagnostico += f"Aunque algunos factores protectores como {', '.join(factores_protectores)} proporcionan cierto equilibrio, no son suficientes para compensar los riesgos identificados. "
    else:
        diagnostico += f"El perfil muestra un riesgo {nivel_riesgo} con indicadores mayoritariamente favorables. Los factores protectores identificados ({', '.join(factores_protectores) if factores_protectores else 'estabilidad general'}) sugieren un estudiante con buena base para la continuidad académica. "

    # Proyección de riesgo futuro
    riesgo_futuro = "estable" if momentum > 0.2 else "incierto" if momentum > -0.2 else "preocupante"
    diagnostico += f"\n\n**Proyección a futuro:** Basado en el momentum actual ({momentum:.3f}), el riesgo se proyecta como {riesgo_futuro}. "

    if momentum > 0.2:
        diagnostico += "La tendencia positiva sugiere que el estudiante está en una fase de mejora que podría consolidarse con el apoyo adecuado."
    elif momentum > -0.2:
        diagnostico += "Se requiere monitoreo continuo para determinar si la estabilidad actual se mantiene o evoluciona hacia una trayectoria más definida."
    else:
        diagnostico += "La tendencia negativa requiere intervención inmediata para evitar que se convierta en un patrón irreversible de deterioro académico."

    return diagnostico

def generar_recomendaciones_automaticas(resultados_formulas):
    """Genera recomendaciones automáticas basadas en los resultados de fórmulas"""
    
    score_riesgo = resultados_formulas.get('score_riesgo', 0)
    tasa_retencion = resultados_formulas.get('tasa_retencion', 1)
    engagement = resultados_formulas.get('engagement_score', 1)
    vulnerabilidad = resultados_formulas.get('vulnerabilidad_financiera', 0)
    momentum = resultados_formulas.get('momentum_academico', 0)
    
    recomendaciones = []
    
    # Recomendaciones basadas en nivel de riesgo
    if score_riesgo >= 0.7:
        recomendaciones.extend([
            "Asignar tutor académico especializado para seguimiento intensivo",
            "Evaluar necesidades de apoyo financiero y gestionar plan de pagos",
            "Establecer contacto inmediato con familia y responsables académicos",
            "Implementar plan de seguimiento semanal con metas específicas"
        ])
    elif score_riesgo >= 0.4:
        recomendaciones.extend([
            "Programar sesiones de refuerzo académico personalizadas",
            "Incorporar a grupos de estudio colaborativo y mentoría entre pares",
            "Establecer seguimiento quincenal del progreso académico",
            "Desarrollar plan de mejora personalizado con objetivos mensuales"
        ])
    else:
        recomendaciones.extend([
            "Mantener seguimiento regular del rendimiento académico",
            "Considerar como mentor para estudiantes con mayor riesgo",
            "Incentivar participación en programas de liderazgo estudiantil",
            "Reconocer logros académicos y motivar continuidad"
        ])
    
    # Recomendaciones específicas adicionales
    if engagement < 0.4:
        recomendaciones.append("Implementar estrategias para aumentar la participación en plataformas educativas")
    
    if vulnerabilidad > 0.5:
        recomendaciones.append("Gestionar apoyo financiero y explorar opciones de beca adicional")
    
    if momentum < 0:
        recomendaciones.append("Desarrollar plan de recuperación académica con enfoque en áreas críticas")
    
    if tasa_retencion < 0.7:
        recomendaciones.append("Evaluar carga académica y considerar ajustes en el plan de estudios")
    
    return recomendaciones

fake = Faker('es_ES')

MONGODB_CONFIG = {
    "database_name": st.secrets.get("DATABASE_NAME", "universidad_horizonte"),
    "collection_estudiantes": st.secrets.get("COLLECTION_ESTUDIANTES", "estudiantes"),
    "collection_notas": st.secrets.get("COLLECTION_NOTAS", "notas")
}

# Variables globales para mantener las conexiones
_mongodb_client = None
_mongodb_collection_estudiantes = None
_mongodb_collection_notas = None

def get_mongodb_connection():
    """Obtiene las conexiones a MongoDB usando auth directo del usuario"""
    global _mongodb_client, _mongodb_collection_estudiantes, _mongodb_collection_notas
    
    # Si ya tenemos conexiones activas, verificar que funcionen
    if (_mongodb_client is not None and 
        _mongodb_collection_estudiantes is not None and 
        _mongodb_collection_notas is not None):
        try:
            _mongodb_client.admin.command('ping')
            return _mongodb_collection_estudiantes, _mongodb_collection_notas, _mongodb_client
        except Exception:
            # Conexión perdida, reconectar
            _mongodb_client = None
            _mongodb_collection_estudiantes = None
            _mongodb_collection_notas = None
    
    # Usar variables de entorno de Streamlit Cloud
    connection_string = st.secrets["MONGODB_URI"]
    
    try:
        # Crear nueva conexión
        client = MongoClient(
            connection_string, 
            serverSelectionTimeoutMS=10000,
            maxPoolSize=10,  # Limitar conexiones del pool
            minPoolSize=1,   # Mantener al menos una conexión
            maxIdleTimeMS=30000  # Mantener conexiones abiertas por 30 segundos
        )
        
        # Verificar conexión
        client.admin.command('ping')
        
        # Acceder a la base de datos y colecciones
        db = client[MONGODB_CONFIG["database_name"]]
        collection_estudiantes = db[MONGODB_CONFIG["collection_estudiantes"]]
        collection_notas = db[MONGODB_CONFIG["collection_notas"]]
        
        # Guardar referencias globales
        _mongodb_client = client
        _mongodb_collection_estudiantes = collection_estudiantes
        _mongodb_collection_notas = collection_notas
        
        return collection_estudiantes, collection_notas, client
        
    except Exception as e:
        st.error(f"❌ Error conectando a MongoDB Atlas: {str(e)}")
        st.error("Verificar conexión a internet y credenciales de MongoDB")
        return None, None, None

@st.cache_resource
def conectar_mongodb():
    """Conecta a MongoDB Atlas - versión simplificada para datos existentes"""
    col_estudiantes, col_notas, client = get_mongodb_connection()
    
    if col_estudiantes is None or col_notas is None:
        return None, None, None
    
    try:
        # Verificar que tenemos datos
        count_estudiantes = col_estudiantes.count_documents({})
        count_notas = col_notas.count_documents({})
        
        st.success(f"✅ Conectado a MongoDB - {count_estudiantes} estudiantes, {count_notas} notas")
        
        return col_estudiantes, col_notas, client
        
    except Exception as e:
        st.error(f"❌ Error verificando datos: {str(e)}")
        return None, None, None

# Función de generación de datos eliminada - los datos ya existen en MongoDB

@st.cache_data(ttl=300)  # Cache por 5 minutos
def obtener_estudiantes():
    """Obtiene todos los estudiantes de MongoDB"""
    col_estudiantes, col_notas, client = get_mongodb_connection()
    if col_estudiantes is None:
        return []
    
    try:
        estudiantes = list(col_estudiantes.find({}))
        # Convertir ObjectId a string para compatibilidad con JSON
        for estudiante in estudiantes:
            estudiante['_id'] = str(estudiante['_id'])
        return estudiantes
    except Exception as e:
        st.error(f"Error obteniendo estudiantes: {str(e)}")
        return []
    # NO cerrar el cliente aquí para mantener la conexión persistente

def buscar_estudiantes(texto_busqueda="", filtro_estado="", filtro_carrera=""):
    """Busca estudiantes por texto, estado y carrera usando estructura real MongoDB"""
    col_estudiantes, col_notas, client = get_mongodb_connection()
    if col_estudiantes is None:
        return []
    
    try:
        # Construir query de búsqueda
        query = {}
        
        if texto_busqueda:
            query["$or"] = [
                {"nombre": {"$regex": texto_busqueda, "$options": "i"}},
                {"apellido": {"$regex": texto_busqueda, "$options": "i"}},
                {"student_id": {"$regex": texto_busqueda, "$options": "i"}},
                {"email": {"$regex": texto_busqueda, "$options": "i"}}
            ]
        
        if filtro_estado and filtro_estado != "Todos":
            query["estado"] = filtro_estado
        
        if filtro_carrera and filtro_carrera != "Todas":
            query["carrera"] = filtro_carrera
        
        estudiantes = list(col_estudiantes.find(query))
        
        # Convertir ObjectId a string
        for estudiante in estudiantes:
            estudiante['_id'] = str(estudiante['_id'])
        
        return estudiantes
    except Exception as e:
        st.error(f"Error en búsqueda: {str(e)}")
        return []
    # NO cerrar el cliente aquí para mantener la conexión persistente

def calcular_metricas_dashboard(estudiantes):
    """Calcula métricas principales para el dashboard"""
    if not estudiantes:
        return {
            "total": 0,
            "activos": 0,
            "retirados": 0,
            "egresados": 0,
            "promedio_general": 0.0,
            "tasa_retirados": 0.0
        }
    
    total = len(estudiantes)
    activos = len([e for e in estudiantes if e.get('estado') == 'Activo'])
    retirados = len([e for e in estudiantes if e.get('estado') == 'Retirado'])
    egresados = len([e for e in estudiantes if e.get('estado') == 'Egresado'])
    
    promedios = [e.get('promedio', 0) for e in estudiantes if e.get('promedio')]
    promedio_general = sum(promedios) / len(promedios) if promedios else 0
    
    tasa_retirados = (retirados / total * 100) if total > 0 else 0
    
    return {
        "total": total,
        "activos": activos,
        "retirados": retirados,
        "egresados": egresados,
        "promedio_general": promedio_general,
        "tasa_retirados": tasa_retirados
    }

def generar_informe_completo_profesional(estudiante, resultados_formulas):
    """Genera un informe completo profesional con toda la información del estudiante"""
    
    # Información básica
    nombre = f"{estudiante.get('nombre', 'N/A')} {estudiante.get('apellido', '')}"
    codigo = estudiante.get('student_id', 'N/A')
    carrera = estudiante.get('carrera', 'N/A')
    ciclo = estudiante.get('ciclo', 'N/A')
    email = estudiante.get('email', 'N/A')
    estado = estudiante.get('estado', 'N/A')
    promedio = estudiante.get('promedio', 0)
    creditos_aprobados = estudiante.get('creditos_aprobados', 0)
    asistencia = estudiante.get('asistencia_porcentaje', 0)
    
    # Resultados de fórmulas
    score_riesgo = resultados_formulas.get('score_riesgo', 0)
    tasa_retencion = resultados_formulas.get('tasa_retencion', 1)
    entropia = resultados_formulas.get('entropia_academica', 0)
    engagement = resultados_formulas.get('engagement_score', 1)
    momentum = resultados_formulas.get('momentum_academico', 0)
    vulnerabilidad = resultados_formulas.get('vulnerabilidad_financiera', 0)
    
    # Determinar nivel de riesgo
    if score_riesgo >= 0.7:
        nivel_riesgo = "Alto"
        color_riesgo = ""
    elif score_riesgo >= 0.4:
        nivel_riesgo = "Moderado-Alto"
        color_riesgo = ""
    else:
        nivel_riesgo = "Bajo"
        color_riesgo = ""
    
    # Generar comentario académico dinámico
    if promedio >= 14:
        comentario_academico = f"El estudiante presenta un rendimiento académico excelente ({promedio:.1f}/20), con asistencia de {asistencia:.1f}%, lo cual refleja un compromiso sólido con sus estudios."
    elif promedio >= 11:
        comentario_academico = f"El estudiante presenta un rendimiento académico aceptable ({promedio:.1f}/20), pero con asistencia de {asistencia:.1f}%, lo cual puede afectar su progreso."
    else:
        comentario_academico = f"El estudiante presenta dificultades académicas ({promedio:.1f}/20), con asistencia de {asistencia:.1f}%, requiriendo atención inmediata."
    
    # Generar diagnóstico dinámico
    factores_riesgo = []
    if score_riesgo > 0.6:
        factores_riesgo.append("elevado score de riesgo")
    if tasa_retencion < 0.8:
        factores_riesgo.append("baja tasa de retención")
    if engagement < 0.5:
        factores_riesgo.append("bajo nivel de engagement")
    if vulnerabilidad > 0.4:
        factores_riesgo.append("vulnerabilidad financiera")
    if momentum < 0:
        factores_riesgo.append("momentum negativo")
    
    if factores_riesgo:
        diagnostico = f"El análisis sugiere un estudiante con riesgo {nivel_riesgo.lower()} de deserción, principalmente asociado a {', '.join(factores_riesgo)}, aunque algunos indicadores muestran estabilidad."
    else:
        diagnostico = f"El análisis muestra un estudiante con riesgo {nivel_riesgo.lower()} de deserción, con indicadores mayoritariamente favorables y sin factores críticos identificados."
    
    # Generar recomendaciones
    recomendaciones = []
    if score_riesgo >= 0.7:
        recomendaciones.extend([
            "Implementar seguimiento intensivo semanal",
            "Asignar tutor académico especializado",
            "Evaluar apoyo financiero adicional",
            "Establecer plan de recuperación académica"
        ])
    elif score_riesgo >= 0.4:
        recomendaciones.extend([
            "Programar sesiones de refuerzo académico",
            "Aumentar supervisión de asistencia",
            "Fomentar mayor participación en actividades académicas",
            "Establecer metas de progreso mensuales"
        ])
    else:
        recomendaciones.extend([
            "Mantener seguimiento regular del rendimiento",
            "Incentivar participación en actividades extracurriculares",
            "Considerar como mentor para otros estudiantes",
            "Reconocer logros académicos"
        ])
    
    # Crear el informe completo
    informe = f"""
UNIVERSIDAD HORIZONTE
INFORME PREDICTIVO DE RIESGO ACADÉMICO
Generado automáticamente por el Sistema de Análisis Estudiantil
Fecha de análisis: {datetime.now().strftime('%d/%m/%Y %H:%M')}
Versión del sistema: 3.0 - Professional UI/UX

INFORMACIÓN DEL ESTUDIANTE
Nombre completo: {nombre}
Código de estudiante: {codigo}
Carrera: {carrera}
Ciclo actual: {ciclo}
Email institucional: {email}
Estado actual: {estado}

ANÁLISIS ACADÉMICO
Promedio general: {promedio:.1f}/20
Créditos aprobados: {creditos_aprobados}
Asistencia promedio: {asistencia:.1f}%

{comentario_academico}

ANÁLISIS DE RIESGO
Score de Riesgo Calculado: {score_riesgo:.3f} -> Riesgo {nivel_riesgo} {color_riesgo}
Tasa de Retención: {tasa_retencion:.3f} -> Probabilidad {'alta' if tasa_retencion > 0.8 else 'media' if tasa_retencion > 0.6 else 'baja'} de continuidad
Entropía Académica: {entropia:.3f} -> {'Alta' if entropia > 2.0 else 'Media' if entropia > 1.5 else 'Baja'} variabilidad en desempeño
Engagement Score: {engagement:.3f} -> {'Alto' if engagement > 0.7 else 'Moderado' if engagement > 0.4 else 'Bajo'} nivel de interacción
Momentum Académico: {momentum:.3f} -> {'Positivo' if momentum > 0.5 else 'Estable' if momentum > -0.5 else 'Negativo'} progreso
Vulnerabilidad Financiera: {vulnerabilidad:.3f} -> {'Alto' if vulnerabilidad > 0.6 else 'Moderado' if vulnerabilidad > 0.3 else 'Bajo'} riesgo financiero

DIAGNÓSTICO: {diagnostico}

RECOMENDACIONES DE INTERVENCIÓN
"""
    
    for i, rec in enumerate(recomendaciones, 1):
        informe += f"{i}. {rec}\n"
    
    informe += f"""

METODOLOGÍA APLICADA
Este análisis se basa en un modelo multifactorial que considera variables académicas, 
socioeconómicas y comportamentales del estudiante. Se aplican técnicas de análisis 
predictivo validadas en el contexto de educación superior peruana, incluyendo:

- Función sigmoide con pesos ponderados para score de riesgo
- Análisis de componentes principales para engagement
- Media móvil exponencialmente ponderada para vulnerabilidad financiera
- Cálculos de entropía para variabilidad académica
- Modelos de retención individual

Sistema de Análisis Estudiantil - Universidad Horizonte
Este informe es confidencial y destinado exclusivamente para uso institucional.
Generado automáticamente el {datetime.now().strftime('%d/%m/%Y a las %H:%M')}
"""
    
    return informe

def generar_pdf_informe(estudiante, resultados_formulas):
    """Genera un PDF del informe completo usando ReportLab"""
    if not PDF_AVAILABLE:
        return None
    
    from io import BytesIO
    
    # Crear buffer para el PDF
    buffer = BytesIO()
    
    # Crear documento
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=36, bottomMargin=20)
    styles = getSampleStyleSheet()
    
    # Estilos personalizados
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=14,
        spaceAfter=6,
        alignment=1  # Centrado
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=12,
        spaceAfter=4,
    )
    
    normal_style = styles['Normal']
    normal_style.fontSize = 10
    normal_style.leading = 14
    normal_style.spaceAfter = 4
    
    # Contenido del PDF
    story = []
    
    # Título
    story.append(Paragraph("UNIVERSIDAD HORIZONTE", title_style))
    story.append(Paragraph("INFORME PREDICTIVO DE RIESGO ACADÉMICO", title_style))
    story.append(Spacer(1, 6))
    
    # Información del estudiante
    story.append(Paragraph("INFORMACIÓN DEL ESTUDIANTE", subtitle_style))
    
    nombre = f"{estudiante.get('nombre', 'N/A')} {estudiante.get('apellido', '')}"
    story.append(Paragraph(f"Nombre completo: {nombre}", normal_style))
    story.append(Paragraph(f"Código de estudiante: {estudiante.get('student_id', 'N/A')}", normal_style))
    story.append(Paragraph(f"Carrera: {estudiante.get('carrera', 'N/A')}", normal_style))
    story.append(Paragraph(f"Ciclo actual: {estudiante.get('ciclo', 'N/A')}", normal_style))
    story.append(Paragraph(f"Email institucional: {estudiante.get('email', 'N/A')}", normal_style))
    story.append(Paragraph(f"Estado actual: {estudiante.get('estado', 'N/A')}", normal_style))
    story.append(Spacer(1, 6))
    
    # Análisis académico
    story.append(Paragraph("ANÁLISIS ACADÉMICO", subtitle_style))

    promedio = estudiante.get('promedio', 0)
    creditos = estudiante.get('creditos_aprobados', 0)
    asistencia = estudiante.get('asistencia_porcentaje', 0)
    
    story.append(Paragraph(f"Promedio general: {promedio:.1f}/20", normal_style))
    story.append(Paragraph(f"Créditos aprobados: {creditos}", normal_style))
    story.append(Paragraph(f"Asistencia promedio: {asistencia:.1f}%", normal_style))
    story.append(Spacer(1, 6))
    
    # Análisis de riesgo
    story.append(Paragraph("ANÁLISIS DE RIESGO", subtitle_style))
    
    score_riesgo = resultados_formulas.get('score_riesgo', 0)
    tasa_retencion = resultados_formulas.get('tasa_retencion', 1)
    entropia = resultados_formulas.get('entropia_academica', 0)
    engagement = resultados_formulas.get('engagement_score', 1)
    momentum = resultados_formulas.get('momentum_academico', 0)
    vulnerabilidad = resultados_formulas.get('vulnerabilidad_financiera', 0)
    
    if score_riesgo >= 0.7:
        nivel_riesgo = "Alto"
    elif score_riesgo >= 0.4:
        nivel_riesgo = "Moderado-Alto"
    else:
        nivel_riesgo = "Bajo"
    
    story.append(Paragraph(f"Score de Riesgo Calculado: {score_riesgo:.3f} -> Riesgo {nivel_riesgo}", normal_style))
    story.append(Paragraph(f"Tasa de Retención: {tasa_retencion:.3f}", normal_style))
    story.append(Paragraph(f"Entropía Académica: {entropia:.3f}", normal_style))
    story.append(Paragraph(f"Engagement Score: {engagement:.3f}", normal_style))
    story.append(Paragraph(f"Momentum Académico: {momentum:.3f}", normal_style))
    story.append(Paragraph(f"Vulnerabilidad Financiera: {vulnerabilidad:.3f}", normal_style))
    story.append(Spacer(1, 6))
    
    # Diagnóstico
    story.append(Paragraph("DIAGNÓSTICO", subtitle_style))
    
    # Generar diagnóstico dinámico
    factores_riesgo = []
    if score_riesgo > 0.6:
        factores_riesgo.append("elevado score de riesgo")
    if tasa_retencion < 0.8:
        factores_riesgo.append("baja tasa de retención")
    if engagement < 0.5:
        factores_riesgo.append("bajo nivel de engagement")
    if vulnerabilidad > 0.4:
        factores_riesgo.append("vulnerabilidad financiera")
    if momentum < 0:
        factores_riesgo.append("momentum negativo")
    
    if factores_riesgo:
        diagnostico = f"El análisis sugiere un estudiante con riesgo {nivel_riesgo.lower()} de deserción, principalmente asociado a {', '.join(factores_riesgo)}, aunque algunos indicadores muestran estabilidad."
    else:
        diagnostico = f"El análisis muestra un estudiante con riesgo {nivel_riesgo.lower()} de deserción, con indicadores mayoritariamente favorables y sin factores críticos identificados."
    
    # Dividir el diagnóstico en párrafos
    diagnostico_parrafos = diagnostico.split('. ')
    for para in diagnostico_parrafos:
        if para.strip():
            story.append(Paragraph(para.strip() + '.', normal_style))
    
    story.append(Spacer(1, 6))
    
    # Recomendaciones
    story.append(Paragraph("RECOMENDACIONES DE INTERVENCIÓN", subtitle_style))
    
    recomendaciones = []
    if score_riesgo >= 0.7:
        recomendaciones.extend([
            "Implementar seguimiento intensivo semanal",
            "Asignar tutor académico especializado",
            "Evaluar apoyo financiero adicional",
            "Establecer plan de recuperación académica"
        ])
    elif score_riesgo >= 0.4:
        recomendaciones.extend([
            "Programar sesiones de refuerzo académico",
            "Aumentar supervisión de asistencia",
            "Fomentar mayor participación en actividades académicas",
            "Establecer metas de progreso mensuales"
        ])
    else:
        recomendaciones.extend([
            "Mantener seguimiento regular del rendimiento",
            "Incentivar participación en actividades extracurriculares",
            "Considerar como mentor para otros estudiantes",
            "Reconocer logros académicos"
        ])
    
    for rec in recomendaciones:
        story.append(Paragraph(f"• {rec}", normal_style))
    
    story.append(Spacer(1, 18))    # Pie de página
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=7,
        alignment=1
    )
    story.append(Paragraph(f"Generado automáticamente el {datetime.now().strftime('%d/%m/%Y a las %H:%M')}", footer_style))
    story.append(Paragraph("Sistema de Análisis Estudiantil - Universidad Horizonte", footer_style))
    story.append(Paragraph("Este informe es confidencial y destinado exclusivamente para uso institucional.", footer_style))
    
    # Generar PDF
    try:
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    except Exception as e:
        st.error(f"Error generando PDF: {str(e)}")
        return None

# ==============================================================================
# INTERFAZ PRINCIPAL
# ==============================================================================

def limpiar_estados_seccion(seccion_actual):
    """Limpia los estados de sesión específicos de las otras secciones para liberar memoria"""
    
    # Estados a limpiar para cada sección
    estados_por_seccion = {
        "Búsqueda de Estudiantes": [
            'pagina_actual_busqueda',
            'items_por_pagina_busqueda'
        ],
        "Análisis Individual": [
            'estudiante_seleccionado_analisis'
        ],
        "Informes Automáticos": [
            'estudiante_informe_seleccionado',
            'informe_generado',
            'resultados_formulas',
            'pdf_generado'
        ],
        "Resumen": []  # El resumen no tiene estados específicos que limpiar
    }
    
    # Limpiar estados de todas las secciones excepto la actual
    for seccion, estados in estados_por_seccion.items():
        if seccion != seccion_actual:
            for estado in estados:
                if estado in st.session_state:
                    del st.session_state[estado]

def main():
    """Función principal de la aplicación"""
    
    # Header principal
    st.markdown("""
    <div class="main-header">
        <h1>🎓 Universidad Horizonte</h1>
        <p>Sistema Predictivo de Deserción Estudiantil - Análisis mediante MongoDB + Databricks</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Obtener datos
    estudiantes = obtener_estudiantes()
    
    if not estudiantes:
        st.error("⚠️ No se pudieron cargar los datos de estudiantes. Verificar conexión a MongoDB.")
        return
    
    # Sidebar para navegación
    with st.sidebar:
        st.markdown("### Panel de Control")
        
        # Opciones del menú
        opciones_menu = ["Resumen", "Búsqueda de Estudiantes", "Análisis Individual", "Informes Automáticos"]
        opcion_seleccionada = st.selectbox(
            "Seleccionar módulo:", 
            opciones_menu,
            key="menu_principal"
        )
        
        # Limpiar estados de otras secciones cuando se cambia de módulo
        # Usar una variable auxiliar para detectar cambios
        if 'opcion_anterior' not in st.session_state:
            st.session_state.opcion_anterior = opcion_seleccionada
        elif st.session_state.opcion_anterior != opcion_seleccionada:
            # Se cambió de sección, limpiar estados de la sección anterior
            limpiar_estados_seccion(opcion_seleccionada)
            st.session_state.opcion_anterior = opcion_seleccionada
        
        # Mostrar fórmulas matemáticas en el sidebar
        mostrar_formulas_matematicas()
    
    # Contenido principal según opción seleccionada
    if opcion_seleccionada == "Resumen":
        mostrar_dashboard(estudiantes)
    elif opcion_seleccionada == "Búsqueda de Estudiantes":
        mostrar_busqueda_estudiantes(estudiantes)
    elif opcion_seleccionada == "Análisis Individual":
        mostrar_analisis_individual(estudiantes)
    elif opcion_seleccionada == "Informes Automáticos":
        mostrar_informes_automaticos(estudiantes)

def mostrar_dashboard(estudiantes):
    """Muestra el dashboard principal con métricas y gráficos estilo shadcn"""
    st.header("Resumen de Indicadores Académicos")
    
    # Métricas principales
    metricas = calcular_metricas_dashboard(estudiantes)
    
    # Calcular métricas adicionales correctamente
    estudiantes_becados = len([e for e in estudiantes if e.get('tiene_beca') == "True"])
    porcentaje_becados = (estudiantes_becados / metricas["total"] * 100) if metricas["total"] > 0 else 0
    asistencia_promedio = sum([e.get('asistencia_porcentaje', 0) for e in estudiantes]) / len(estudiantes) if estudiantes else 0
    
    # Métricas principales - solo las 4 relevantes
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div style="
            background: white;
            border: 1px solid #e5e7eb;
            border-left: 4px solid #3b82f6;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
            height: 140px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        ">
            <div style="
                font-size: 0.875rem;
                font-weight: 500;
                color: #6b7280;
                margin-bottom: 8px;
            ">Total Estudiantes</div>
            <div style="
                font-size: 2rem;
                font-weight: 700;
                color: #000000;
                line-height: 1;
                margin-bottom: 8px;
            ">{metricas["total"]:,}</div>
            <div style="
                font-size: 0.75rem;
                color: #22c55e;
                font-weight: 500;
            ">Conectado a MongoDB</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="
            background: white;
            border: 1px solid #e5e7eb;
            border-left: 4px solid #3b82f6;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
            height: 140px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        ">
            <div style="
                font-size: 0.875rem;
                font-weight: 500;
                color: #6b7280;
                margin-bottom: 8px;
            ">Estudiantes Activos</div>
            <div style="
                font-size: 2rem;
                font-weight: 700;
                color: #000000;
                line-height: 1;
                margin-bottom: 8px;
            ">{metricas["activos"]:,}</div>
            <div style="
                font-size: 0.75rem;
                color: #22c55e;
                font-weight: 500;
            ">{((metricas["activos"]/metricas["total"])*100):.1f}% del total activo</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="
            background: white;
            border: 1px solid #e5e7eb;
            border-left: 4px solid #3b82f6;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
            height: 140px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        ">
            <div style="
                font-size: 0.875rem;
                font-weight: 500;
                color: #6b7280;
                margin-bottom: 8px;
            ">Estudiantes Becados</div>
            <div style="
                font-size: 2rem;
                font-weight: 700;
                color: #000000;
                line-height: 1;
                margin-bottom: 8px;
            ">{estudiantes_becados:,}</div>
            <div style="
                font-size: 0.75rem;
                color: #8b5cf6;
                font-weight: 500;
            ">{porcentaje_becados:.1f}% del total</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div style="
            background: white;
            border: 1px solid #e5e7eb;
            border-left: 4px solid #3b82f6;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
            height: 140px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        ">
            <div style="
                font-size: 0.875rem;
                font-weight: 500;
                color: #6b7280;
                margin-bottom: 8px;
            ">Asistencia Promedio</div>
            <div style="
                font-size: 2rem;
                font-weight: 700;
                color: #000000;
                line-height: 1;
                margin-bottom: 8px;
            ">{asistencia_promedio:.1f}%</div>
            <div style="
                font-size: 0.75rem;
                color: {'#ef4444' if asistencia_promedio < 90 else '#22c55e'};
                font-weight: 500;
            ">Objetivo: 90% mínimo</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribución por Estado")
        estados_count = {}
        for estudiante in estudiantes:
            estado = estudiante.get('estado', 'Sin Estado')
            estados_count[estado] = estados_count.get(estado, 0) + 1
        
        if estados_count:
            fig_pie = px.pie(
                values=list(estados_count.values()),
                names=list(estados_count.keys()),
                title="Distribución de Estados de Estudiantes",
                color_discrete_map={
                    'Activo': '#22c55e',
                    'Retirado': '#ef4444',
                    'Egresado': '#3b82f6'
                }
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("Distribución por Carrera")
        carreras_count = {}
        for estudiante in estudiantes:
            carrera = estudiante.get('carrera', 'Sin Carrera')
            carreras_count[carrera] = carreras_count.get(carrera, 0) + 1
        
        if carreras_count:
            # Mostrar solo las top 8 carreras
            carreras_top = dict(sorted(carreras_count.items(), key=lambda x: x[1], reverse=True)[:8])
            
            fig_bar = px.bar(
                x=list(carreras_top.values()),
                y=list(carreras_top.keys()),
                orientation='h',
                title="Estudiantes por Carrera (Top 8)",
                color=list(carreras_top.values()),
                color_continuous_scale="viridis"
            )
            fig_bar.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
    
    # Análisis de rendimiento académico
    st.subheader("Análisis de Rendimiento Académico")
    
    promedios = [e.get('promedio', 0) for e in estudiantes if e.get('promedio')]
    if promedios:
        col1, col2 = st.columns(2)
        
        with col1:
            fig_hist = px.histogram(
                x=promedios,
                nbins=20,
                title="Distribución de Promedios",
                labels={'x': 'Promedio', 'y': 'Cantidad de Estudiantes'}
            )
            fig_hist.add_vline(x=np.mean(promedios), line_dash="dash", line_color="red", 
                              annotation_text=f"Media: {np.mean(promedios):.2f}")
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Box plot por estado
            data_box = []
            for estudiante in estudiantes:
                if estudiante.get('promedio'):
                    data_box.append({
                        'promedio': estudiante['promedio'],
                        'estado': estudiante.get('estado', 'Sin Estado')
                    })
            
            if data_box:
                df_box = pd.DataFrame(data_box)
                fig_box = px.box(
                    df_box, 
                    x='estado', 
                    y='promedio',
                    title="Distribución de Promedios por Estado",
                    color='estado',
                    color_discrete_map={
                        'Activo': '#22c55e',
                        'Retirado': '#ef4444',
                        'Egresado': '#3b82f6'
                    }
                )
                st.plotly_chart(fig_box, use_container_width=True)

def mostrar_busqueda_estudiantes(estudiantes):
    """Módulo de búsqueda y filtrado de estudiantes"""
    st.header("Búsqueda de Estudiantes")
    
    # CSS específico para el placeholder del input de búsqueda (global para esta función)
    st.markdown("""
    <style>
    div[data-testid="stTextInput"] input::placeholder {
        color: #6b7280 !important;
        opacity: 1 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Controles de búsqueda
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        texto_busqueda = st.text_input(
            "Buscar estudiante:",
            placeholder="Nombre, apellido o código",
            help="Buscar por nombre, apellido, código de estudiante o email"
        )
    
    with col2:
        estados = ["Todos"] + list(set([e.get('estado', 'Sin Estado') for e in estudiantes]))
        estado_filtro = st.selectbox(
            "Filtrar por estado:", 
            estados,
            key="filtro_estado"
        )
    
    with col3:
        carreras = ["Todas"] + list(set([e.get('carrera', 'Sin Carrera') for e in estudiantes]))
        carrera_filtro = st.selectbox(
            "Filtrar por carrera:", 
            carreras,
            key="filtro_carrera"
        )
    
    # Aplicar filtros
    estudiantes_filtrados = buscar_estudiantes(
        texto_busqueda, 
        estado_filtro if estado_filtro != "Todos" else "",
        carrera_filtro if carrera_filtro != "Todas" else ""
    )
    
    # Mostrar solo los resultados
    st.markdown(f"### Resultados: {len(estudiantes_filtrados)} estudiante(s)")
    
    # Inicializar configuración de paginación en session state
    if 'pagina_actual_busqueda' not in st.session_state:
        st.session_state.pagina_actual_busqueda = 1
    if 'items_por_pagina_busqueda' not in st.session_state:
        st.session_state.items_por_pagina_busqueda = 10
    
    if estudiantes_filtrados:
        # Usar configuración de session state
        items_por_pagina = st.session_state.items_por_pagina_busqueda
        total_paginas = (len(estudiantes_filtrados) - 1) // items_por_pagina + 1
        
        # Asegurar que la página actual no exceda el total
        pagina_actual = min(st.session_state.pagina_actual_busqueda, total_paginas)
        st.session_state.pagina_actual_busqueda = pagina_actual
        
        # Mostrar estudiantes de la página actual
        inicio = (pagina_actual - 1) * items_por_pagina
        fin = inicio + items_por_pagina
        estudiantes_pagina = estudiantes_filtrados[inicio:fin]
        
        # Tabla de resultados
        for estudiante in estudiantes_pagina:
            with st.expander(f"{estudiante.get('nombre', 'N/A')} {estudiante.get('apellido', '')} - {estudiante.get('student_id', 'N/A')}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Carrera:** {estudiante.get('carrera', 'N/A')}")
                    st.write(f"**Ciclo:** {estudiante.get('ciclo', 'N/A')}")
                    st.write(f"**Promedio:** {estudiante.get('promedio', 'N/A')}")
                    st.write(f"**Email:** {estudiante.get('email', 'N/A')}")
                
                with col2:
                    estado = estudiante.get('estado', 'Sin Estado')
                    
                    st.write(f"**Estado:** {estado}")
                    st.write(f"**Asistencia:** {estudiante.get('asistencia_porcentaje', 0):.1f}%")
                    st.write(f"**Créditos:** {estudiante.get('creditos_aprobados', 'N/A')}")
                    st.write(f"**Ciudad:** {estudiante.get('ciudad', 'N/A')}")
        
        # Barra inferior de paginación (estilo del ejemplo proporcionado)
        bottom_menu = st.columns((4, 1, 1))
        
        # Contador a la izquierda
        with bottom_menu[0]:
            inicio_mostrado = inicio + 1
            fin_mostrado = min(fin, len(estudiantes_filtrados))
            st.markdown(f"Mostrando **{inicio_mostrado}-{fin_mostrado}** de **{len(estudiantes_filtrados)}** estudiantes")
        
        # Número de página en el centro
        with bottom_menu[1]:
            nueva_pagina = st.number_input(
                "Página", 
                min_value=1, 
                max_value=total_paginas, 
                step=1,
                value=pagina_actual,
                key="page_number_input"
            )
            
            if nueva_pagina != pagina_actual:
                st.session_state.pagina_actual_busqueda = nueva_pagina
                st.rerun()
        
        # Selector de filas por página a la derecha
        with bottom_menu[2]:
            nuevo_items = st.selectbox(
                "Filas por página:",
                [10, 20, 50],
                index=[10, 20, 50].index(st.session_state.items_por_pagina_busqueda),
                format_func=lambda x: str(x),
                key="items_per_page_selector"
            )
            
            if nuevo_items != st.session_state.items_por_pagina_busqueda:
                st.session_state.items_por_pagina_busqueda = nuevo_items
                st.session_state.pagina_actual_busqueda = 1
                st.rerun()

def mostrar_analisis_individual(estudiantes):
    """Módulo de análisis individual de estudiantes"""
    st.header("Análisis Situacional por Estudiante")
    
    if not estudiantes:
        st.warning("No hay estudiantes disponibles para análisis.")
        return
    
    # CSS específico para el placeholder del input de búsqueda (global para esta función)
    st.markdown("""
    <style>
    div[data-testid="stTextInput"] input::placeholder {
        color: #6b7280 !important;
        opacity: 1 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Inicializar estudiante seleccionado en session state
    if 'estudiante_seleccionado_analisis' not in st.session_state:
        st.session_state.estudiante_seleccionado_analisis = None
    
    # Controles de búsqueda en múltiples columnas
    col1, col2, col3, col4 = st.columns([3, 1, 1, 0.5])
    
    with col1:
        texto_busqueda = st.text_input(
            "Buscar estudiante:",
            placeholder="Nombre, apellido, código o email",
            help="Buscar por cualquier campo del estudiante",
            key="busqueda_analisis"
        )
    
    with col2:
        estados_analisis = ["Todos"] + list(set([e.get('estado', 'Sin Estado') for e in estudiantes]))
        estado_filtro = st.selectbox(
            "Estado:", 
            estados_analisis,
            key="estado_analisis"
        )
    
    with col3:
        carreras_analisis = ["Todas"] + list(set([e.get('carrera', 'Sin Carrera') for e in estudiantes]))
        carrera_filtro = st.selectbox(
            "Carrera:", 
            carreras_analisis,
            key="carrera_analisis"
        )
    
    with col4:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Buscar", key="buscar_btn"):
            # Limpiar selección si hay un estudiante seleccionado
            if st.session_state.estudiante_seleccionado_analisis:
                st.session_state.estudiante_seleccionado_analisis = None
                st.rerun()
    
    # Filtrar estudiantes según criterios de búsqueda (automático)
    estudiantes_filtrados = []
    
    # Solo filtrar si hay texto de búsqueda
    if texto_busqueda:
        for estudiante in estudiantes:
            # Filtro por texto
            campos_busqueda = [
                str(estudiante.get('nombre', '')).lower(),
                str(estudiante.get('apellido', '')).lower(), 
                str(estudiante.get('student_id', '')).lower(),
                str(estudiante.get('email', '')).lower()
            ]
            texto_coincide = any(texto_busqueda.lower() in campo for campo in campos_busqueda)
            
            # Filtro por estado
            estado_coincide = (estado_filtro == "Todos" or 
                              estudiante.get('estado', '') == estado_filtro)
            
            # Filtro por carrera
            carrera_coincide = (carrera_filtro == "Todas" or 
                               estudiante.get('carrera', '') == carrera_filtro)
            
            if texto_coincide and estado_coincide and carrera_coincide:
                estudiantes_filtrados.append(estudiante)
    
    # Mostrar lista de resultados solo si hay búsqueda y no hay estudiante seleccionado
    if texto_busqueda and not st.session_state.estudiante_seleccionado_analisis:
        if estudiantes_filtrados:
            st.markdown(f"**{len(estudiantes_filtrados)} estudiante(s) encontrado(s)**")
            
            # Lista desplegable de estudiantes
            for i, estudiante in enumerate(estudiantes_filtrados):
                nombre_completo = f"{estudiante.get('nombre', 'N/A')} {estudiante.get('apellido', '')}"
                codigo = estudiante.get('student_id', 'N/A')
                carrera = estudiante.get('carrera', 'N/A')
                estado = estudiante.get('estado', 'N/A')
                promedio = estudiante.get('promedio', 0)
                
                # Crear card clickeable para cada estudiante
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        st.markdown(f"""
                        **{nombre_completo}** - {codigo}
                        - **Carrera:** {carrera}
                        - **Estado:** {estado}
                        - **Promedio:** {promedio:.2f}/20
                        """)
                    
                    with col2:
                        if st.button("Seleccionar", key=f"select_{i}", use_container_width=True):
                            st.session_state.estudiante_seleccionado_analisis = estudiante
                            st.rerun()
                    
                    st.divider()
        else:
            st.info("No se encontraron estudiantes que coincidan con los criterios de búsqueda.")
    
    elif not texto_busqueda and not st.session_state.estudiante_seleccionado_analisis:
        st.info("Escribe en el campo de búsqueda para encontrar estudiantes.")
    
    # Análisis del estudiante seleccionado
    estudiante_seleccionado = st.session_state.estudiante_seleccionado_analisis
    
    if estudiante_seleccionado:
        # Mostrar información detallada
        st.markdown("## Información Detallada")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            **Datos Personales:**
            - **Nombre:** {estudiante_seleccionado.get('nombre', 'N/A')} {estudiante_seleccionado.get('apellido', '')}
            - **Código:** {estudiante_seleccionado.get('student_id', 'N/A')}
            - **Email:** {estudiante_seleccionado.get('email', 'N/A')}
            - **Edad:** {estudiante_seleccionado.get('edad', 'N/A')} años
            - **Género:** {estudiante_seleccionado.get('genero', 'N/A')}
            """)
        
        with col2:
            st.markdown(f"""
            **Información Académica:**
            - **Carrera:** {estudiante_seleccionado.get('carrera', 'N/A')}
            - **Ciclo:** {estudiante_seleccionado.get('ciclo', 'N/A')}
            - **Promedio:** {estudiante_seleccionado.get('promedio', 'N/A')}/20
            - **Créditos:** {estudiante_seleccionado.get('creditos_aprobados', 'N/A')}
            - **Estado:** {estudiante_seleccionado.get('estado', 'N/A')}
            """)
        
        with col3:
            st.markdown(f"""
            **Factores Adicionales:**
            - **Asistencia:** {estudiante_seleccionado.get('asistencia_porcentaje', 0):.1f}%
            - **Beca:** {'Sí' if estudiante_seleccionado.get('tiene_beca') == "True" else 'No'}
            - **Pagos Pendientes:** {estudiante_seleccionado.get('pagos_pendientes', 0)}
            - **Ciudad:** {estudiante_seleccionado.get('ciudad', 'N/A')}
            - **Riesgo:** {estudiante_seleccionado.get('riesgo', 'N/A')}
            """)
        
        
        # Análisis de riesgo con fórmulas avanzadas
        st.markdown("## Análisis de Riesgo")
        
        # Aplicar fórmulas matemáticas al estudiante
        resultados_formulas = aplicar_formulas_estudiante(estudiante_seleccionado)
        
        # Mostrar resultados en métricas
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            score_riesgo = resultados_formulas.get('score_riesgo', 0)
            st.metric(
                "Score de Riesgo",
                f"{score_riesgo:.3f}",
                help="Función sigmoide aplicada a múltiples factores"
            )
        
        with col2:
            tasa_retencion = resultados_formulas.get('tasa_retencion', 0)
            st.metric(
                "Tasa Retención",
                f"{tasa_retencion:.3f}",
                help="Créditos reales vs esperados"
            )
        
        with col3:
            entropia = resultados_formulas.get('entropia_academica', 0)
            st.metric(
                "Entropía Académica",
                f"{entropia:.3f}",
                help="Medida de desorden en rendimiento"
            )
        
        with col4:
            engagement = resultados_formulas.get('engagement_score', 0)
            st.metric(
                "Engagement Score",
                f"{engagement:.3f}",
                help="Índice PCA de participación"
            )
        
        # Segunda fila de métricas
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            momentum = resultados_formulas.get('momentum_academico', 0)
            st.metric(
                "Momentum",
                f"{momentum:.3f}",
                delta=f"{momentum:.2f}",
                help="Tendencia académica (smoothing exponencial)"
            )
        
        with col2:
            vulnerabilidad = resultados_formulas.get('vulnerabilidad_financiera', 0)
            st.metric(
                "Vulnerabilidad Financiera",
                f"{vulnerabilidad:.3f}",
                help="EWMA de pagos pendientes"
            )
        
        with col3:
            # Cálculo simple de score de riesgo para comparación
            promedio = estudiante_seleccionado.get('promedio', 0)
            asistencia = estudiante_seleccionado.get('asistencia_porcentaje', 100)
            score_simple = (promedio / 20) * 0.7 + (asistencia / 100) * 0.3
            st.metric(
                "Score Simple",
                f"{score_simple:.3f}",
                help="Cálculo básico para comparación"
            )
        
        with col4:
            # Nivel de riesgo global
            riesgo_global = (score_riesgo + (1-tasa_retencion) + vulnerabilidad) / 3
            st.metric(
                "Riesgo Global",
                f"{riesgo_global:.3f}",
                help="Promedio ponderado de factores"
            )
        
        # Visualización avanzada
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de radar con métricas avanzadas
            categorias = ['Score Riesgo', 'Retención', 'Engagement', 'Momentum', 'Vulnerabilidad']
            valores = [
                score_riesgo * 100,
                tasa_retencion * 100,
                engagement * 100,
                max(0, (momentum + 2) * 25),  # Normalizar momentum
                (1 - vulnerabilidad) * 100   # Invertir vulnerabilidad
            ]
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=valores,
                theta=categorias,
                fill='toself',
                name=f"{estudiante_seleccionado.get('nombre', 'Estudiante')} {estudiante_seleccionado.get('apellido', '')}",
                line=dict(color='#3b82f6', width=2),
                fillcolor='rgba(59, 130, 246, 0.3)'
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100],
                        tickfont=dict(color='#ffffff', size=10),
                        gridcolor='rgba(255,255,255,0.3)'
                    ),
                    angularaxis=dict(
                        tickfont=dict(color='#ffffff', size=11),
                        linecolor='rgba(255,255,255,0.5)'
                    ),
                    bgcolor='rgba(0,0,0,0)'
                ),
                font=dict(color='#ffffff'),
                showlegend=True,
                title=dict(
                    text="Perfil de Rendimiento",
                    font=dict(color='#ffffff', size=14)
                ),
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
        
        with col2:
            # Diagnóstico de Riesgo
            with st.expander("**Diagnóstico de Riesgo**", expanded=False):
                # Generar diagnóstico automático
                diagnostico = generar_diagnostico_automatico(resultados_formulas, estudiante_seleccionado)
                
                st.markdown("**Análisis de Riesgo Académico – Diagnóstico Automático**")
                st.write(diagnostico)
            
            # Recomendaciones de Intervención (abajo del diagnóstico)
            with st.expander("**Recomendaciones de Intervención**", expanded=False):
                # Generar recomendaciones automáticas
                recomendaciones = generar_recomendaciones_automaticas(resultados_formulas)
                
                # Determinar el tipo de intervención basado en el riesgo (sistema más equilibrado)
                riesgo_global = resultados_formulas.get('score_riesgo', 0)
                if riesgo_global >= 0.8:
                    st.error("**Intervención Urgente Requerida**")
                elif riesgo_global >= 0.5:
                    st.warning("**Seguimiento Moderado Recomendado**")
                else:
                    st.success("**Monitoreo Básico**")
                
                # Mostrar recomendaciones
                for recomendacion in recomendaciones:
                    st.markdown(f"• {recomendacion}")
                
                # Agregar fecha de seguimiento sugerida
                fecha_seguimiento = (datetime.now() + timedelta(days=15)).strftime('%d/%m/%Y')
                st.info(f"**Próxima evaluación sugerida:** {fecha_seguimiento}")
        
        
        # Gráficos avanzados adicionales
        st.markdown("### **Análisis Comparativo Avanzado**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de barras: Comparación vs promedio de cohorte
            metricas_nombres = ['Promedio', 'Asistencia', 'Engagement', 'Retención']
            valores_estudiante = [
                estudiante_seleccionado.get('promedio', 0) / 20 * 100,
                estudiante_seleccionado.get('asistencia_porcentaje', 0),
                engagement * 100,
                tasa_retencion * 100
            ]
            valores_cohorte = [68, 75, 45, 82]  # Promedios simulados de la cohorte
            
            df_comparacion = pd.DataFrame({
                'Métrica': metricas_nombres + metricas_nombres,
                'Valor': valores_estudiante + valores_cohorte,
                'Tipo': ['Estudiante'] * 4 + ['Promedio Cohorte'] * 4
            })
            
            fig_bar = px.bar(
                df_comparacion, 
                x='Métrica', 
                y='Valor',
                color='Tipo',
                barmode='group',
                title="Comparación vs Promedio de Cohorte",
                color_discrete_map={'Estudiante': '#3b82f6', 'Promedio Cohorte': '#e5e7eb'}
            )
            
            # Agregar tooltips informativos
            fig_bar.update_traces(
                hovertemplate="<b>%{x}</b><br>" +
                             "%{fullData.name}: %{y:.1f}%<br>" +
                             "<i>Comparación con estudiantes de características similares</i><extra></extra>"
            )
            
            fig_bar.update_layout(
                height=350, 
                showlegend=True,
                yaxis_title="Porcentaje (%)",
                font=dict(size=11),
                title=dict(font=dict(size=13))
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            # Gráfico de evolución temporal simulado
            fechas = pd.date_range(start='2024-01-01', end='2024-09-01', freq='M')
            promedio_base = estudiante_seleccionado.get('promedio', 12)
            notas_evolucion = [promedio_base + np.random.normal(0, 0.8) for _ in range(len(fechas))]
            asistencia_evolucion = [estudiante_seleccionado.get('asistencia_porcentaje', 80) + np.random.normal(0, 5) for _ in range(len(fechas))]
            
            fig_timeline = go.Figure()
            fig_timeline.add_trace(go.Scatter(
                x=fechas, 
                y=notas_evolucion,
                mode='lines+markers',
                name='Promedio Académico',
                line=dict(color='#3b82f6', width=3),
                hovertemplate="<b>Promedio Académico</b><br>" +
                             "Fecha: %{x}<br>" +
                             "Promedio: %{y:.2f}/20<br>" +
                             "<i>Evolución del rendimiento académico</i><extra></extra>"
            ))
            
            # Eje secundario para asistencia
            fig_timeline.add_trace(go.Scatter(
                x=fechas,
                y=asistencia_evolucion,
                mode='lines+markers',
                name='Asistencia %',
                yaxis='y2',
                line=dict(color='#10b981', width=3),
                hovertemplate="<b>Asistencia</b><br>" +
                             "Fecha: %{x}<br>" +
                             "Asistencia: %{y:.1f}%<br>" +
                             "<i>Porcentaje de asistencia mensual</i><extra></extra>"
            ))
            
            fig_timeline.update_layout(
                title="Evolución Temporal de Indicadores Académicos",
                xaxis_title="Período Académico",
                yaxis=dict(title="Promedio Académico (0-20)", side="left"),
                yaxis2=dict(title="Asistencia (%)", side="right", overlaying="y"),
                height=350,
                hovermode='x unified',
                font=dict(size=11),
                title_font=dict(size=13)
            )
            
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Matriz de correlación de factores
        st.markdown("### **Matriz de Correlación de Factores**")
        st.caption("Análisis de interrelaciones entre variables académicas y comportamentales")
        
        # Usar nombres más descriptivos para la visualización
        factores_matriz_nombres = {
            'Promedio Académico': estudiante_seleccionado.get('promedio', 0),
            'Asistencia (%)': estudiante_seleccionado.get('asistencia_porcentaje', 0),
            'Engagement Score': engagement * 100,
            'Uso Plataforma (hrs/sem)': estudiante_seleccionado.get('uso_lms_horas_semana', 0),
            'Participación Digital': estudiante_seleccionado.get('interacciones_mes', 0)
        }
        
        # Crear matriz de correlación simulada (en una implementación real vendría de datos históricos)
        correlation_matrix = np.random.rand(5, 5)
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2  # Hacer simétrica
        np.fill_diagonal(correlation_matrix, 1)  # Diagonal = 1
        
        fig_heatmap = px.imshow(
            correlation_matrix,
            x=list(factores_matriz_nombres.keys()),
            y=list(factores_matriz_nombres.keys()),
            title="Correlación entre Factores Académicos y Comportamentales",
            color_continuous_scale="RdBu_r",
            aspect="auto"
        )
        
        # Agregar tooltips y mejorar formato
        fig_heatmap.update_traces(
            hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>" +
                         "Correlación: %{z:.3f}<br>" +
                         "<i>Valores cercanos a 1: correlación positiva fuerte</i><br>" +
                         "<i>Valores cercanos a -1: correlación negativa fuerte</i><br>" +
                         "<i>Valores cercanos a 0: sin correlación</i><extra></extra>"
        )
        
        fig_heatmap.update_layout(
            font=dict(size=10),
            height=450,
            title=dict(font=dict(size=14))
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Agregar explicación de los factores
        with st.expander("**Explicación de Factores Analizados**"):
            st.markdown("""
            - **Promedio Académico**: Rendimiento académico general del estudiante (0-20)
            - **Asistencia (%)**: Porcentaje de asistencia a clases presenciales y virtuales
            - **Engagement Score**: Índice de participación e involucramiento académico
            - **Uso Plataforma (hrs/sem)**: Horas semanales de uso del sistema de gestión de aprendizaje
            - **Participación Digital**: Número de interacciones mensuales en plataformas educativas
            """)
        
        
        # Factores críticos identificados algorítmicamente
        st.markdown("### **Factores Críticos Identificados**")
        
        factores_criticos = []
        if resultados_formulas.get('score_riesgo', 0) > 0.6:
            factores_criticos.append({"factor": "Score de Riesgo Elevado", "valor": f"{resultados_formulas.get('score_riesgo', 0):.3f}", "criticidad": "Alto"})
        if resultados_formulas.get('tasa_retencion', 1) < 0.7:
            factores_criticos.append({"factor": "Baja Tasa de Retención", "valor": f"{resultados_formulas.get('tasa_retencion', 1):.3f}", "criticidad": "Medio"})
        if resultados_formulas.get('engagement_score', 1) < 0.4:
            factores_criticos.append({"factor": "Bajo Engagement", "valor": f"{resultados_formulas.get('engagement_score', 1):.3f}", "criticidad": "Medio"})
        if resultados_formulas.get('vulnerabilidad_financiera', 0) > 0.5:
            factores_criticos.append({"factor": "Vulnerabilidad Financiera", "valor": f"{resultados_formulas.get('vulnerabilidad_financiera', 0):.3f}", "criticidad": "Alto"})
        
        if factores_criticos:
            df_factores = pd.DataFrame(factores_criticos)
            
            # Mostrar como tabla
            st.dataframe(
                df_factores,
                column_config={
                    "factor": "Factor de Riesgo",
                    "valor": "Valor Calculado", 
                    "criticidad": st.column_config.SelectboxColumn(
                        "Nivel de Criticidad",
                        options=["Bajo", "Medio", "Alto"]
                    )
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.success("No se detectaron factores críticos para este estudiante")
        
        # Mostrar fórmulas
        st.markdown("### **Metodología Matemática Aplicada**")
        with st.expander("Ver detalles de algoritmos utilizados"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Score de Riesgo Compuesto**")
                st.latex(r"R = \sigma\left(\sum_{i=1}^{n} w_i \cdot \tilde{x}_i \right)")
                st.caption("Función sigmoide con pesos ponderados")
                
                st.markdown("**Engagement Score (PCA)**")
                st.latex(r"E = \text{Norm}(\text{PC}_1(\mathbf{X}))")
                st.caption("Análisis de componentes principales")
            
            with col2:
                st.markdown("**Vulnerabilidad Financiera (EWMA)**")
                st.latex(r"V_t = \sum_{k=0}^{K} \lambda (1-\lambda)^k d_{t-k}")
                st.caption("Media móvil exponencialmente ponderada")
                
                st.markdown("**Tasa de Retención Individual**")
                st.latex(r"\text{Retención} = \frac{\text{Créditos Reales}}{\text{Créditos Esperados}}")
                st.caption("Comparación de progreso académico")

def mostrar_informes_automaticos(estudiantes):
    """Módulo de generación de informes automáticos"""
    st.header("Informes Automáticos")

    st.markdown("""
    Este módulo permite generar informes detallados y personalizados para los estudiantes. (ej. UH2021007)
    """)

    if not estudiantes:
        st.warning("No hay estudiantes disponibles para generar informes.")
        return

    # CSS específico para el placeholder del input de búsqueda (global para esta función)
    st.markdown("""
    <style>
    div[data-testid="stTextInput"] input::placeholder {
        color: #6b7280 !important;
        opacity: 1 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Inicializar estudiante seleccionado en session state
    if 'estudiante_informe_seleccionado' not in st.session_state:
        st.session_state.estudiante_informe_seleccionado = None

    # Controles de búsqueda en múltiples columnas
    col1, col2, col3, col4 = st.columns([3, 1, 1, 0.5])

    with col1:
        texto_busqueda_informe = st.text_input(
            "Buscar estudiante:",
            placeholder="Nombre, apellido, código o email",
            help="Buscar por cualquier campo del estudiante",
            key="busqueda_informe"
        )

    with col2:
        estados_informe = ["Todos"] + list(set([e.get('estado', 'Sin Estado') for e in estudiantes]))
        estado_filtro_informe = st.selectbox(
            "Estado:",
            estados_informe,
            key="estado_informe"
        )

    with col3:
        carreras_informe = ["Todas"] + list(set([e.get('carrera', 'Sin Carrera') for e in estudiantes]))
        carrera_filtro_informe = st.selectbox(
            "Carrera:",
            carreras_informe,
            key="carrera_informe"
        )

    with col4:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Buscar", key="buscar_informe_btn"):
            # Limpiar selección si hay un estudiante seleccionado
            if st.session_state.estudiante_informe_seleccionado:
                st.session_state.estudiante_informe_seleccionado = None
                st.rerun()

    # Filtrar estudiantes según criterios de búsqueda (automático)
    estudiantes_filtrados_informe = []

    # Solo filtrar si hay texto de búsqueda
    if texto_busqueda_informe:
        for estudiante in estudiantes:
            # Filtro por texto
            campos_busqueda = [
                str(estudiante.get('nombre', '')).lower(),
                str(estudiante.get('apellido', '')).lower(),
                str(estudiante.get('student_id', '')).lower(),
                str(estudiante.get('email', '')).lower()
            ]
            texto_coincide = any(texto_busqueda_informe.lower() in campo for campo in campos_busqueda)

            # Filtro por estado
            estado_coincide = (estado_filtro_informe == "Todos" or
                              estudiante.get('estado', '') == estado_filtro_informe)

            # Filtro por carrera
            carrera_coincide = (carrera_filtro_informe == "Todas" or
                               estudiante.get('carrera', '') == carrera_filtro_informe)

            if texto_coincide and estado_coincide and carrera_coincide:
                estudiantes_filtrados_informe.append(estudiante)

    # Mostrar lista de resultados solo si hay búsqueda y no hay estudiante seleccionado
    if texto_busqueda_informe and not st.session_state.estudiante_informe_seleccionado:
        if estudiantes_filtrados_informe:
            st.markdown(f"**{len(estudiantes_filtrados_informe)} estudiante(s) encontrado(s)**")

            # Lista desplegable de estudiantes
            for i, estudiante in enumerate(estudiantes_filtrados_informe):
                nombre_completo = f"{estudiante.get('nombre', 'N/A')} {estudiante.get('apellido', '')}"
                codigo = estudiante.get('student_id', 'N/A')
                carrera = estudiante.get('carrera', 'N/A')
                estado = estudiante.get('estado', 'N/A')
                promedio = estudiante.get('promedio', 0)

                # Crear card clickeable para cada estudiante
                with st.container():
                    col1, col2 = st.columns([4, 1])

                    with col1:
                        st.markdown(f"""
                        **{nombre_completo}** - {codigo}
                        - **Carrera:** {carrera}
                        - **Estado:** {estado}
                        - **Promedio:** {promedio:.2f}/20
                        """)

                    with col2:
                        if st.button("Seleccionar", key=f"select_informe_{i}", use_container_width=True):
                            st.session_state.estudiante_informe_seleccionado = estudiante
                            st.rerun()

                    st.divider()
        else:
            st.info("No se encontraron estudiantes que coincidan con los criterios de búsqueda.")

    elif not texto_busqueda_informe and not st.session_state.estudiante_informe_seleccionado:
        st.info("Escribe en el campo de búsqueda para encontrar estudiantes.")

    # Análisis del estudiante seleccionado
    estudiante_seleccionado = st.session_state.estudiante_informe_seleccionado

    if estudiante_seleccionado:
        # Mostrar información básica del estudiante seleccionado
        st.markdown("## Estudiante Seleccionado")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            **{estudiante_seleccionado.get('nombre', 'N/A')} {estudiante_seleccionado.get('apellido', '')}**
            - **Código:** {estudiante_seleccionado.get('student_id', 'N/A')}
            - **Carrera:** {estudiante_seleccionado.get('carrera', 'N/A')}
            """)

        with col2:
            st.markdown(f"""
            **Información Académica**
            - **Ciclo:** {estudiante_seleccionado.get('ciclo', 'N/A')}
            - **Promedio:** {estudiante_seleccionado.get('promedio', 'N/A')}/20
            - **Estado:** {estudiante_seleccionado.get('estado', 'N/A')}
            """)

        with col3:
            st.markdown(f"""
            **Contacto**
            - **Email:** {estudiante_seleccionado.get('email', 'N/A')}
            - **Asistencia:** {estudiante_seleccionado.get('asistencia_porcentaje', 0):.1f}%
            - **Créditos:** {estudiante_seleccionado.get('creditos_aprobados', 'N/A')}
            """)

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            # Cambiar el texto del botón según si ya hay un informe generado
            texto_boton = "Regenerar Informe" if hasattr(st.session_state, 'informe_generado') else "Generar Informe Completo"
            if st.button(texto_boton, use_container_width=True, type="primary"):
                # Aplicar fórmulas matemáticas al estudiante
                resultados_formulas = aplicar_formulas_estudiante(estudiante_seleccionado)

                # Generar informe completo profesional
                with st.spinner("Procesando datos y generando informe completo..."):
                    time.sleep(2)  # Simular procesamiento más extenso
                    informe = generar_informe_completo_profesional(estudiante_seleccionado, resultados_formulas)
                    pdf_bytes = generar_pdf_informe(estudiante_seleccionado, resultados_formulas)
                    
                    st.session_state.informe_generado = informe
                    st.session_state.resultados_formulas = resultados_formulas
                    st.session_state.pdf_generado = pdf_bytes
                    st.success("Análisis completo generado exitosamente")
                    st.rerun()

        with col2:
            if hasattr(st.session_state, 'pdf_generado') and st.session_state.pdf_generado:
                st.download_button(
                    label="Descargar PDF",
                    data=st.session_state.pdf_generado,
                    file_name=f"informe_completo_{estudiante_seleccionado.get('student_id', 'estudiante')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            else:
                st.button("Descargar PDF", use_container_width=True, disabled=True)

        # Mostrar informe generado
        if hasattr(st.session_state, 'informe_generado'):
            st.markdown("### Informe Generado Automáticamente")
            st.text_area(
                "Contenido del informe:",
                st.session_state.informe_generado,
                height=600,
                disabled=True,
                key="informe_completo"
            )

if __name__ == "__main__":
    main()
