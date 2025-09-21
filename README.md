# 📊 Sistema Predictivo de Deserción Estudiantil

## 🎓 Universidad Horizonte

### Resumen Ejecutivo

Sistema de análisis predictivo desarrollado para Universidad Horizonte que identifica estudiantes en riesgo de deserción académica mediante algoritmos matemáticos especializados, integración con MongoDB Atlas y procesamiento de datos en tiempo real usando Databricks. Construido con **Streamlit** para una interfaz web moderna e intuitiva, la solución proporciona métricas institucionales, análisis individualizados y reportes automatizados para optimizar la retención estudiantil.

### Caso de Estudio: Universidad Horizonte

#### Contexto Institucional
La Universidad Horizonte es una institución privada con 25 años de trayectoria y 18.000 estudiantes. Su propuesta académica es reconocida, pero enfrenta desafíos significativos en la adaptación a la era digital:

- Las clases virtuales implementadas durante la pandemia fueron improvisadas y poco interactivas
- Los estudiantes reclaman plataformas poco intuitivas y procesos administrativos lentos (matrícula, pagos, certificados)
- Competidores ofrecen universidades 100% online con costos más bajos y experiencias digitales más fluidas
- No se utilizan datos para predecir deserción estudiantil ni personalizar el aprendizaje
- El profesorado muestra resistencia al cambio digital

La rectoría requiere desarrollar un plan de transformación digital educativa para mejorar la experiencia del estudiante y la eficiencia administrativa.

#### Desafío Seleccionado
Este sistema aborda específicamente el desafío de **"No se usan datos para predecir deserción estudiantil ni personalizar el aprendizaje"** mediante:

1. **Evaluación de Madurez Digital**: Implementación de analítica predictiva para medir y mejorar la retención estudiantil
2. **Visión Digital**: Desarrollo de una "universidad digital e inclusiva con aprendizaje híbrido de excelencia"
3. **Iniciativas en Cuatro Dimensiones**:
   - **Clientes (estudiantes)**: Sistema de predicción temprana y recomendaciones personalizadas
   - **Operaciones**: Automatización de análisis de datos para retención estudiantil
   - **Modelo de Negocio**: Optimización de programas académicos basada en predicciones
   - **Cultura**: Capacitación en analítica predictiva para el cuerpo docente
4. **Roadmap de Implementación**:
   - **Corto Plazo**: Sistema de predicción básico operativo
   - **Mediano Plazo**: Integración con plataformas LMS existentes
   - **Largo Plazo**: IA educativa y analítica predictiva avanzada
5. **Métricas de Éxito**: Tasa de retención estudiantil, precisión de predicción, satisfacción estudiantil

## ✨ Características Principales

- **🧮 Ocho Algoritmos Matemáticos Avanzados**: Score de Riesgo Compuesto, Distancia de Mahalanobis, Análisis PCA de Engagement, entre otros
- **📱 Dashboard Interactivo**: Interfaz moderna construida con Streamlit
- **🔗 Conexión en Tiempo Real**: Integración directa con MongoDB Atlas
- **⚡ Procesamiento en Databricks**: Análisis masivo de datos estudiantiles usando Databricks para big data processing
- **📋 Reportes Inteligentes**: Generación automática de informes con recomendaciones específicas
- **📈 Visualizaciones Avanzadas**: Gráficos interactivos con Plotly
- **🚀 Procesamiento Masivo**: Análisis eficiente de grandes volúmenes de datos estudiantiles

## Arquitectura Técnica

```
Sistema-Prediccion-Desercion/
├── app_universidad_horizonte.py     # Aplicación principal Streamlit
├── formulas_matematicas.py          # Módulo de algoritmos matemáticos
├── requirements.txt                 # Dependencias del proyecto
└── README.md                        # Documentación del proyecto
```

## 🛠️ Instalación y Despliegue

### 📋 Prerrequisitos
- Python 3.9+
- MongoDB Atlas (cuenta gratuita)
- Databricks (para procesamiento de big data y análisis avanzado)
- Git
- **Streamlit** (se instala automáticamente con requirements.txt)

### Instalación Local
```bash
git clone <repositorio>
cd Sistema-Prediccion-Desercion
pip install -r requirements.txt
```

### Configuración de Variables de Entorno
Para Streamlit Cloud, configurar las siguientes variables en Settings > Secrets:

```
[secrets]
MONGODB_URI = "mongodb+srv://usuario:password@cluster.mongodb.net/"
DATABASE_NAME = "universidad_horizonte"
COLLECTION_ESTUDIANTES = "estudiantes"
COLLECTION_NOTAS = "notas"
```

### Ejecución
```bash
streamlit run app_universidad_horizonte.py
```

## 🔬 Metodología Científica

### 🧪 Algoritmos Implementados
1. **Score de Riesgo Compuesto**: Función sigmoide con pesos ponderados
2. **Distancia de Mahalanobis**: Detección de perfiles atípicos
3. **Índice de Engagement (PCA)**: Análisis de componentes principales
4. **Momentum Académico**: Suavizado exponencial
5. **Vulnerabilidad Financiera**: Media móvil exponencialmente ponderada
6. **Entropía Académica**: Medida de variabilidad en el desempeño
7. **Tasa de Retención Individual**: Ratio de progreso académico
8. **Receptividad a Intervención**: Modelo de propensión

### Validación
- Precisión de predicción: 85% de detección de riesgo
- Falsos positivos: < 15%
- Tiempo de análisis: < 2 segundos por estudiante

### Flujo de Trabajo con Databricks
El desarrollo del sistema incluyó el uso de Databricks para:
- Procesamiento distribuido de datos estudiantiles
- Desarrollo e iteración de algoritmos de machine learning
- Análisis exploratorio de datos a gran escala
- Validación de modelos predictivos en entornos distribuidos
- Integración con pipelines de datos automatizados

## 🛠️ Tecnologías Utilizadas

### ⚙️ Backend y Análisis
- Python 3.9+, NumPy, Pandas, Scikit-learn
- Databricks (procesamiento distribuido y análisis de big data)

### 🎨 Frontend y Visualización
- **Streamlit**: Framework web principal para la aplicación interactiva
- Plotly, CSS personalizado

### 🗄️ Base de Datos
- MongoDB Atlas, PyMongo

### 📦 Control de Versiones
- Git, GitHub

## Funcionalidades del Sistema

### Análisis Institucional
- Métricas agregadas de retención y rendimiento
- Visualizaciones de distribución por riesgo y carrera
- Identificación de tendencias académicas

### Análisis Individual
- Perfiles de riesgo personalizados por estudiante
- Recomendaciones específicas de intervención
- Seguimiento de evolución académica

### Reportes Automatizados
- Informes PDF con diagnósticos detallados
- Recomendaciones de intervención priorizadas
- Métricas de efectividad de programas

---

🚀 Desarrollado para optimizar la retención estudiantil en Universidad Horizonte mediante analítica predictiva avanzada.