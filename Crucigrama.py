import random
import pandas as pd
import streamlit as st
import numpy as np
import time

from unidecode import unidecode

# Configurar el ancho de la página
st.set_page_config(layout="wide")

Vocabulario = {
"Algoritmo" : "Es un conjunto finito de pasos organizados y secuenciales diseñados para resolver un problema o realizar una tarea específica. En el contexto de la inteligencia artificial, los algoritmos son fundamentales, ya que representan la base de cualquier sistema inteligente. Por ejemplo, los algoritmos de aprendizaje automático permiten a las máquinas identificar patrones en los datos.",
"Datos" : "La base de todo sistema de inteligencia artificial. Los datos, en forma de texto, imágenes o números, son procesados por algoritmos para entrenar modelos. Su calidad y cantidad influyen directamente en los resultados.",
"Modelo" : "Un modelo es una representación matemática que encapsula patrones aprendidos por un sistema de IA. Puede ser usado para realizar tareas como predicción, clasificación o generación de contenido.",
"Entrenar" : "Es el proceso mediante el cual un modelo ajusta sus parámetros para aprender de los datos. El objetivo es minimizar el error entre las predicciones del modelo y los valores reales.",
"Predicción" : "Es la capacidad de un modelo para proporcionar resultados basados en datos nuevos. Por ejemplo, un modelo puede predecir si un correo electrónico es spam.",

"Regresión" : "Un tipo de análisis utilizado para predecir valores continuos, como precios de vivienda o temperaturas. En IA, se aplica para modelar relaciones entre variables.",

"Cluster" : "Proceso de agrupar datos similares en conjuntos. Es clave en tareas no supervisadas.",

"Batch" : "Porción de datos procesada en una iteración durante el entrenamiento. Optimiza el uso de recursos computacionales.",

"Epoch" : "Un ciclo completo a través de un conjunto de datos durante el entrenamiento.",

"Gradio" : "El gradiente es una herramienta matemática utilizada para calcular la dirección del cambio más pronunciado en una función. En el aprendizaje automático, se emplea en algoritmos de optimización como el gradiente descendente, que ajusta los parámetros del modelo para minimizar la función de pérdida.",

"DeepNet" : "Redes neuronales profundas que contienen múltiples capas ocultas. Estas redes son el núcleo del aprendizaje profundo (deep learning) y son utilizadas para resolver problemas complejos como la visión por computadora, el reconocimiento de voz y la traducción automática.",

"Loss" : "La función de pérdida (loss) mide qué tan lejos están las predicciones de un modelo respecto a los valores reales. Minimizar esta métrica es crucial para que el modelo sea efectivo.",

"GPU" : "Las Unidades de Procesamiento Gráfico son piezas de hardware diseñadas para cálculos paralelos masivos, esenciales en tareas de IA como el entrenamiento de redes neuronales profundas debido a su alta capacidad de procesamiento.",

"Dropout" : "Técnica de regularización que desactiva aleatoriamente neuronas durante el entrenamiento, evitando que el modelo dependa demasiado de características específicas y ayudando a prevenir el overfitting.",

"Sparse" : "Se refiere a matrices dispersas, donde la mayoría de los valores son cero. Este concepto es clave en técnicas como la representación dispersa en NLP.",

"Embeds" : "Son representaciones vectoriales densas de datos, usadas para capturar características semánticas, como en palabras, imágenes o nodos de grafos.",

"AutoML" : "Automated Machine Learning (AutoML) automatiza el proceso de creación de modelos de aprendizaje automático. Permite a los usuarios seleccionar automáticamente algoritmos, ajustar hiperparámetros y realizar validaciones cruzadas, simplificando el trabajo para quienes no tienen experiencia en IA.",

"Kernel" : "En aprendizaje automático, el kernel es una función utilizada para transformar datos en un espacio de mayor dimensión. Es esencial en métodos como las Máquinas de Soporte Vectorial (SVM) para resolver problemas no lineales al trazar límites más complejos entre clases.",

"Reward" : "La recompensa es una señal de retroalimentación que un agente en aprendizaje por refuerzo recibe después de realizar una acción. Su objetivo es guiar al agente hacia comportamientos deseados en su entorno.",

"Vector" : "Los vectores son estructuras matemáticas que representan magnitudes y direcciones. En IA, se usan para representar datos, como palabras en procesamiento del lenguaje natural o píxeles en imágenes.",

"Bias" : "El bias (sesgo) tiene dos significados en IA. En redes neuronales, es un término añadido a las ecuaciones para ajustar resultados. En datos, se refiere a prejuicios presentes en los conjuntos de datos que pueden afectar la imparcialidad de los modelos.",

"Noise" : "El ruido es información irrelevante o aleatoria presente en los datos, que puede dificultar el aprendizaje de un modelo. Reducir el ruido es crucial para mejorar el rendimiento.",

"Entropy" : "La entropía mide el grado de incertidumbre o desorden en un sistema. En IA, se usa en algoritmos como el Árbol de Decisión para dividir datos de manera eficiente y en aprendizaje por refuerzo para medir exploración frente a explotación.",

"Python" : "Un lenguaje de programación ampliamente utilizado en IA debido a su simplicidad y sus bibliotecas, como TensorFlow, PyTorch y Scikit,-learn, que facilitan el desarrollo de modelos y análisis de datos.",

"Perceptrón" : "Modelo básico de red neuronal que utiliza una sola capa de nodos. Es uno de los primeros algoritmos de aprendizaje automático desarrollado y forma la base de redes más complejas.",

"Nómina" : "En redes neuronales, se refiere al número total de nodos o unidades en una capa. Determina la capacidad de la red para aprender patrones complejos.",

"Capa" : "En redes neuronales, las capas son los niveles que procesan los datos. Pueden ser de entrada, ocultas o de salida, y cada una transforma los datos de manera específica.",

"Solver" : "Algoritmo utilizado para resolver problemas de optimización en IA, como ajustar parámetros de un modelo. Ejemplos comunes son Adam, SGD y RMSProp.",

"Prueba" : "El testeo evalúa el rendimiento de un modelo utilizando un conjunto de datos nunca antes visto. Es crucial para determinar si un modelo generaliza bien.",

"Vectorial" : "Representación matemática en forma de vectores. En IA, las palabras, imágenes y datos se transforman en vectores para facilitar cálculos y aprendizaje.",

"Gradiente" : "La dirección de mayor cambio en una función. Es utilizado en optimización para ajustar parámetros de modelos de aprendizaje automático.",

"Escalar" : "Proceso de ajustar valores de los datos a un rango específico, mejorando el rendimiento de algoritmos que son sensibles a la magnitud de las variables.",

"MinMax" : "Método de escalado que transforma datos a un rango entre 0 y 1. Es útil en modelos sensibles a la escala de las características.",

"Capacitor" : "En hardware para IA, los capacitores almacenan energía temporalmente, asegurando un flujo constante de energía durante cálculos intensivos.",

"Denoiser" : "Modelo diseñado para eliminar ruido de datos, mejorando la calidad de las entradas. Por ejemplo, un autoencoder puede actuar como denoiser.",

"Patrón" : "En IA, los patrones son estructuras repetitivas en los datos que los modelos detectan para hacer predicciones o clasificaciones.",

"Ganancia" : "Métrica utilizada en árboles de decisión para medir qué atributo divide mejor los datos, ayudando a crear nodos más efectivos.",

"Vector" : "Representación numérica en múltiples dimensiones. En procesamiento de lenguaje natural, las palabras se convierten en vectores para capturar su significado.",

"Librería" : "Conjunto de funciones predefinidas que facilitan el desarrollo de aplicaciones de IA, como NumPy, Pandas o PyTorch.",

"Híbrido" : "Sistemas que combinan múltiples enfoques de IA, como aprendizaje supervisado y no supervisado, para resolver problemas complejos.",

"Árboles" : "Estructuras que organizan decisiones en niveles jerárquicos. Cada división representa una pregunta y cada respuesta dirige a una nueva rama o a un resultado final.",

"Pesos" : "Valores ajustables que determinan la importancia de una conexión dentro de un sistema que procesa información. Se optimizan durante el entrenamiento para mejorar el rendimiento.",

"Lote" : "Porción de datos procesada en una iteración, utilizada para equilibrar la carga de trabajo y optimizar los recursos durante el ajuste de parámetros.",

"Búsqueda" : "Proceso para encontrar valores óptimos dentro de un espacio definido, como seleccionar los mejores parámetros para un modelo.",

"Nodo" : "Elemento básico dentro de una estructura que procesa información. Recibe datos, realiza cálculos y transmite los resultados a otros componentes.",

"Salida" : "Resultado generado después de procesar información. Puede ser una clasificación, predicción o una recomendación, dependiendo de la tarea.",

"Vector" : "Estructura matemática que contiene magnitudes ordenadas en un espacio definido, utilizada para representar características de datos.",

"Normal" : "Técnica que ajusta los valores para que tengan una distribución uniforme o estándar, mejorando la estabilidad de los cálculos posteriores.",

"Delta" : "Diferencia calculada entre el resultado esperado y el obtenido, empleada para ajustar parámetros y mejorar el rendimiento.",

"Sparse" : "Estructura con elementos mayormente nulos o sin valores significativos, optimizada para reducir el espacio y mejorar la eficiencia.",

"Factor" : "Elemento que contribuye al comportamiento o rendimiento de un sistema. Puede ser ajustado o combinado con otros para obtener mejores resultados.",

"Plano" : "Representación geométrica que separa o clasifica conjuntos de datos en diferentes categorías dentro de un espacio multidimensional.",

"Ruido" : "Información irrelevante o aleatoria que dificulta el análisis o la identificación de patrones en los datos.",

"Salida" : "El valor o conjunto de valores que resulta del procesamiento de información, utilizado como base para decisiones o predicciones.",

"Label" : "Identificador asignado a una entrada, que sirve como referencia para entrenar y evaluar el rendimiento de un sistema.",

"Ajuste" : "Proceso para encontrar los valores más adecuados que mejoren el rendimiento general de un sistema, minimizando errores.",

"Filtro" : "Herramienta que selecciona características relevantes y elimina información redundante o innecesaria.",

"Kernel" : "Función que transforma los datos en un espacio diferente, permitiendo encontrar relaciones que no son visibles en su forma original.",

"Barrera" : "Límite o separación que organiza datos en grupos distintos, facilitando la clasificación o predicción.",

"Salida" : "El conjunto de valores finales generados por un proceso. Es el resultado que se utiliza para tomar decisiones, ya sea una predicción, una clasificación o una acción específica, dependiendo de la tarea en cuestión.",

"Entradas" : "Datos iniciales que alimentan un proceso. Estas pueden ser numéricas, categóricas, imágenes, texto o combinaciones de ellas, dependiendo del problema que se busca resolver.",

"Parámetro" : "Valor ajustable que define el comportamiento de un sistema. Se optimiza durante el entrenamiento para que el modelo aprenda a realizar predicciones o tareas específicas con mayor precisión.",

"Recurso" : "Cualquier elemento utilizado para realizar un cálculo o entrenamiento. Puede ser hardware como procesadores o memoria, o software como bibliotecas y herramientas.",

"Vector" : "Representación numérica en forma de lista ordenada. Sirve para expresar relaciones o características en un espacio geométrico, facilitando cálculos y comparaciones.",

"Criterio" : "Regla que guía un proceso de decisión. Se utiliza para evaluar qué tan efectivamente un sistema logra los objetivos definidos en la tarea.",

"Límite" : "Separación entre diferentes grupos de datos. Representa una frontera que ayuda a clasificar o distinguir patrones de acuerdo con las características observadas.",

"Error" : "Diferencia entre un valor real y uno calculado. Este valor se utiliza como retroalimentación para mejorar el rendimiento ajustando los componentes involucrados.",

"Conjunto" : "Agrupación de datos organizados para una tarea específica, como entrenamiento, validación o prueba. Puede contener características y etiquetas asociadas.",

"Estructura" : "Organización o disposición de los elementos que conforman un sistema. Influye en su capacidad para procesar información y adaptarse a los problemas.",

"Ruta" : "Camino que siguen los datos o cálculos a través de un sistema, desde las entradas iniciales hasta los resultados finales.",

"Propiedad" : "Cualquier atributo medible o característica que describe un objeto o dato. Se usa para identificar patrones o hacer predicciones.",

"Capas" : "Niveles organizados de procesamiento que transforman los datos de una forma a otra. En cada nivel, se extraen características específicas para alcanzar un resultado final.",

"Nodo" : "Unidad fundamental en una estructura que realiza cálculos individuales. Cada uno recibe información, la procesa y envía el resultado a otros elementos conectados.",

"Bias" : "Valor que se suma a las entradas para ajustar resultados, mejorando la capacidad de la estructura para modelar relaciones complejas.",

"Entrenar" : "Proceso de ajuste iterativo mediante el cual un sistema aprende a realizar tareas específicas al minimizar los errores cometidos sobre datos proporcionados.",

"Perdida" : "Medida de qué tan lejos está el resultado de un sistema de un objetivo deseado. Se utiliza para evaluar el rendimiento y ajustar parámetros.",

"Agrupar" : "Tarea de organizar elementos en grupos basados en similitudes. Se utiliza para analizar datos y encontrar patrones no obvios.",

"Separar" : "Proceso de dividir datos en diferentes categorías o regiones, basado en reglas que distinguen características específicas.",

"Segmento" : "Porción de datos seleccionada para realizar un análisis específico. Puede dividirse de acuerdo con criterios como temporalidad, ubicación o características.",

"Filtro" : "Mecanismo para seleccionar elementos relevantes y descartar información que no contribuye al objetivo del análisis.",

"Normal" : "Proceso de ajustar valores a un rango común para que tengan la misma escala, mejorando el rendimiento en cálculos y comparaciones.",

"Ruido" : "Información irrelevante o aleatoria que puede interferir en la capacidad de un sistema para encontrar patrones en los datos.",

"Salida" : "Resultado procesado que se genera al final de un cálculo. Representa la conclusión de todo el análisis o entrenamiento.",

"Cluster" : "Agrupación de elementos similares que no están etiquetados previamente. Se utiliza para encontrar relaciones dentro de los datos.",

"Desvío" : "Medida que refleja qué tan dispersos están los datos respecto a un valor central. Es útil para identificar anomalías o patrones atípicos.",

"Redondeo" : "Ajuste de valores numéricos a una precisión definida. Se emplea para simplificar cálculos o resultados finales.",

"Barrera" : "Frontera que separa diferentes regiones en un espacio de características, ayudando a distinguir entre grupos o categorías.",

"Época" : "Una pasada completa por todo el conjunto de datos durante el entrenamiento. Cada iteración ajusta parámetros para mejorar el rendimiento del sistema en las siguientes pasadas.",

"Datos" : "Información en bruto que se utiliza para entrenar, validar y probar sistemas. Pueden incluir números, texto, imágenes, sonidos o combinaciones de estos formatos.",

"Clases" : "Categorías en las que se agrupan los datos. Estas sirven como etiquetas para que un sistema pueda aprender a identificar y clasificar correctamente nuevos ejemplos.",

"Base" : "Conjunto estructurado de información organizada en tablas o archivos. Es fundamental para almacenar y acceder a grandes volúmenes de datos de forma eficiente.",

"Modelo" : "Representación matemática o computacional de un problema. Aprende patrones a partir de datos y utiliza ese conocimiento para realizar predicciones o decisiones.",

"Regla" : "Instrucción lógica que guía las decisiones. Puede derivarse de datos o establecerse manualmente para clasificar, predecir o procesar información.",

"Fase" : "Una etapa específica dentro de un proceso mayor. Puede referirse al entrenamiento, la validación o la prueba, entre otras partes del desarrollo.",

"Rejilla" : "Técnica de búsqueda en la que se exploran combinaciones de valores posibles para encontrar la configuración óptima de un sistema.",

"Umbral" : "Valor límite que determina la activación de una acción o decisión. Ayuda a distinguir entre resultados aceptables y no aceptables.",

"Cluster" : "Agrupación de datos con características similares. Se utiliza para explorar relaciones ocultas y para organizar datos no etiquetados.",

"Validez" : "Medida de qué tan bien un sistema cumple con los objetivos esperados. Se evalúa mediante pruebas independientes del conjunto de entrenamiento.",

"Huella" : "Representación compacta de características clave en un conjunto de datos, utilizada para identificar patrones únicos o resumir información.",

"Salidas" : "Valores generados después de procesar las entradas. Representan el objetivo del cálculo, ya sea una predicción, recomendación o acción.",

"Patrón" : "Estructura repetitiva detectada en los datos. Es la base para que un sistema aprenda y generalice a nuevos ejemplos.",

"Canales" : "Flujos de datos independientes que se procesan simultáneamente. En imágenes, cada canal puede representar un color o una característica específica.",

"Curvas" : "Representaciones gráficas de la relación entre variables. Ayudan a visualizar el rendimiento o los resultados de un modelo.",

"Algoritmo" : "Conjunto de pasos definidos para realizar una tarea específica. Es la base de cualquier proceso automatizado.",

"Nodo" : "Unidad dentro de una estructura que realiza cálculos o almacena información. Puede conectarse con otros para formar sistemas complejos.",

"Embudo" : "Proceso de reducción progresiva de datos para enfocarse en los elementos más relevantes o importantes para la tarea en cuestión.",

"Ponderar" : "Proceso de asignar diferentes niveles de importancia a las entradas o características según su relevancia en el resultado final.",

"Grados" : "Medida de cambio en parámetros o valores utilizados para ajustar el comportamiento de un sistema y optimizar su rendimiento.",

"Signos" : "Indicadores que muestran tendencias o comportamientos en los datos. Son útiles para anticipar resultados o identificar problemas.",

"Residuos" : "Diferencias entre los valores reales y los estimados por un modelo. Ayudan a evaluar errores y mejorar el sistema.",

"Matrices" : "Tablas bidimensionales de valores numéricos que representan relaciones entre datos. Son esenciales en cálculos de estructuras avanzadas.",

"Variables" : "Elementos cuyo valor puede cambiar dentro de un modelo. Pueden ser entradas, salidas o parámetros que definen un comportamiento.",

"Corte" : "Punto en el que un conjunto de datos se divide en categorías o grupos. Es clave para clasificaciones y agrupamientos.",

"Desfase" : "Diferencia temporal o en magnitud entre dos elementos. Puede afectar la precisión de los cálculos y necesita ser ajustado.",

"Perfil" : "Conjunto de características que describen un grupo o un individuo. Es útil para personalización y análisis de comportamiento.",

"Entreno" : "Conjunto de pasos que ajustan parámetros para que un sistema sea capaz de realizar tareas específicas con alta precisión.",

"Escalas" : "Ajustes que convierten valores en un rango definido. Mejoran la comparabilidad y estabilidad de cálculos matemáticos.",

"Estado" : "Representa la condición actual de un sistema, usualmente en un momento específico dentro de un proceso o durante la ejecución de un modelo. Permite guardar y recuperar configuraciones o valores.",

"Peso" : "Un valor que amplifica o reduce la influencia de una entrada. Es ajustado durante el entrenamiento para mejorar la precisión del sistema en tareas específicas.",

"Bucle" : "Estructura que permite repetir un conjunto de operaciones hasta cumplir una condición, usada para iterar cálculos o entrenar modelos en múltiples pasos.",

"Nivel" : "Una etapa o capa dentro de un sistema jerárquico. Cada uno puede representar una transformación, extracción de características o procesamiento específico.",

"Red" : "Conjunto interconectado de nodos o elementos que colaboran para procesar información, resolver problemas o clasificar datos.",

"Margen" : "Espacio entre los límites de clasificación de un sistema y los datos procesados. Ayuda a medir la confianza en las decisiones tomadas.",

"Columna" : "Conjunto de valores asociados a una característica específica dentro de un conjunto de datos tabular. Representa atributos individuales de las entradas.",

"Rango" : "Extensión entre los valores mínimos y máximos de un conjunto de datos. Se utiliza para entender su distribución y realizar ajustes.",

"Token" : "Unidad mínima de información procesada, como una palabra, carácter o subpalabra en tareas de lenguaje natural.",

"Meta" : "Objetivo o criterio final que guía el desarrollo y evaluación de un modelo o sistema. Puede ser maximizar la precisión, minimizar el error o ambos.",

"Sesgo" : "Preferencia inherente en un sistema hacia ciertos valores o resultados, a menudo debido a desequilibrios en los datos de entrenamiento.",

"Tarea" : "Problema específico que un sistema busca resolver, como clasificar imágenes, generar texto o realizar predicciones numéricas.",

"Marco" : "Conjunto de herramientas o bibliotecas diseñadas para facilitar el desarrollo de modelos. Proveen funcionalidades prediseñadas para tareas comunes.",

"Clase" : "Etiqueta o categoría asignada a un dato. Se utiliza para diferenciar y agrupar elementos según características comunes.",

"Valor" : "Número o dato asociado a una característica, parámetro o salida. Representa la magnitud de un atributo o el resultado de un cálculo.",

"Ancho" : "Dimensión que indica la cantidad de características o nodos en una capa de una estructura. Define la capacidad de procesamiento.",

"Perdida" : "Métrica que evalúa qué tan lejos está el sistema de alcanzar su objetivo. Ayuda a identificar errores y optimizar el rendimiento.",

"Testeo" : "Proceso que evalúa el rendimiento de un sistema utilizando un conjunto de datos independiente del usado para su entrenamiento.",

"Canal" : "Ruta o flujo de datos individuales dentro de una representación. Por ejemplo, en imágenes, puede ser un componente de color como rojo, verde o azul.",

"Tamaño" : "Cantidad de datos procesados en una sola iteración o entrenamiento. Influye en la eficiencia y el tiempo necesario para completar una tarea.",

"Filtro" : "Operación que selecciona elementos relevantes de un conjunto de datos y descarta información redundante o irrelevante.",

"Corte" : "División que separa los datos en categorías distintas o en conjuntos de entrenamiento, validación y prueba.",

"Sigma" : "Término que representa la desviación estándar en estadística. Se usa para medir la dispersión de un conjunto de datos.",

"Margen" : "Diferencia o distancia entre puntos críticos en un espacio de características, útil para optimizar modelos de clasificación.",

"Nodo" : "Unidad básica de cálculo que recibe información, la transforma y la transmite a otros componentes en sistemas más grandes.",

"Mapa" : "Representación visual de datos o características en una estructura organizada, como una matriz de activaciones o una proyección de variables.",

"Repetir" : "Realizar un conjunto de acciones varias veces para refinar resultados o explorar todas las combinaciones posibles en un proceso.",

"Escalar" : "Proceso de ajustar valores para que estén en un rango común, mejorando la estabilidad de los cálculos y la interpretación de resultados.",

"Salida" : "El conjunto de resultados finales después de procesar datos. Puede incluir clasificaciones, predicciones o transformaciones específicas.",

"Prueba" : "Etapa final en el desarrollo donde se evalúa un modelo con datos no vistos para medir su capacidad de generalización y utilidad en problemas reales.",

"Overfit" : "Fenómeno donde un modelo aprende demasiado bien los detalles específicos del conjunto de entrenamiento, perdiendo la capacidad de generalizar.",

"Underfit" : "Situación en la que un modelo no logra capturar patrones suficientes en los datos, resultando en un rendimiento pobre tanto en entrenamiento como en prueba.",

"Callback" : "Función personalizada que se ejecuta durante el entrenamiento de un modelo para realizar tareas específicas, como guardar puntos de control o ajustar parámetros dinámicamente.",

"Tokenizer" : "Herramienta que divide texto en unidades más pequeñas, como palabras, subpalabras o caracteres, para su análisis o procesamiento.",

"Embedding" : "Representación densa de datos en un espacio de menor dimensión que preserva relaciones significativas entre los elementos.",

"Pipeline" : "Serie de pasos organizados para procesar datos y entrenar modelos de manera secuencial y reproducible.",

"Optimizer" : "Algoritmo que ajusta los parámetros del modelo para minimizar la pérdida y mejorar su rendimiento.",

"Gradient" : "Vector que indica la dirección y magnitud de cambio en los parámetros, usado para optimizar modelos mediante descenso de gradiente.",

"Momentum" : "Técnica que acelera el entrenamiento al tener en cuenta las actualizaciones previas de los parámetros para suavizar la convergencia.",

"Activation" : "Función que introduce no linealidad en un modelo, permitiendo que aprenda relaciones más complejas entre los datos.",

"Regularizer" : "Método que evita el sobreajuste añadiendo penalizaciones al cálculo de la pérdida, limitando la complejidad del modelo.",

"Batching" : "Estrategia de dividir datos en pequeños subconjuntos para procesarlos en paralelo, mejorando la eficiencia computacional.",

"GradientClip" : "Técnica para limitar la magnitud de los gradientes durante el entrenamiento, evitando inestabilidad numérica.",

"Augment" : "Proceso de generar nuevas muestras de datos aplicando transformaciones a los originales, aumentando la diversidad y el tamaño del conjunto de entrenamiento.",

"Feature" : "Atributo o característica que describe un dato y que se utiliza para construir modelos predictivos.",

"Sampling" : "Selección de un subconjunto representativo de datos para reducir el tamaño del problema sin perder información esencial.",

"Outlier" : "Dato que se aleja significativamente de los patrones generales del conjunto, indicando posibles errores o variaciones atípicas.",

"Hyperopt" : "Técnica para encontrar los valores óptimos de hiperparámetros en un modelo, usualmente mediante búsqueda en cuadrícula o aleatoria.",

"Decoder" : "Componente que transforma representaciones internas en salidas interpretables, como texto o imágenes.",

"Encoder" : "Parte del modelo que transforma datos en representaciones más compactas y útiles para el análisis o la predicción.",

"Latent" : "Espacio abstracto donde los datos son representados como vectores densos durante su procesamiento.",

"Masking" : "Técnica que oculta ciertas partes de los datos para enfocarse en elementos relevantes o reducir ruido durante el entrenamiento.",

"Shuffling" : "Proceso de mezclar datos de manera aleatoria antes de procesarlos, garantizando que los modelos no aprendan patrones indeseados.",

"OneHot" : "Método para representar categorías como vectores binarios en los que solo un valor es activado.",

"Overhead" : "Carga adicional de computación o memoria necesaria para realizar operaciones más complejas.",

"Scaling" : "Ajuste de las magnitudes de los datos a un rango común para evitar que características dominantes influyan demasiado en los cálculos.",

"ReLU" : "Función que activa únicamente valores positivos y establece en cero los negativos, simplificando cálculos en redes profundas.",

"Pruning" : "Reducción de la complejidad de un modelo eliminando componentes redundantes o poco significativos.",

"Drop" : "Técnica que descarta temporalmente ciertas conexiones durante el entrenamiento para mejorar la robustez del modelo.",

"Checkpoint" : "Punto de guardado en el entrenamiento de un modelo que permite reiniciar el proceso desde un estado intermedio en lugar de comenzar de cero.",

"Simulación" : "Proceso en el cual se modelan sistemas o fenómenos del mundo real para observar cómo responden bajo diferentes condiciones. A menudo se utiliza en la prueba de modelos, con el fin de replicar situaciones que son difíciles o costosas de reproducir en la vida real.",

"Predicción" : "Es el acto de estimar un resultado o valor futuro basado en datos previos. En muchos contextos, las predicciones son fundamentales para la toma de decisiones, donde el modelo aprovecha patrones previos para extrapolar futuros comportamientos o valores.",

"Secuencia" : "Conjunto de datos organizados en un orden específico que debe ser procesado de manera que se respete su estructura temporal o de dependencia. Se utiliza en tareas como la traducción automática o la generación de texto.",

"Distribución" : "Describe cómo se dispersan los datos a través de un espacio de características. Es crucial para entender la variabilidad dentro de un conjunto de datos y ajustar modelos para que se adapten correctamente a la naturaleza de estos datos.",

"Clasificador" : "Es un sistema que asigna etiquetas a los elementos en función de las características extraídas de los datos. El modelo se entrena utilizando ejemplos etiquetados y luego puede clasificar nuevos ejemplos según patrones aprendidos.",

"Generalización" : "La habilidad de un modelo para realizar predicciones precisas sobre datos no vistos previamente. Es crucial para evitar el sobreajuste y garantizar que el modelo pueda adaptarse a situaciones del mundo real.",

"Desviación" : "Mide cuán dispersos están los datos con respecto a la media o a un valor central. Esta métrica es esencial para entender la variabilidad dentro de un conjunto de datos y para la evaluación de modelos.",

"Convergencia" : "Es el proceso mediante el cual los parámetros de un modelo se ajustan progresivamente durante el entrenamiento, hasta alcanzar un valor óptimo donde ya no se producen mejoras significativas.",

"Ponderación" : "Es el proceso de asignar diferentes grados de importancia a las características de los datos, en función de su relevancia para una tarea particular. A través de la ponderación, un modelo puede aprender cuáles son los factores más influyentes.",

"Transformación" : "Proceso mediante el cual los datos originales se modifican para que sean más útiles para el entrenamiento del modelo. Las transformaciones pueden incluir la normalización, la reducción de dimensionalidad o la conversión de variables categóricas.",

"Identificación" : "Es la tarea de reconocer o clasificar patrones o características dentro de los datos que permitan asignarles una categoría o etiqueta. Se utiliza en tareas como el reconocimiento de voz o la clasificación de imágenes.",

"Simetría" : "Característica que refleja la invariancia de un sistema o modelo ante ciertos cambios. Es útil en la detección de patrones y en la simplificación de los cálculos en modelos complejos.",

"Homogeneidad" : "Describe la uniformidad de los datos dentro de un conjunto. En modelos de clasificación, los datos homogéneos tienden a mejorar la precisión y eficiencia de los modelos.",

"Ensemble" : "Método que combina múltiples modelos para mejorar la precisión y robustez del sistema. En lugar de depender de un solo modelo, el enfoque en ensamblaje busca la complementariedad de los modelos individuales.",

"Bipartición" : "Es una técnica en la que los datos se dividen en dos grupos, frecuentemente utilizada en problemas de clasificación binaria, donde se desea separar dos categorías de manera clara y precisa.",

"Reducción" : "Proceso de disminuir la cantidad de datos o características sin perder información crítica. Es importante para mejorar la eficiencia de los modelos y evitar el sobreajuste.",

"Reforzamiento" : "Es un enfoque en el cual los modelos aprenden a través de recompensas y penalizaciones, optimizando sus decisiones en función de las recompensas obtenidas por sus acciones. Esta técnica es clave en la automatización de procesos de toma de decisiones.",

"Umbral" : "Es el valor que define la frontera entre diferentes categorías o resultados.",

"Combinación" : "Proceso de unir múltiples elementos o resultados en un solo resultado. En los modelos, se pueden combinar las predicciones de varios modelos para mejorar la precisión final.",

"Normalización" : "Es el proceso de ajustar los datos para que sigan una escala estándar, como la transformación de las características para que tengan media cero y varianza unitaria. Es esencial para mejorar la convergencia de algunos algoritmos.",

"Densidad" : "Hace referencia a la distribución de los valores dentro de un espacio de características.",

"Escalabilidad" : "La capacidad de un sistema para manejar un aumento en la carga de trabajo sin afectar su rendimiento. Es crucial cuando se trabaja con grandes volúmenes de datos o en entornos de producción.",

"Cohorte" : "Grupo de elementos que comparten características o comportamientos similares. En análisis predictivo, las cohortes ayudan a segmentar datos y mejorar las predicciones.",

"Interfaz" : "Conjunto de herramientas y métodos que permiten la interacción entre diferentes componentes de un sistema. Las interfaces facilitan la integración de nuevas funciones o modelos dentro de una arquitectura existente.",

"Diferenciación" : "Proceso en el cual se mide el cambio en los resultados de un modelo con respecto a pequeñas variaciones en los datos de entrada. Es útil en el ajuste fino de modelos para obtener la mayor precisión posible.",

"Veracidad" : "La medida en que los datos o los resultados de un modelo son precisos y consistentes. Es fundamental para asegurar que los modelos basados en estos datos generen conclusiones fiables y útiles.",

"Rango" : "Es la diferencia entre el valor máximo y el mínimo de un conjunto de datos. Es útil para medir la dispersión y la variabilidad de las características dentro de un conjunto de entrenamiento o test.",

"Híbrido" : "Hace referencia a sistemas que combinan diferentes métodos o técnicas de aprendizaje para mejorar el rendimiento. Los modelos híbridos buscan fusionar las fortalezas de varios enfoques para obtener mejores resultados.",

"Nodo" : "En estructuras como redes neuronales o árboles, un nodo representa un punto de conexión que agrupa o distribuye información. En redes, cada nodo es esencial para la propagación de señales.",

"Segmento" : "Conjunto de datos que comparte ciertas características o que ha sido extraído o clasificado de una manera específica. Se utiliza en tareas de análisis de datos para separar diferentes partes del conjunto de información.",

"Bucle" : "Es una estructura de control que repite un conjunto de instrucciones hasta que se cumpla una condición determinada. Es fundamental en la iteración durante el entrenamiento de modelos.",

"Redondeo" : "Operación matemática que aproxima un número al valor más cercano en una escala establecida.",

"Escala" : "Es el proceso de ajustar los datos a un rango específico para estandarizar o normalizar la magnitud de las características.",

"Modelo" : "Representación matemática o algorítmica de un sistema que captura patrones y relaciones entre datos. Los modelos son el corazón de las predicciones y análisis en diversas disciplinas.",

"Tasa" : "Se refiere a la velocidad o frecuencia con la que ocurren ciertos eventos o cambios en un proceso. En el entrenamiento de modelos, se suele asociar con la ____ de aprendizaje, que determina cuán rápidamente el modelo ajusta sus parámetros.",

"Curva" : "Representación visual de una función o relación entre variables. Las curvas permiten observar de manera clara los comportamientos y tendencias en un conjunto de datos.",

"Tendencia" : "Dirección general en la que evolucionan los datos a lo largo del tiempo. Identificar tendencias es crucial para la predicción y la toma de decisiones basada en datos históricos.",

"Comando" : "Instrucción que se da a un sistema o programa para ejecutar una acción específica. Los comandos en los algoritmos y programas permiten realizar operaciones complejas o simples.",

"Vector" : "Conjunto ordenado de valores o características, que puede representar puntos en un espacio multidimensional. En muchos modelos, los vectores se utilizan para representar entradas y salidas.",

"Rendimiento" : "Medida de la efectividad de un modelo o sistema para alcanzar un objetivo específico. Se evalúa a través de métricas como la precisión, el recall o el F1,-score.",

"Solución" : "Resultado obtenido al aplicar un conjunto de métodos para resolver un problema específico.",

"Entrada" : "Información o datos proporcionados al sistema o modelo para su procesamiento. Las entradas pueden ser variables numéricas, texto, imágenes u otros tipos de datos.",

"Lógica" : "Conjunto de reglas y principios que determinan cómo se toman las decisiones dentro de un sistema.",

"Filtro" : "Proceso que selecciona o descarta datos en función de ciertas características, para aislar los valores relevantes. Los filtros son esenciales en el preprocesamiento de datos.",

"Índice" : "Valor utilizado para hacer referencia a la posición o ubicación de un elemento dentro de una estructura de datos, como un arreglo o lista.",

"Bajo" : "Se refiere a valores que se encuentran en el extremo inferior de un rango o escala. En análisis de datos, los valores bajos pueden indicar resultados excepcionales o significativos.",

"Fase" : "Una etapa o período específico en el ciclo de vida de un proceso o modelo. Las fases pueden ser fases de entrenamiento, pruebas o evaluación, entre otras.",

"Iteración" : "Proceso repetitivo en el cual se ajustan los parámetros de un modelo en cada paso para mejorar los resultados. Refina el modelo hacia una mejor solución.",

"Ponderar" : "Proceso de asignar un peso o valor relativo a diferentes elementos dentro de un modelo, para indicar su importancia en la toma de decisiones o predicciones.",

"Métrica" : "Indicador numérico utilizado para evaluar el rendimiento de un modelo. Las métricas comunes incluyen precisión, recall y F1-score, entre otras.",

"Cluster" : "Conjunto de elementos que comparten características similares, agrupados en un solo grupo o categoría, utilizado en tareas de clasificación y segmentación.",

"Capa" : "Componente de una red neuronal que contiene una serie de neuronas o nodos. Las capas se organizan en la red para realizar distintas transformaciones de los datos.",

"Vector" : "Representación numérica de un conjunto de características o datos. Los vectores son fundamentales en el procesamiento de entradas y salidas de modelos de aprendizaje.",

"Rango" : "Diferencia entre el valor máximo y el mínimo de un conjunto de datos, utilizado para evaluar la dispersión de los mismos.",

"Ajuste" : "Proceso de modificar los parámetros o características de un modelo para que se adapten mejor a los datos de entrenamiento, optimizando su rendimiento.",

"Muestra" : "Subconjunto representativo de un conjunto de datos, utilizado para entrenar o probar un modelo. Las muestras deben ser representativas para evitar sesgos.",

"Bloque" : "Componente o unidad dentro de un sistema que realiza una operación específica. En redes neuronales, los bloques pueden ser capas de procesamiento.",

"Tensión" : "En optimización, se refiere a la fuerza que controla cómo se modifican los parámetros del modelo para evitar el sobreajuste o subajuste.",

"Causa" : "Elemento que origina un cambio o efecto dentro de un proceso. En análisis de datos, las causas pueden identificarse para mejorar el modelo predictivo.",

"Peso" : "Valor numérico asignado a un parámetro dentro de un modelo, que indica su importancia en la predicción o clasificación.",

"Estruct" : "Representación de datos organizada en estructuras como árboles, listas o tablas, que facilitan el procesamiento y análisis.",

"Columna" : "Cada variable o característica de un conjunto de datos tabulados. En bases de datos y hojas de cálculo, las columnas representan diferentes atributos.",

"Nodo" : "Punto de conexión en estructuras como redes o árboles, donde se realizan operaciones o se almacenan datos para su procesamiento.",

"Valor" : "Resultado numérico o categórico que se asigna a una variable o característica dentro de un modelo o conjunto de datos.",

"Iterar" : "Repetir un proceso para realizar ajustes continuos y mejorar un modelo. En algoritmos, la iteración es clave para encontrar una solución óptima.",

"Muestreo" : "Método utilizado para seleccionar una muestra representativa de datos de un conjunto grande. Se usa en técnicas estadísticas y de análisis.",

"Alarma" : "Se refiere a una alerta generada cuando un modelo detecta una condición inusual o cuando un valor predicho excede un umbral específico.",

"Filtro" : "Proceso para seleccionar o eliminar ciertos datos según criterios establecidos. En el análisis de imágenes, por ejemplo, se pueden aplicar filtros para mejorar las características.",

"Causa" : "En un análisis, refiere a los factores que influyen en los resultados obtenidos. Comprender las causas subyacentes ayuda a mejorar los modelos predictivos.",

"Raza" : "En el contexto de datos de clasificación, 'raza' puede referirse a la categoría o etiqueta que se asigna a ciertos grupos dentro de un conjunto de datos.",

"Señal" : "Información que se transmite a través de un sistema, como una red neuronal, para realizar una predicción o clasificación basada en los datos de entrada.",

"Filtro" : "Proceso de selección de datos relevantes o la eliminación de ruido o información no deseada antes de su análisis o uso en un modelo.",

}

class Crossword:
    def __init__(self, width=15, height=15):
        self.width = width
        self.height = height
        self.grid = [[' ' for _ in range(width)] for _ in range(height)]
        self.words = []
        self.word_positions = []
    
    def add_word_horizontal(self, word, row, col):
        """Intenta agregar una palabra horizontalmente."""
        if col + len(word) <= self.width:
            for i in range(len(word)):
                if self.grid[row][col + i] != ' ' and self.grid[row][col + i] != word[i]:
                    return False  # Colisión
            for i in range(len(word)):
                self.grid[row][col + i] = word[i]
            return True
        return False

    def add_word_vertical(self, word, row, col):
        """Intenta agregar una palabra verticalmente."""
        if row + len(word) <= self.height:
            for i in range(len(word)):
                if self.grid[row + i][col] != ' ' and self.grid[row + i][col] != word[i]:
                    return False  # Colisión
            for i in range(len(word)):
                self.grid[row + i][col] = word[i]
            return True
        return False

    def is_valid_position(self, word, row, col, direction):
        """Verifica si una palabra puede ser colocada en la posición."""
        if direction == "horizontal":
            return self.add_word_horizontal(word, row, col)
        elif direction == "vertical":
            return self.add_word_vertical(word, row, col)
        return False

    def backtrack(self, index=0):
        """Algoritmo de backtracking para colocar todas las palabras."""
        if index == len(self.words):
            return True  # Todas las palabras han sido colocadas

        word = self.words[index]
        random.shuffle(self.word_positions)  # Intentar diferentes posiciones aleatoriamente

        for row, col in self.word_positions:
            for direction in ['horizontal', 'vertical']:
                if self.is_valid_position(word, row, col, direction):
                    if self.backtrack(index + 1):
                        return True
                    self.remove_word(word, row, col, direction)  # Backtrack

        return False  # No se encontró una solución

    def remove_word(self, word, row, col, direction):
        """Eliminar palabra de la cuadrícula."""
        if direction == "horizontal":
            for i in range(len(word)):
                self.grid[row][col + i] = ' '
        elif direction == "vertical":
            for i in range(len(word)):
                self.grid[row + i][col] = ' '

    def print_grid(self):
        """Imprimir el crucigrama."""
        for row in self.grid:
            print(" ".join(row))

    def add_words(self, words):
        """Agregar palabras al crucigrama."""
        self.words = words
        self.word_positions = [(row, col) for row in range(self.height) for col in range(self.width)]
        if not self.backtrack():
            print("No se pudo generar el crucigrama.")
        else:
            self.print_grid()

# Función para generar un tablero aleatorio
def generar_tablero():
    letras = list("abcdefghijklmnopqrstuvwxyzñ ")
    return np.random.choice(letras, (TAM_TABLERO, TAM_TABLERO))

# Inicialización de la población
def inicializar_poblacion():
    return [generar_tablero() for _ in range(TAM_POBLACION)]

def fitness(tablero):
    puntuacion = 0
    for i in range(TAM_TABLERO):
        for j in range(TAM_TABLERO):
            if tablero[i][j] == crucigrama.grid[i][j]:
                puntuacion += 1
    return puntuacion

# Selección por ruleta
def seleccion(poblacion, fitness_scores):
    total_fitness = sum(fitness_scores)
    if total_fitness > 0 :
        probabilidades = [score / total_fitness for score in fitness_scores]
    else:
        probabilidades = [0 for _ in fitness_scores]
    return random.choices(poblacion, weights=probabilidades, k=2)

# Cruce (crossover)
def crossover(parent1, parent2):
    if random.random() > TASA_CRUCE:
        return parent1.copy(), parent2.copy()
    punto_corte = random.randint(1, TAM_TABLERO - 1) # elección al azar del punto de cruce
    hijo1 = np.vstack((parent1[:punto_corte], parent2[punto_corte:]))
    hijo2 = np.vstack((parent2[:punto_corte], parent1[punto_corte:]))
    return hijo1, hijo2

# Mutación
def mutacion(tablero):
    if random.random() < TASA_MUTACION:
        x, y = random.randint(0, TAM_TABLERO - 1), random.randint(0, TAM_TABLERO - 1)
        tablero[x, y] = random.choice(list("abcdefghijklmnopqrstuvwxyzñ "))
    return tablero

# Algoritmo genético
def algoritmo_genetico():
    poblacion = inicializar_poblacion()
    mejor_tablero = None
    mejor_fitness = 0
    
    for generacion in range(GENERACIONES + 1):
        fitness_scores = [fitness(tablero) for tablero in poblacion]
        
        # Actualizar mejor solución
        max_fitness = max(fitness_scores)
        if max_fitness > mejor_fitness:
            mejor_fitness = max_fitness
            mejor_tablero = poblacion[np.argmax(fitness_scores)]
        
        # Nueva generación
        nueva_poblacion = []
        if isinstance(mejor_tablero, np.ndarray):
            nueva_poblacion.extend([mejor_tablero])

        while len(nueva_poblacion) < TAM_POBLACION:
            padre1, padre2 = seleccion(poblacion, fitness_scores)
            hijo1, hijo2 = crossover(padre1, padre2)
            hijo1 = mutacion(hijo1)
            hijo2 = mutacion(hijo2)
            if fitness(hijo1) >= fitness(hijo2):
                nueva_poblacion.extend([hijo1])
            else:
                nueva_poblacion.extend([hijo2])
            
        poblacion = nueva_poblacion[:TAM_POBLACION]
        
        # Mostrar progreso
        if generacion % 50 == 0:
            st.markdown(f"Generación {generacion}: Mejor Fitness = {mejor_fitness}")
            st.dataframe(mejor_tablero, width=800, height=550)
    
    return mejor_tablero, mejor_fitness

TAM_TABLERO = 15
TAM_POBLACION = 100
GENERACIONES = 500
TASA_CRUCE = 0.7
TASA_MUTACION = 0.1

# Lista de palabras a agregar
VersionDeCrucigrama = 0
palabras = []
definiciones = []
crucigrama = Crossword()

def show_crossword():
    global palabras, definiciones, crucigrama
    col3, col4 = st.columns([1, 3])

    # Generar palabras
    porcentaje = 1
    while(porcentaje>0.75):
        aux = pd.DataFrame([[len(x), unidecode(x.lower()), y] for  x, y in Vocabulario.items() if len(x) < 11]).sample(frac=1).head(26)
        palabras = aux[1].values
        definiciones = aux[2].values
        porcentaje = sum([len(x) for x in palabras])/15**2
        print(f"Porcentaje de Palabras: {porcentaje:.04f}")

    # Crear y agregar palabras al crucigrama
    crucigrama = Crossword()
    crucigrama.add_words(palabras)
    CasillasUsadas = sum([sum([1 for y in x  if y != ' ']) for x in crucigrama.grid])/15**2
    print(f"Porcentaje de Casillas Usadas: {CasillasUsadas:.04f}")
    
    # Mostrar las palabras horizontales y verticales
    with col3:
        st.markdown("## Palabras")
        text = "\n"
        for i, word in enumerate(definiciones):
            text += f"### Palabra {i+1} \n"
            text += word + "\n\n"

        st.markdown(f"""
        <div style="max-height: 700px; overflow-y: scroll; border: 0px solid #ddd; padding: 10px;">
            {text}
            """, unsafe_allow_html=True)
    if st.button("Nuevo Crucigrama"):
        print("Hola mundo")
    
    # Mostrar el tablero en el centro
    with col4:
        st.markdown("## Tablero De Juego")

        mejor_tablero, mejor_fitness = algoritmo_genetico()

        st.markdown("### Tablero Real")
        st.table(crucigrama.grid)

        print(f"Mejor Fitness: {mejor_fitness}")
        print("Tablero resultante:")
        for fila in mejor_tablero:
            for letra in fila:
                print(letra, end=" ")
            print()



# Iniciar Streamlit
if __name__ == "__main__":
    st.title("Generador Y Resolvedor De Crucigramas")
    show_crossword()
