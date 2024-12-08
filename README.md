RAG LegalTech IPS

RAG LegalTech IPS es un sistema basado en Retrieval-Augmented Generation (RAG) diseñado para optimizar la gestión de consultas legales simples en el Instituto de Previsión Social (IPS). Este proyecto combina inteligencia artificial generativa con recuperación de documentos relevantes, proporcionando respuestas rápidas, precisas y contextualizadas en base a datos normativos y administrativos.

Índice
	1.	Introducción
 
	2.	Características Principales

	3.	Requisitos
 
	4.	Arquitectura del Sistema
 
	5.	Endpoints de la API
 
	6.	Pruebas y Resultados
 
	7.	Contribuciones
 
	8.	Licencia
 
1. Introducción

RAG LegalTech IPS permite:
	•	Procesar grandes volúmenes de datos normativos.
	•	Generar respuestas relevantes a consultas legales simples.
	•	Reducir significativamente los tiempos de respuesta en entornos operativos reales.

Propósito

Desarrollado para el Desafío Ley Ágil IPS, este proyecto aborda problemáticas relacionadas con la gestión de consultas legales simples en instituciones públicas.

2. Características Principales
	1.	Generación de respuestas contextuales basadas en modelos de lenguaje como Llama3.
	2.	Recuperación de documentos relevantes mediante ChromaDB.
	3.	Optimización continua con MLflow.
	4.	API RESTful escalable y adaptable a sistemas existentes.

 3. Requisitos

Software necesario
	•	Sistema operativo: Linux Debian 10.2.1-6
	•	Herramientas principales:
	•	Ollama Server v0.1.32
	•	Conda v23.3.1
	•	Python v3.10
	•	Librerías Python: Listadas en el archivo requirements.txt.

 4. Arquitectura del Sistema

El sistema está compuesto por los siguientes módulos:

	1.	GenAI API: API basada en Flask que coordina las solicitudes entre los componentes del sistema.
 
	2.	ChromaDB: Base de datos vectorial para almacenamiento y recuperación eficiente de embeddings.
 
	3.	MLflow: Gestión de experimentos y monitoreo de métricas.
 
	4.	Ollama Server: Generación de embeddings a partir de datos normativos y administrativos.
 
	5.	Modelo LLM: Utilizado para generar respuestas basadas en el contexto proporcionado.

Diagrama de Arquitectura

<img width="384" alt="image" src="https://github.com/user-attachments/assets/3431f913-824c-4737-b5af-01ccf6f3ef5d">

5. Endpoints de la API

/resumir_reclamos
	•	Descripción: Genera resúmenes de reclamos basados en prompts.
	•	Método HTTP: POST
 
 <img width="369" alt="image" src="https://github.com/user-attachments/assets/8229f3a7-ed0b-43bb-9a94-61fdb20e3177">


/buscar_reclamos_prob
	•	Descripción: Busca problemáticas según criterios específicos.
	•	Método HTTP: POST

 <img width="387" alt="image" src="https://github.com/user-attachments/assets/41fc4ce2-a96b-4d35-8925-c0697040185b">


/vectorizar_datos
	•	Descripción: Vectoriza datos desde tablas PostgreSQL o documentos y los almacena en ChromaDB.
	•	Método HTTP: POST
 
<img width="395" alt="image" src="https://github.com/user-attachments/assets/bd7b3a23-2d5c-4f71-9060-8ed789a79702">

6. Pruebas y Resultados

Se realizaron pruebas masivas para evaluar la calidad y precisión de las respuestas generadas. Los resultados muestran configuraciones óptimas para BLEU Score, Meteor Score y Similaridad.


Configuración Recomendada
	•	Temperature: 0.5
	•	Top_p: 0.85 - 0.9

Ejemplo de Resultados

<img width="340" alt="image" src="https://github.com/user-attachments/assets/10c388f7-453e-4a5b-b537-22b7233ad6e9">


<img width="340" alt="image" src="https://github.com/user-attachments/assets/682d2e64-6e7c-4582-b027-08984dd57811">


<img width="340" alt="image" src="https://github.com/user-attachments/assets/2b8a90bf-795a-4846-b5f6-34f8705481d0">


<img width="340" alt="image" src="https://github.com/user-attachments/assets/a564ebb4-6a2c-48c8-9036-704a5895d18a">


<img width="340" alt="image" src="https://github.com/user-attachments/assets/62dd6c65-aeaa-4095-8bac-ef493ef587ff">


<img width="340" alt="image" src="https://github.com/user-attachments/assets/96282ed5-6782-4b6b-bfbe-d6c8cb07aceb">









 
