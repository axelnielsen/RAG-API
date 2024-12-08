RAG LegalTech IPS

RAG LegalTech  es una propuesta de solución basada en Retrieval-Augmented Generation (RAG) diseñado para optimizar la gestión de consultas legales simples en el Instituto de Previsión Social (IPS). Este proyecto combina inteligencia artificial generativa con recuperación de documentos relevantes, proporcionando respuestas rápidas, precisas y contextualizadas en base a datos normativos y administrativos.

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

RAG LegalTech permite:

	•	Procesar grandes volúmenes de datos normativos y de distinta naturaleza.
 
	•	Generar respuestas relevantes a consultas legales simples.
 
 	•	Clasificar consultas en base a parámetros personalizables
  
	•	Reducir significativamente los tiempos de respuesta en entornos operativos reales.
 

Propósito

Desarrollado para el Desafío Ley Ágil IPS, este proyecto aborda problemáticas relacionadas con la gestión de consultas legales simples en instituciones públicas.

2. Características Principales
	1.	Generación de respuestas contextuales basadas en modelos de lenguaje como Llama3.
	2.	Recuperación de documentos relevantes mediante ChromaDB.
	3.	Optimización continua con MLflow.
	4.	API RESTful escalable y adaptable a sistemas existentes.

 3. Requisitos.

	1. Levantar MLflow

MLflow es una plataforma para gestionar experimentos de machine learning. A continuación, se detalla cómo configurarlo:

Requisitos previos
	•	Python 3.7 o superior.
	•	Librería mlflow instalada (pip install mlflow).
	•	Acceso a un servicio de almacenamiento, como S3, para registrar modelos.

Paso a paso

	1.	Instalar MLflow: pip install mlflow
 
 	2.	Crear un directorio de proyecto: mkdir mlflow_project && cd mlflow_project
  
	3.	Iniciar el servidor MLflow:
 
		mlflow server \
		    --backend-store-uri sqlite:///mlflow.db \
		    --default-artifact-root ./mlruns \
		    --host 0.0.0.0 \
		    --port 5000

		--backend-store-uri: URI de la base de datos para almacenar metadatos. En este caso, una base SQLite.
		--default-artifact-root: Carpeta o bucket para almacenar artefactos (modelos, logs, etc.).

 	4.	Acceder a la interfaz de usuario:
  
Abrir un navegador y acceder a http://127.0.0.1:5000 para visualizar la interfaz de MLflow.
	5.	Registrar experimentos:
 
		Dentro de  script de entrenamiento, agregar:
  
		import mlflow

		mlflow.set_tracking_uri("http://127.0.0.1:5000")
		mlflow.set_experiment("nombre_del_experimento")
		
		with mlflow.start_run():
		    mlflow.log_param("parametro", valor)
		    mlflow.log_metric("metrica", valor)
		    mlflow.log_artifact("ruta_al_archivo")

2. Configurar y correr Ollama para Llama3

Requisitos previos
	•	Ollama Server instalado (compatible con Linux, macOS o Windows).
	•	Docker (opcional, si usas la versión Dockerizada de Ollama).
	•	Llama3 modelo preentrenado descargado o configurado.

Paso a paso
	1.	Instalar Ollama:
	2.	En sistemas compatibles, instala Ollama con: brew install ollama
 	3.	Iniciar Ollama Server: ollama serve
  	4.	Descargar Llama3: ollama chat llama3
	5.	Configurar una API para usar Llama3:
Conectarlo con API:
	•	Crear un archivo app.py con Flask para exponer un endpoint: 
 		from flask import Flask, request, jsonify
   
			import ollama
			
			app = Flask(__name__)
			
			@app.route('/predict', methods=['POST'])
			def predict():
			    data = request.json.get('prompt')
			    response = ollama.completion("llama3", prompt=data)
			    return jsonify(response)
			
			if __name__ == '__main__':
			    app.run(host='0.0.0.0', port=5001)


 4. Arquitectura del Sistema

El sistema está compuesto por los siguientes módulos:

	1.	GenAI API: API basada en Flask que coordina las solicitudes entre los componentes del sistema.
 
	2.	ChromaDB: Base de datos vectorial para almacenamiento y recuperación eficiente de embeddings.
 
	3.	MLflow: Gestión de experimentos y monitoreo de métricas.
 
	4.	Ollama Server: Generación de embeddings a partir de datos normativos y administrativos.
 
	5.	Modelo LLM: Utilizado para generar respuestas basadas en el contexto proporcionado.

Diagrama de Arquitectura

A grandes rasgos la arquitectura del sistema RAG se compone de los siguientes elementos:

<img width="384" alt="image" src="https://github.com/user-attachments/assets/3431f913-824c-4737-b5af-01ccf6f3ef5d">

Así, para llevar dicha solución a la nube de AWS, los componentes srquitectónicos son los siguientes:

<img width="1196" alt="image" src="https://github.com/user-attachments/assets/32c22f39-2341-497e-8494-d827738bcaef">

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









 
