from app.models import CustomLLM
from app.utils import generate_response, calculate_metrics, generate_resume
from flask import request, jsonify, Blueprint
import mlflow
import logging
import psycopg2
import pandas as pd
import time

logger = logging.getLogger(__name__)

main_bp = Blueprint('main', __name__)

@main_bp.route('/resumir_reclamos', methods=['POST'])
def resumir_reclamos():
    """
    Resumir reclamos desde el modelo
    """
    return generate_resume_response()

@main_bp.route('/buscar_reclamos_prob', methods=['POST'])
def buscar_reclamos_prob():
    """
    Buscar reclamos desde el modelo según problemática de consumo
    """
    return generate_generic_response()

def generate_resume_response():
    try:
        data = request.json
        if data is None:
            raise ValueError("Invalid JSON")
        
        prompt = data.get('prompt')
        ground_truth = data.get('ground_truth')
        temperature = data.get('temperature', 0.7)
        top_k = data.get('top_k', 50)
        top_p = data.get('top_p', 0.9)
        repetition_penalty = data.get('repetition_penalty', 1.0)
        frequency_penalty = data.get('frequency_penalty', 0.0)
        presence_penalty = data.get('presence_penalty', 0.0)
        run_name = data.get('run_name')
        collection_name = data.get('collection')
        
        llm = CustomLLM(model_name="llama3")

        response = generate_resume(
            llm, prompt, temperature, top_k, top_p, 
            repetition_penalty, frequency_penalty, presence_penalty, 
            run_name, collection_name
        )

        metrics = calculate_metrics([prompt], [response['response']], [ground_truth], [response['response']])
        
        result = {
            "response": response['response'],
            "metrics": metrics,
            "ids_reclamos": response.get("ids_reclamos", [])  # Incluye los IDs de los reclamos en la respuesta
        }

        return jsonify(result)
    except ValueError as ve:
        logger.error(f"Invalid JSON: {str(ve)}")
        return jsonify({'error': 'Invalid JSON'}), 400
    except Exception as e:
        logger.error(f"Error in generate endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

def generate_generic_response():
    try:
        data = request.json
        if data is None:
            raise ValueError("Invalid JSON")
        
        prompt = data.get('prompt')
        ground_truth = data.get('ground_truth')
        temperature = data.get('temperature', 0.5)
        top_k = data.get('top_k', 50)
        top_p = data.get('top_p', 0.9)
        repetition_penalty = data.get('repetition_penalty', 1.0)
        frequency_penalty = data.get('frequency_penalty', 0.0)
        presence_penalty = data.get('presence_penalty', 0.0)
        run_name = data.get('run_name')
        collection_name = data.get('collection')
        
        llm = CustomLLM(model_name="llama3")

        # Genera la respuesta y recupera los reclamos
        response = generate_response(
            llm, prompt, temperature, top_k, top_p, 
            repetition_penalty, frequency_penalty, presence_penalty, 
            run_name, collection_name
        )

        # Extrae los reclamos y sus metadatos para formar la respuesta final
        ids_reclamos = response.get("ids_reclamos", [])
        
        # Crear la respuesta final que incluye los ids de los reclamos
        result = {
            "response": response['response'],
            "metrics": calculate_metrics([prompt], [response['response']], [ground_truth], [response['response']]),
            "ids_reclamos": ids_reclamos  # Incluir los IDs de reclamos en la respuesta
        }

        return jsonify(result)
    except ValueError as ve:
        logger.error(f"Invalid JSON: {str(ve)}")
        return jsonify({'error': 'Invalid JSON'}), 400
    except Exception as e:
        logger.error(f"Error in generate endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500
