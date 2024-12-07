import mlflow
import chromadb
import time
import traceback
import logging
import ollama
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score

logger = logging.getLogger(__name__)

def calculate_similarity(answer, ground_truth):
    vectorizer = TfidfVectorizer().fit_transform([answer, ground_truth])
    vectors = vectorizer.toarray()
    similarity = cosine_similarity(vectors)
    return similarity[0, 1]

def calculate_bleu(answers, ground_truths):
    smoothie = SmoothingFunction().method4
    bleu_scores = [sentence_bleu([g.split()], a.split(), smoothing_function=smoothie) for a, g in zip(answers, ground_truths)]
    return sum(bleu_scores) / len(bleu_scores)

def calculate_rouge(answers, ground_truths):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge1_scores = [scorer.score(a, g)['rouge1'].fmeasure for a, g in zip(answers, ground_truths)]
    rougeL_scores = [scorer.score(a, g)['rougeL'].fmeasure for a, g in zip(answers, ground_truths)]
    return sum(rouge1_scores) / len(rouge1_scores), sum(rougeL_scores) / len(rougeL_scores)

def calculate_meteor(answers, ground_truths):
    tokenized_answers = [a.split() for a in answers]
    tokenized_ground_truths = [[g.split()] for g in ground_truths]
    meteor_scores = [meteor_score(g, a) for a, g in zip(tokenized_answers, tokenized_ground_truths)]
    return sum(meteor_scores) / len(meteor_scores)

def calculate_accuracy(answers, ground_truths):
    correct_answers = sum([1 if a == g else 0 for a, g in zip(answers, ground_truths)])
    accuracy = correct_answers / len(ground_truths)
    return accuracy

def calculate_metrics(questions, generated_answers, ground_truths, contexts):
    similarity_scores = [calculate_similarity(a, g) for a, g in zip(generated_answers, ground_truths)]
    accuracy = calculate_accuracy(generated_answers, ground_truths)
    bleu_score = calculate_bleu(generated_answers, ground_truths)
    rouge1_score, rougeL_score = calculate_rouge(generated_answers, ground_truths)
    meteor_score = calculate_meteor(generated_answers, ground_truths)
    
    return {
        "similarity": similarity_scores,
        "accuracy": accuracy,
        "bleu_score": bleu_score,
        "rouge1_score": rouge1_score,
        "rougeL_score": rougeL_score,
        "meteor_score": meteor_score
    }

def generate_response(model, prompt, temperature, top_k, top_p, repetition_penalty, frequency_penalty, presence_penalty, run_name, collection_name):
    mlflow.set_tracking_uri("http://localhost:8089")

    chroma_client = chromadb.PersistentClient(path="./chromadb_electricidad_32000_rec")
    collection = chroma_client.get_or_create_collection(name=collection_name)

    with mlflow.start_run(run_name=run_name) as run:
        try:
            mlflow.log_params({
                "model": model.model_name,
                "prompt": prompt,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty
            })

            total_start_time = time.time()

            embedding_response = ollama.embeddings(prompt=prompt, model=model.model_name)
            embedding = embedding_response["embedding"]
            mlflow.log_param("embedding", embedding)
            mlflow.log_metric("embedding_size", len(embedding))

            results = collection.query(query_embeddings=[embedding], n_results=20)
            retrieved_texts = []
            retrieved_metadatas = []
            ids_reclamos = []  # Lista para almacenar los ID de reclamos correctamente recuperados

            for sublist_docs, sublist_metas in zip(results['documents'], results['metadatas']):
                for doc, meta in zip(sublist_docs, sublist_metas):
                    # Asegúrate de que 'id_reclamo' esté presente y correctamente formateado
                    id_reclamo = meta.get('id_reclamo') if isinstance(meta, dict) else 'N/A'
                    if id_reclamo == 'N/A':
                        logger.error(f"'id_reclamo' no encontrado en metadatos: {meta}")
                    else:
                        ids_reclamos.append(id_reclamo)  # Añadir a la lista de IDs de reclamos recuperados

                    metadata_str = f"id_reclamo: {id_reclamo}"
                    retrieved_texts.append(metadata_str + ". descripcion_reclamo: " + doc)
                    retrieved_metadatas.append(meta)
                    logger.info(f"Reclamo recuperado con ID: {id_reclamo}")

            mlflow.log_param("retrieved_documents", retrieved_texts)
            mlflow.log_param("retrieved_metadatas", retrieved_metadatas)
            mlflow.log_metric("num_documents_considered", len(results['documents']))

            # Crear un texto más explícito que incluya el id_reclamo
            combined_texts = "\n".join([
                f"Reclamo ID {meta.get('id_reclamo', 'N/A')}: {doc}"
                for sublist_docs, sublist_metas in zip(results['documents'], results['metadatas'])
                for doc, meta in zip(sublist_docs, sublist_metas)
                if isinstance(meta, dict)
            ])
            
            metadatas_text = "\n".join([
                f"{meta.get('id_reclamo', 'N/A')}"
                for meta in retrieved_metadatas
                if isinstance(meta, dict)
            ])
                
            full_prompt_data = f"Usando la siguiente información de los reclamos, asegúrate de mencionar el 'id_reclamo' tal cual como lo lees cuando describas cada caso: {combined_texts}.\n\nAhora, responde este prompt: {prompt}."

            response = model.generate(full_prompt_data, temperature, top_k, top_p, repetition_penalty, frequency_penalty, presence_penalty)+"\n\nListado de casos con mayor similitud a la problemática:\n\n" + metadatas_text
            response_with_metadata = response

            mlflow.log_param("response", response)
            response_time = time.time() - total_start_time
            mlflow.log_metric("response_time", response_time)
            mlflow.log_metric("response_length", len(response))

            score_df = calculate_metrics([prompt], [response], [combined_texts], [combined_texts])
            for metric_name, scores in score_df.items():
                if isinstance(scores, (list, tuple)):
                    for i, score in enumerate(scores):
                        mlflow.log_metric(f"{metric_name}_{i}", score)
                else:
                    mlflow.log_metric(metric_name, scores)

            result = {
                "response": response_with_metadata,
                "retrieved_documents": retrieved_texts,
                "retrieved_metadatas": retrieved_metadatas,
                "metrics": score_df,
                "ids_reclamos": ids_reclamos  # Devolver la lista de IDs de reclamos
            }

            return result
        
        except Exception as e:
            error_message = str(e)
            error_traceback = traceback.format_exc()
            mlflow.log_param("error_message", error_message)
            mlflow.log_param("error_traceback", error_traceback)
            logger.error(f"Exception during response generation: {error_message}")
            raise


def vectorize_and_store_reclamos(reclamos, metadata_fields, text_field):
    chroma_client = chromadb.PersistentClient(path="./chromadb_storage")
    collection = chroma_client.get_or_create_collection(name="frr_test_vectorizacion")

    for reclamo in reclamos:
        metadata = {field: reclamo[field] for field in metadata_fields}
        descripcion_reclamo = reclamo[text_field]

        try:
            response = ollama.embeddings(model="llama3", prompt=descripcion_reclamo)
            embedding = response["embedding"]
            collection.add(
                ids=[f"r_{metadata['CASO_NUMERO']}"],  # Usar un campo único para asegurar unicidad
                embeddings=[embedding],
                documents=[descripcion_reclamo],
                metadatas=[metadata]
            )
            logging.info(f"Vectorizado el reclamo ID: {metadata['CASO_NUMERO']}")
        except Exception as e:
            logging.error(f"Error al vectorizar y almacenar el reclamo ID {metadata['CASO_NUMERO']}: {e}")
    logging.info(f"Vectorizados y almacenados {len(reclamos)} reclamos en ChromaDB.")

def generate_resume(model, prompt, temperature, top_k, top_p, repetition_penalty, frequency_penalty, presence_penalty, run_name, collection_name):
    mlflow.set_tracking_uri("http://localhost:8089")

    chroma_client = chromadb.PersistentClient(path="./chromadb_electricidad_32000_rec")
    collection = chroma_client.get_or_create_collection(name=collection_name)

    with mlflow.start_run(run_name=run_name) as run:
        try:
            mlflow.log_params({
                "model": model.model_name,
                "prompt": prompt,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty
            })

            total_start_time = time.time()

            embedding_response = ollama.embeddings(prompt=prompt, model=model.model_name)
            embedding = embedding_response["embedding"]
            mlflow.log_param("embedding", embedding)
            mlflow.log_metric("embedding_size", len(embedding))

            results = collection.query(query_embeddings=[embedding], n_results=100)
            retrieved_texts = []
            retrieved_metadatas = []

            for sublist_docs, sublist_metas in zip(results['documents'], results['metadatas']):
                for doc, meta in zip(sublist_docs, sublist_metas):
                    metadata_str = f"id_reclamo: {meta.get('id_reclamo', 'N/A')}"
                    retrieved_texts.append(metadata_str + ". descripcion_reclamo: " + doc)
                    retrieved_metadatas.append(meta)

            mlflow.log_param("retrieved_documents", retrieved_texts)
            mlflow.log_param("retrieved_metadatas", retrieved_metadatas)
            mlflow.log_metric("num_documents_considered", len(results['documents']))

            combined_texts = " ".join(retrieved_texts)
            
            metadatas_text = ""
            for metadata in retrieved_metadatas:
                metadatas_text += f"{metadata.get('id_reclamo', 'N/A')}, \n"
                
            full_prompt_data = f"Usando esta data que describe reclamos {combined_texts}. Haz un resumen de los casos analizados identificando de 6 a 10 problemáticas de consumo. Debe ser un resumen a modo de análisis, en tercera persona. En cada problemática identificada debe haber una explicación de la problemática."
            
            response = model.generate(full_prompt_data, temperature, top_k, top_p, repetition_penalty, frequency_penalty, presence_penalty)
            mlflow.log_param("response", response)
            response_time = time.time() - total_start_time
            mlflow.log_metric("response_time", response_time)
            mlflow.log_metric("response_length", len(response))

            response_with_metadata = response + "\n\n" + metadatas_text

            score_df = calculate_metrics([prompt], [response], [combined_texts], [combined_texts])
            for metric_name, scores in score_df.items():
                if isinstance(scores, (list, tuple)):
                    for i, score in enumerate(scores):
                        mlflow.log_metric(f"{metric_name}_{i}", score)
                else:
                    mlflow.log_metric(metric_name, scores)

            result = {
                "response": response_with_metadata,
                "retrieved_documents": retrieved_texts,
                "retrieved_metadatas": retrieved_metadatas,
                "metrics": score_df
            }

            return result
        
        except Exception as e:
            error_message = str(e)
            error_traceback = traceback.format_exc()
            mlflow.log_param("error_message", error_message)
            mlflow.log_param("error_traceback", error_traceback)
            logger.error(f"Exception during response generation: {error_message}")
            raise
