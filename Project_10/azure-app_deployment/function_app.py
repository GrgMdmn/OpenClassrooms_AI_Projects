import azure.functions as func
import logging
import json
import os
import sys

# Ajouter le dossier utils au path pour les imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import load_embeddings, load_user_data
from utils.content_based import get_recommendations

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

@app.route(route="recommend")
def recommend(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Recommendation request received.')

    # Récupérer le user_id depuis les paramètres de requête
    user_id = req.params.get('user_id')
    
    if not user_id:
        try:
            req_body = req.get_json()
            user_id = req_body.get('user_id') if req_body else None
        except ValueError:
            pass

    if not user_id:
        return func.HttpResponse(
            json.dumps({"error": "user_id parameter required"}),
            status_code=400,
            mimetype="application/json"
        )

    try:
        # Convertir en entier
        user_id = int(user_id)
        
        # Générer les recommandations
        recommendations = get_recommendations(user_id)
        
        if recommendations:
            return func.HttpResponse(
                json.dumps({
                    "user_id": user_id,
                    "recommendations": recommendations
                }),
                status_code=200,
                mimetype="application/json"
            )
        else:
            return func.HttpResponse(
                json.dumps({
                    "error": f"No recommendations found for user {user_id}"
                }),
                status_code=404,
                mimetype="application/json"
            )
            
    except ValueError:
        return func.HttpResponse(
            json.dumps({"error": "user_id must be a valid integer"}),
            status_code=400,
            mimetype="application/json"
        )
    except Exception as e:
        logging.error(f"Error generating recommendations: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": "Internal server error"}),
            status_code=500,
            mimetype="application/json"
        )