import os
import json
import hmac
import hashlib
import base64
from typing import Dict, List, Optional

import boto3
from fastapi import FastAPI, Header, HTTPException, Depends, status
from pydantic import BaseModel, Field

# ==============================
# Environment / Config
# ==============================

# API key for protecting the endpoint
API_KEY = os.getenv("API_KEY", "changeme")  # override in Vercel env

# Cognito configuration (all required)
COGNITO_REGION = os.getenv("COGNITO_REGION")
COGNITO_USER_POOL_ID = os.getenv("COGNITO_USER_POOL_ID")
COGNITO_CLIENT_ID = os.getenv("COGNITO_CLIENT_ID")
COGNITO_CLIENT_SECRET = os.getenv("COGNITO_CLIENT_SECRET", "")
COGNITO_USERNAME = os.getenv("COGNITO_USERNAME")
COGNITO_PASSWORD = os.getenv("COGNITO_PASSWORD")
COGNITO_IDENTITY_POOL_ID = os.getenv("COGNITO_IDENTITY_POOL_ID")

# Bedrock configuration
BEDROCK_MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID",
    "anthropic.claude-3-5-sonnet-20241022-v2:0",  # default, override in env
)
BEDROCK_REGION = os.getenv("BEDROCK_REGION") or COGNITO_REGION or "eu-west-1"

# ==============================
# Security – simple API Key auth
# ==============================

async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return True


# ==============================
# Pydantic models (same schema as UI)
# ==============================

class TurnMatching(BaseModel):
    scope: Optional[str] = Field(
        default="any",
        description="Turn matching scope: any | recent | current",
    )
    evaluationStrategy: Optional[str] = Field(
        default="first_match",
        description="Evaluation strategy: first_match | best_match | latest_match",
    )
    recentTurnCount: Optional[int] = Field(
        default=None, description="Required when scope='recent'"
    )


class SubObjective(BaseModel):
    description: str = Field(..., description="What should be achieved")
    isBlocking: Optional[bool] = Field(
        default=False,
        description="If true, failure stops the test",
    )
    instructions: Optional[str] = Field(
        default=None,
        description="Special instructions for evaluation",
    )
    satisfactionCriteria: Optional[List[str]] = Field(
        default=None,
        description="Concrete pass conditions for this sub-objective",
    )
    maxTurnsForObjective: Optional[int] = Field(
        default=12,
        description="Maximum turns for this sub objective",
    )
    turnMatching: Optional[TurnMatching] = Field(
        default=None,
        description="Turn-based objective matching configuration",
    )


class CompositeObjective(BaseModel):
    name: Optional[str] = Field(default=None)
    description: Optional[str] = Field(default=None)
    domain: Optional[str] = Field(
        default=None,
        description="Optional domain label, e.g. telecom_billing, banking",
    )
    persona: str = Field(..., description="Persona or customer type")
    userVariables: Optional[Dict[str, str]] = Field(
        default=None,
        description="Context variables for the persona",
    )
    subObjectives: List[SubObjective] = Field(
        ..., min_items=1, description="At least one sub objective required"
    )


# ==============================
# Response models (what Bedrock should return)
# ==============================

class Recommendation(BaseModel):
    reason: str
    suggestedDefiningObjective: str
    alternativeDefiningObjective: str


class SubObjectiveRecommendation(BaseModel):
    index: int
    currentDefiningObjective: str
    recommendation: Recommendation


class RecommendResponse(BaseModel):
    subObjectives: List[SubObjectiveRecommendation]


# ==============================
# Cognito helpers (User Pool + Identity Pool)
# ==============================

def _compute_secret_hash(username: str) -> Optional[str]:
    """
    Compute Cognito SECRET_HASH if client secret is configured.
    If no client secret is set, return None.
    """
    if not COGNITO_CLIENT_SECRET:
        return None

    message = (username + COGNITO_CLIENT_ID).encode("utf-8")
    key = COGNITO_CLIENT_SECRET.encode("utf-8")
    digest = hmac.new(key, message, hashlib.sha256).digest()
    return base64.b64encode(digest).decode("utf-8")


def authenticate_user_and_get_id_token() -> str:
    """
    Authenticate against Cognito User Pool and return the ID token.
    Uses USER_PASSWORD_AUTH and optional SECRET_HASH.
    """
    if not all([COGNITO_REGION, COGNITO_CLIENT_ID, COGNITO_USERNAME, COGNITO_PASSWORD, COGNITO_USER_POOL_ID]):
        raise RuntimeError("Cognito User Pool environment variables are not fully configured")

    cognito_idp = boto3.client("cognito-idp", region_name=COGNITO_REGION)

    auth_params = {
        "USERNAME": COGNITO_USERNAME,
        "PASSWORD": COGNITO_PASSWORD,
    }

    secret_hash = _compute_secret_hash(COGNITO_USERNAME)
    if secret_hash:
        auth_params["SECRET_HASH"] = secret_hash

    resp = cognito_idp.initiate_auth(
        AuthFlow="USER_PASSWORD_AUTH",
        AuthParameters=auth_params,
        ClientId=COGNITO_CLIENT_ID,
    )

    id_token = resp["AuthenticationResult"]["IdToken"]
    return id_token


def get_temporary_aws_credentials(id_token: str) -> Dict[str, str]:
    """
    Use Cognito Identity Pool to exchange the ID token for temporary AWS credentials.
    """
    if not all([COGNITO_REGION, COGNITO_IDENTITY_POOL_ID, COGNITO_USER_POOL_ID]):
        raise RuntimeError("Cognito Identity Pool environment variables are not fully configured")

    provider = f"cognito-idp.{COGNITO_REGION}.amazonaws.com/{COGNITO_USER_POOL_ID}"

    cognito_identity = boto3.client("cognito-identity", region_name=COGNITO_REGION)

    identity = cognito_identity.get_id(
        IdentityPoolId=COGNITO_IDENTITY_POOL_ID,
        Logins={provider: id_token},
    )

    creds_resp = cognito_identity.get_credentials_for_identity(
        IdentityId=identity["IdentityId"],
        Logins={provider: id_token},
    )

    return creds_resp["Credentials"]


def get_bedrock_client():
    """
    Build a Bedrock Runtime client using temporary AWS credentials
    obtained via Cognito (user pool + identity pool).
    """
    id_token = authenticate_user_and_get_id_token()
    creds = get_temporary_aws_credentials(id_token)

    bedrock_client = boto3.client(
        "bedrock-runtime",
        region_name=BEDROCK_REGION,
        aws_access_key_id=creds["AccessKeyId"],
        aws_secret_access_key=creds["SecretKey"],
        aws_session_token=creds["SessionToken"],
    )
    return bedrock_client


# ==============================
# Bedrock prompt / call
# ==============================

SYSTEM_PROMPT = """
You are a test objective recommendation assistant.

The user will send a JSON object called CompositeObjective, with this structure:

{
  "name": "string (optional)",
  "description": "string (optional)",
  "domain": "string (optional)",
  "persona": "string (required)",
  "userVariables": { "key": "value" },
  "subObjectives": [
    {
      "description": "string",
      "isBlocking": boolean,
      "instructions": "string",
      "satisfactionCriteria": ["string"],
      "maxTurnsForObjective": number,
      "turnMatching": {
        "scope": "any | recent | current",
        "evaluationStrategy": "first_match | best_match | latest_match",
        "recentTurnCount": number
      }
    }
  ]
}

For each sub-objective, you must:

1. Analyse the current "description" as the current defining objective.
2. Suggest a clearer, more testable defining objective.
3. Provide an alternative defining objective.
4. Explain briefly why your suggestion is better.

Return ONLY valid JSON with this structure (no explanation text outside JSON):

{
  "subObjectives": [
    {
      "index": 0,
      "currentDefiningObjective": "string",
      "recommendation": {
        "reason": "string",
        "suggestedDefiningObjective": "string",
        "alternativeDefiningObjective": "string"
      }
    }
  ]
}
""".strip()


def call_bedrock(composite: CompositeObjective) -> RecommendResponse:
    """
    Call Bedrock (Claude-style) and parse the JSON the model returns.
    """

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "system": SYSTEM_PROMPT,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(composite.model_dump(), indent=2),
                    }
                ],
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.0,
    }

    client = get_bedrock_client()

    response = client.invoke_model(
        modelId=BEDROCK_MODEL_ID,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
    )

    raw_body = response.get("body").read()
    resp_json = json.loads(raw_body)

    text_chunks = resp_json.get("content", [])
    if not text_chunks or "text" not in text_chunks[0]:
        raise ValueError("Model returned no text content")

    raw_text = text_chunks[0]["text"].strip()

    parsed = json.loads(raw_text)

    return RecommendResponse(**parsed)


# ==============================
# FastAPI app & endpoints
# ==============================

app = FastAPI(
    title="Objective Recommender API",
    description="FastAPI wrapper around Bedrock (via Cognito) to recommend clearer defining objectives.",
    version="1.0.0",
)


@app.get("/", include_in_schema=False)
async def root():
    return {"status": "ok", "message": "Objective Recommender API"}


@app.post(
    "/recommend",
    response_model=RecommendResponse,
    summary="Recommend clearer defining objectives",
    dependencies=[Depends(verify_api_key)],
)
async def recommend_objectives(payload: CompositeObjective):
    """
    Accepts a CompositeObjective and returns improved defining objectives
    for each sub-objective.

    Security: requires `X-API-Key` header with the configured API key.
    """
    try:
        result = call_bedrock(payload)
        return result
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse model JSON: {e}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calling Bedrock: {e}",
        )


# Optional: local dev entrypoint
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
