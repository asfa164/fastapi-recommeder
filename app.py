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
# Simple schema – single objective with structured context
# ==============================


class SimpleContext(BaseModel):
    persona: Optional[str] = Field(
        default=None,
        description="(optional) Persona or customer type, e.g. 'Postpaid telecom customer in Ireland'.",
    )
    domain: Optional[str] = Field(
        default=None,
        description="(optional) Domain label, e.g. 'telecom_billing', 'banking', 'travel'.",
    )
    instructions: Optional[str] = Field(
        default=None,
        description="(optional) High-level instructions for how the agent/test should behave.",
    )
    satisfactionCriteria: Optional[List[str]] = Field(
        default=None,
        description="(optional) List of concrete pass/fail criteria for this defining objective.",
    )
    extraNotes: Optional[str] = Field(
        default=None,
        description="(optional) Any additional scenario details not covered above.",
    )


class SimpleObjectiveRequest(BaseModel):
    objective: str = Field(
        ...,
        description="(required) The defining objective or prompt you want to rewrite/refine.",
    )
    context: Optional[SimpleContext] = Field(
        default=None,
        description=(
            "(optional) Structured context object. You may omit it entirely. "
            "If provided, ALL fields inside context are also optional."
        ),
    )

    class Config:
        schema_extra = {
            "example": {
                "objective": "What is this extra charge?",
                "context": {
                    "persona": "Postpaid telecom customer in Ireland",
                    "domain": "telecom_billing",
                    "instructions": (
                        "Treat this as a vague billing query. The customer only says "
                        "'What is this extra charge?' with no further context. Focus on how the "
                        "agent elicits the right information before promising a resolution."
                    ),
                    "satisfactionCriteria": [
                        "Agent acknowledges the concern about the extra charge.",
                        "Agent asks for at least one specific piece of information to identify the charge "
                        "(e.g. date, amount, invoice ID, last 4 digits of card).",
                        "Agent avoids confirming the cause of the charge before seeing the relevant bill details."
                    ],
                    "extraNotes": "Customer is not angry yet, just confused. Keep tone calm and reassuring."
                }
            }
        }


# ==============================
# Response model
# ==============================


class SimpleRecommendResponse(BaseModel):
    reason: str
    suggestedDefiningObjective: str
    alternativeDefiningObjective: str


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
    if not all(
        [
            COGNITO_REGION,
            COGNITO_CLIENT_ID,
            COGNITO_USERNAME,
            COGNITO_PASSWORD,
            COGNITO_USER_POOL_ID,
        ]
    ):
        raise RuntimeError(
            "Cognito User Pool environment variables are not fully configured"
        )

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
        raise RuntimeError(
            "Cognito Identity Pool environment variables are not fully configured"
        )

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
# Bedrock prompt / call (simple mode only)
# ==============================

SYSTEM_PROMPT_SIMPLE = """
You are a test objective rewriting assistant.

The user will send you a JSON object with this structure:

{
  "objective": "string (required)",
  "context": {
    "persona": "string (optional)",
    "domain": "string (optional)",
    "instructions": "string (optional)",
    "satisfactionCriteria": ["string", "... (optional)"],
    "extraNotes": "string (optional)"
  }
}

- "objective" is the only required field. It may be vague, underspecified or poorly worded.
- "context" is an optional structured helper. It may include persona, domain, instructions, satisfaction criteria,
  and any other scenario details, but you must still be able to work if it is missing.

Your task:

1. Analyse the current "objective" as a defining objective for a test scenario.
2. Suggest a clearer, more specific, and more testable defining objective.
3. Provide an alternative defining objective that takes a slightly different angle.
4. Briefly explain why your primary suggestion is better than the original.

Important:
- Focus on rewriting the objective itself.
- Do NOT invent extra constraints that are not implied by the original objective plus the optional structured context.
- If context is missing, still produce good suggestions based only on the objective text.

Return ONLY valid JSON with this structure (no explanation text outside JSON):

{
  "reason": "string",
  "suggestedDefiningObjective": "string",
  "alternativeDefiningObjective": "string"
}
""".strip()


def _invoke_bedrock(body: dict) -> dict:
    client = get_bedrock_client()

    response = client.invoke_model(
        modelId=BEDROCK_MODEL_ID,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
    )

    raw_body = response.get("body").read()
    return json.loads(raw_body)


def call_bedrock_simple(payload: SimpleObjectiveRequest) -> SimpleRecommendResponse:
    """
    Call Bedrock for a single vague objective (simple mode) and parse JSON.
    """
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "system": SYSTEM_PROMPT_SIMPLE,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        # send structured JSON so the model sees the structured context object
                        "text": json.dumps(payload.model_dump(), indent=2),
                    }
                ],
            }
        ],
        "max_tokens": 512,
        "temperature": 0.0,
    }

    resp_json = _invoke_bedrock(body)

    text_chunks = resp_json.get("content", [])
    if not text_chunks or "text" not in text_chunks[0]:
        raise ValueError("Model returned no text content")

    raw_text = text_chunks[0]["text"].strip()
    parsed = json.loads(raw_text)

    return SimpleRecommendResponse(**parsed)


# ==============================
# FastAPI app & single endpoint
# ==============================

app = FastAPI(
    title="Objective Recommendation API",
    description=(
        "Single-purpose API to refine a vague defining objective into a clearer, testable one.\n\n"
        "Endpoint: `/recommendation`\n\n"
        "- Required: `objective` (string)\n"
        "- Optional: `context` object (and all fields inside it are optional).\n\n"
        "Authentication: requires `X-API-Key` header."
    ),
    version="2.0.0",
)


@app.get("/", include_in_schema=False)
async def root():
    return {"status": "ok", "message": "Objective Recommendation API"}


@app.post(
    "/recommendation",
    response_model=SimpleRecommendResponse,
    summary="Recommend clearer defining objective",
    dependencies=[Depends(verify_api_key)],
)
async def recommendation(payload: SimpleObjectiveRequest):
    """
    Lightweight endpoint for a single defining objective.

    **Required field:**
    - `objective` (string)

    **Optional field:**
    - `context` (object). You may omit it entirely.

    **All fields inside `context` are optional:**
    - `persona`
    - `domain`
    - `instructions`
    - `satisfactionCriteria`
    - `extraNotes`
    """
    try:
        result = call_bedrock_simple(payload)
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
