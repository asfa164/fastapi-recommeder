# Objective Recommender FastAPI (Vercel, Cognito + Bedrock)

This project exposes a FastAPI backend that wraps AWS Bedrock using AWS Cognito
(User Pool + Identity Pool) to recommend clearer defining objectives.

## Files

- app.py            # FastAPI app (entrypoint for Vercel)
- requirements.txt  # Python dependencies

## Endpoints

- `GET /` – Health/status
- `POST /recommend` – Takes a CompositeObjective JSON and returns recommended defining objectives.

## Auth

Send `X-API-Key` header. Configure the API key as `API_KEY` environment variable in Vercel.

## Required environment variables

- API_KEY

- COGNITO_REGION
- COGNITO_USER_POOL_ID
- COGNITO_CLIENT_ID
- COGNITO_CLIENT_SECRET (optional if your app client has no secret)
- COGNITO_USERNAME
- COGNITO_PASSWORD
- COGNITO_IDENTITY_POOL_ID

- BEDROCK_MODEL_ID
- BEDROCK_REGION (optional, falls back to COGNITO_REGION or eu-west-1)

## Deploy on Vercel

1. Push this folder to a Git repo.
2. Create a new Vercel project from the repo.
3. Do **not** add vercel.json; Vercel will auto-detect FastAPI from app.py.
4. Set environment variables in Vercel as above.
5. Deploy.

Vercel will run FastAPI as a single serverless function with Fluid compute.
