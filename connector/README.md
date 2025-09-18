# Minima Connector

A lightweight HTTP service that exposes the Minima indexer's capabilities for external tools (e.g., ChatGPT Custom Connector/Actions).

It forwards requests to the internal `indexer` service and returns the results unchanged.

## Endpoints

- POST `/query`
  - body: `{ "query": "string" }`
  - returns: `{ "result": { "links": string[], "output": string } }` or `{ "error": string }`

- POST `/embedding`
  - body: `{ "query": "string" }`
  - returns: `{ "result": number[] }` or `{ "error": string }`

- GET `/search`
  - query: `?q=your+search`
  - returns: `{ "items": [ { "title": string, "url": string, "snippet": string } ] }`

- GET `/health` → `{ "status": "ok" }`
- GET `/.well-known/openapi.json` → copy of OpenAPI schema
- GET `/.well-known/ai-plugin.json` → ChatGPT plugin/connector manifest
- GET `/logo.svg` → simple logo for the manifest

## Environment

- `INDEXER_BASE` (default: `http://indexer:8000`) – URL of indexer inside Docker network
- `PUBLIC_BASE_URL` (optional, default: `http://localhost:8004`) – Used to populate OpenAPI `servers` entry for external tools
### OAuth2
- `OAUTH_CLIENT_ID` (default: `minima-chatgpt-public`)
- `OAUTH_ISSUER` (default: `minima-connector`)
- `OAUTH_JWT_SECRET` (auto-generated if not set; set for stable tokens)
- `OAUTH_TOKEN_TTL` (default: `3600` seconds)
- `OAUTH_CODE_TTL` (default: `300` seconds)
- `OAUTH_DEMO_USER` / `OAUTH_DEMO_PASSWORD` – demo credentials for the auth form

## Run locally

This service is wired into the compose file. Once the stack is up, you can hit:

- Query: `POST http://localhost:8004/query` with `{ "query": "your text" }`
- OpenAPI: `GET http://localhost:8004/openapi.json`
- OAuth authorize: `GET http://localhost:8004/oauth/authorize?client_id=minima-chatgpt-public&redirect_uri=http://localhost:8004/health&response_type=code&scope=basic`
- OAuth token: `POST http://localhost:8004/oauth/token` with form fields `grant_type=authorization_code`, `code`, `redirect_uri`, `client_id`, `code_verifier` (if PKCE used)

## Notes

This is not a full ChatGPT Custom Connector manifest. ChatGPT can ingest OpenAPI directly; you can point it at `http://localhost:8004/openapi.json` or `/.well-known/openapi.json`.
OAuth2 (Authorization Code + PKCE) is implemented with a simple username/password form for demo and local setups; replace with your own identity provider in production.
