import os
import time
import hmac
import base64
import hashlib
import logging
import secrets
from typing import Any, Dict, Optional

import jwt
from fastapi import APIRouter, HTTPException, Request, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import OAuth2AuthorizationCodeBearer

logger = logging.getLogger("connector.oauth")


# Env/config
OAUTH_CLIENT_ID = os.getenv("OAUTH_CLIENT_ID", "minima-chatgpt-public")
OAUTH_ISSUER = os.getenv("OAUTH_ISSUER", "minima-connector")
OAUTH_JWT_SECRET = os.getenv("OAUTH_JWT_SECRET", secrets.token_urlsafe(32))
OAUTH_TOKEN_TTL = int(os.getenv("OAUTH_TOKEN_TTL", "3600"))
OAUTH_CODE_TTL = int(os.getenv("OAUTH_CODE_TTL", "300"))
OAUTH_USERNAME = os.getenv("OAUTH_DEMO_USER", "user")
OAUTH_PASSWORD = os.getenv("OAUTH_DEMO_PASSWORD", "pass")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://localhost:8004")


router = APIRouter()


class InMemoryAuthStore:
    def __init__(self):
        self.codes: Dict[str, Dict[str, Any]] = {}

    def save_code(self, code: str, data: Dict[str, Any]):
        self.codes[code] = data

    def consume_code(self, code: str) -> Optional[Dict[str, Any]]:
        data = self.codes.pop(code, None)
        return data


auth_store = InMemoryAuthStore()


def _b64url_sha256(value: str) -> str:
    digest = hashlib.sha256(value.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def issue_access_token(sub: str, scope: str, client_id: str) -> Dict[str, Any]:
    now = int(time.time())
    payload = {
        "iss": OAUTH_ISSUER,
        "sub": sub,
        "aud": client_id,
        "scope": scope,
        "iat": now,
        "exp": now + OAUTH_TOKEN_TTL,
    }
    token = jwt.encode(payload, OAUTH_JWT_SECRET, algorithm="HS256")
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": OAUTH_TOKEN_TTL,
        "scope": scope,
    }


oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl=f"{PUBLIC_BASE_URL}/oauth/authorize",
    tokenUrl=f"{PUBLIC_BASE_URL}/oauth/token",
    scopes={"basic": "Basic access to Minima connector"},
    auto_error=True,
)


def verify_token(token: str) -> Dict[str, Any]:
    try:
        payload = jwt.decode(
            token,
            OAUTH_JWT_SECRET,
            algorithms=["HS256"],
            audience=OAUTH_CLIENT_ID,
            options={"require": ["exp", "iat"]},
        )
        return payload
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")


@router.get("/oauth/authorize", response_class=HTMLResponse, include_in_schema=False)
async def oauth_authorize(
    client_id: str,
    redirect_uri: str,
    response_type: str = "code",
    scope: str = "basic",
    state: str = "",
    code_challenge: Optional[str] = None,
    code_challenge_method: str = "S256",
):
    if response_type != "code":
        raise HTTPException(status_code=400, detail="unsupported_response_type")
    if client_id != OAUTH_CLIENT_ID:
        raise HTTPException(status_code=400, detail="invalid_client_id")
    form_html = f"""
    <html><body>
    <h2>Minima Connector Login</h2>
    <form method="post" action="/oauth/authorize">
        <input type="hidden" name="client_id" value="{client_id}" />
        <input type="hidden" name="redirect_uri" value="{redirect_uri}" />
        <input type="hidden" name="response_type" value="{response_type}" />
        <input type="hidden" name="scope" value="{scope}" />
        <input type="hidden" name="state" value="{state}" />
        <input type="hidden" name="code_challenge" value="{code_challenge or ''}" />
        <input type="hidden" name="code_challenge_method" value="{code_challenge_method}" />
        <label>Username: <input name="username" /></label><br/>
        <label>Password: <input type="password" name="password" /></label><br/>
        <button type="submit">Authorize</button>
    </form>
    </body></html>
    """
    return HTMLResponse(content=form_html)


@router.post("/oauth/authorize", include_in_schema=False)
async def oauth_authorize_post(
    client_id: str = Form(...),
    redirect_uri: str = Form(...),
    response_type: str = Form("code"),
    scope: str = Form("basic"),
    state: str = Form(""),
    code_challenge: str = Form(""),
    code_challenge_method: str = Form("S256"),
    username: str = Form(...),
    password: str = Form(...),
):
    if response_type != "code":
        raise HTTPException(status_code=400, detail="unsupported_response_type")
    if client_id != OAUTH_CLIENT_ID:
        raise HTTPException(status_code=400, detail="invalid_client_id")
    if not (hmac.compare_digest(username, OAUTH_USERNAME) and hmac.compare_digest(password, OAUTH_PASSWORD)):
        raise HTTPException(status_code=401, detail="invalid_credentials")
    if code_challenge_method and code_challenge_method.upper() not in ("S256",):
        raise HTTPException(status_code=400, detail="unsupported_code_challenge_method")

    code = secrets.token_urlsafe(32)
    auth_store.save_code(
        code,
        {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "scope": scope,
            "state": state,
            "code_challenge": code_challenge,
            "created_at": int(time.time()),
        },
    )
    sep = "&" if ("?" in redirect_uri) else "?"
    return RedirectResponse(url=f"{redirect_uri}{sep}code={code}&state={state}", status_code=302)


@router.post("/oauth/token", include_in_schema=False)
async def oauth_token(request: Request):
    form = await request.form()
    grant_type = form.get("grant_type")
    code = form.get("code")
    redirect_uri = form.get("redirect_uri")
    client_id = form.get("client_id") or OAUTH_CLIENT_ID
    code_verifier = form.get("code_verifier")

    if grant_type != "authorization_code":
        raise HTTPException(status_code=400, detail="unsupported_grant_type")
    if client_id != OAUTH_CLIENT_ID:
        raise HTTPException(status_code=400, detail="invalid_client_id")

    data = auth_store.consume_code(code)
    if not data:
        raise HTTPException(status_code=400, detail="invalid_or_used_code")

    if data.get("redirect_uri") != redirect_uri:
        raise HTTPException(status_code=400, detail="redirect_uri_mismatch")

    stored_challenge = data.get("code_challenge")
    if stored_challenge:
        if not code_verifier:
            raise HTTPException(status_code=400, detail="missing_code_verifier")
        derived = _b64url_sha256(code_verifier)
        if derived != stored_challenge:
            raise HTTPException(status_code=400, detail="invalid_code_verifier")

    if int(time.time()) - int(data.get("created_at", 0)) > OAUTH_CODE_TTL:
        raise HTTPException(status_code=400, detail="code_expired")

    token = issue_access_token(sub=OAUTH_USERNAME, scope=data.get("scope", "basic"), client_id=client_id)
    return token


def inject_oauth_into_openapi(doc: Dict[str, Any]) -> Dict[str, Any]:
    comp = doc.setdefault("components", {}).setdefault("securitySchemes", {})
    comp.setdefault(
        "OAuth2",
        {
            "type": "oauth2",
            "flows": {
                "authorizationCode": {
                    "authorizationUrl": f"{PUBLIC_BASE_URL}/oauth/authorize",
                    "tokenUrl": f"{PUBLIC_BASE_URL}/oauth/token",
                    "scopes": {"basic": "Basic access to Minima connector"},
                }
            },
        },
    )
    doc.setdefault("security", [{"OAuth2": []}])
    return doc
