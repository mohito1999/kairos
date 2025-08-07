from fastapi.security import OAuth2PasswordBearer

# This tells FastAPI to look for a token in the "Authorization" header
# with the value "Bearer <token>". The tokenUrl is not used for validation,
# but is required for OpenAPI spec compliance.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")