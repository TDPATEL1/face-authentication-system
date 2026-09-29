# Docker deployment

Stack: FastAPI, PostgreSQL 18, Alembic (runs in the API container before Uvicorn), Nginx (HTTP→HTTPS on ports 80 and 443).

There are no fixed `container_name` values. Database files live in the named volume `postgres_data`. Schema is created by Alembic, not a dump.

## One-time setup

1. Create both env files from the example (never commit them):

   ```powershell
   copy .env.example .env
   copy .env.example .env.docker
   ```

   Put the same strong `POSTGRES_PASSWORD` in `.env` and in `DATABASE_URL` inside `.env.docker`.
   In `.env.docker`, `DATABASE_URL` must use hostname `postgres`.

   Compose interpolates `POSTGRES_PASSWORD` from `.env`. The API reads `.env.docker`.

2. Generate a local development TLS certificate (do not commit `server.key`):

   ```powershell
   .\docker\generate-dev-certs.ps1
   ```

3. If you will open the API as something other than `localhost` / `127.0.0.1`, add that host to `ALLOWED_HOSTS` in `.env.docker`.

## Run

```powershell
docker compose config --quiet
docker compose build
docker compose up -d
docker compose ps
```

Existing Postgres data is kept in the named volume. Do not run `docker compose down -v` unless you intend to delete it.

Do not paste `docker compose config` (without `--quiet`) into chat or tickets; it prints secrets.

## Check

```powershell
docker compose exec face-auth-api python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/health').read().decode())"
docker compose exec face-auth-api alembic current
curl.exe -I http://127.0.0.1/health
curl.exe -k https://127.0.0.1/health
curl.exe -k https://127.0.0.1/docs
```

Nginx is the public entrypoint. It proxies `/health`, `/docs`, `/api/v1/auth/*`, `/api/v1/face/*` (enroll, login, liveness), and other API routes to `face-auth-api:8000`. The API port is not published on the host.

Browsers will warn about the self-signed certificate. That is expected for local certs.
