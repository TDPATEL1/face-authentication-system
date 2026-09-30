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


## PostgreSQL Backup and Restore

The PostgreSQL database can be backed up and restored safely using Docker Compose. The procedure below uses PostgreSQL's **custom dump format** (`-Fc`).

> **Important:** Do not use PowerShell `>` redirection with `pg_dump -Fc`. Custom-format dumps are binary files, and PowerShell redirection can corrupt the dump. Create the dump inside the PostgreSQL container first, then copy it to the host using `docker cp`.

### 1. Create the backup directory

From the project root:

```powershell
New-Item -ItemType Directory -Force .\backups
```

The `backups/` directory should be excluded from Git.

Verify:

```powershell
git check-ignore -v .\backups\face_auth_backup.dump
```

### 2. Create a PostgreSQL backup

Create the custom-format dump inside the PostgreSQL container:

```powershell
docker compose exec postgres pg_dump -U postgres -d user_auth_db -Fc -f /tmp/face_auth_backup.dump
```

This does not modify the database.

### 3. Verify the backup inside the container

Before copying the backup to the host, verify that PostgreSQL can read it:

```powershell
docker compose exec postgres pg_restore -l /tmp/face_auth_backup.dump
```

The output should contain:

* `Format: CUSTOM`
* PostgreSQL database/version information
* `users`
* `face_templates`
* `audit_logs`
* `alembic_version`
* table data
* indexes
* constraints
* foreign keys

### 4. Copy the backup to the host

Copy the verified binary dump to the project's backup directory:

```powershell
docker cp user-auth-api-postgres-1:/tmp/face_auth_backup.dump .\backups\face_auth_backup.dump
```

Verify the file:

```powershell
Get-Item .\backups\face_auth_backup.dump | Select-Object Name,Length,LastWriteTime
```

The file size should be greater than zero.

### 5. Test the backup by restoring to a temporary database

**Do not restore over `user_auth_db` for a backup test.**

Create a separate temporary database:

```powershell
docker compose exec postgres psql -U postgres -d postgres -c "CREATE DATABASE face_auth_restore_test;"
```

Restore the backup:

```powershell
docker compose exec postgres pg_restore -U postgres -d face_auth_restore_test --exit-on-error /tmp/face_auth_backup.dump
```

A successful command returns to the PowerShell prompt without an error.

### 6. Verify the restored database

List the restored tables:

```powershell
docker compose exec postgres psql -U postgres -d face_auth_restore_test -c "\dt"
```

Expected tables include:

```text
alembic_version
audit_logs
face_templates
users
```

Check application data:

```powershell
docker compose exec postgres psql -U postgres -d face_auth_restore_test -c "SELECT 'users' AS table_name, COUNT(*) AS rows FROM users UNION ALL SELECT 'face_templates', COUNT(*) FROM face_templates UNION ALL SELECT 'audit_logs', COUNT(*) FROM audit_logs;"
```

Check the Alembic migration revision:

```powershell
docker compose exec postgres psql -U postgres -d face_auth_restore_test -c "SELECT * FROM alembic_version;"
```

For a complete verification, compare these results with the original database:

```powershell
docker compose exec postgres psql -U postgres -d user_auth_db -c "SELECT 'users' AS table_name, COUNT(*) AS rows FROM users UNION ALL SELECT 'face_templates', COUNT(*) FROM face_templates UNION ALL SELECT 'audit_logs', COUNT(*) FROM audit_logs;"
```

```powershell
docker compose exec postgres psql -U postgres -d user_auth_db -c "SELECT * FROM alembic_version;"
```

The restored database should have matching table structures, row counts, and Alembic revision.

### 7. Remove the temporary restore database

After successful verification:

```powershell
docker compose exec postgres psql -U postgres -d postgres -c "DROP DATABASE face_auth_restore_test;"
```

Remove the temporary copy inside the PostgreSQL container:

```powershell
docker compose exec postgres rm -f /tmp/face_auth_backup.dump
```

The host backup remains available at:

```text
backups/face_auth_backup.dump
```

### 8. Restore a backup during an actual recovery

If the production/application database needs to be restored, first stop application traffic and ensure that the target database is backed up if required.

Copy the backup into the PostgreSQL container:

```powershell
docker cp .\backups\face_auth_backup.dump user-auth-api-postgres-1:/tmp/face_auth_backup.dump
```

For a full database replacement, the target database must be recreated or otherwise emptied before restoration. Do **not** run destructive commands unless the database replacement is intentionally planned and the existing database has been backed up.

Example recovery workflow:

```powershell
docker compose exec postgres psql -U postgres -d postgres -c "DROP DATABASE user_auth_db;"
docker compose exec postgres psql -U postgres -d postgres -c "CREATE DATABASE user_auth_db;"
docker compose exec postgres pg_restore -U postgres -d user_auth_db --exit-on-error /tmp/face_auth_backup.dump
```

> **Warning:** The `DROP DATABASE` command is destructive. It permanently removes the current `user_auth_db`. Always verify that the backup is valid before performing this operation.

After restoration, verify:

```powershell
docker compose exec postgres psql -U postgres -d user_auth_db -c "\dt"
```

Then verify the API and database connection:

```powershell
docker compose exec face-auth-api python -c "from app.core.database import engine; engine.connect(); print('DATABASE OK')"
```

```powershell
curl.exe -k https://127.0.0.1/health
```

Expected health response:

```json
{"status":"ok"}
```

### Backup Safety Notes

* Never commit database dump files to Git.
* The `backups/` directory is excluded through `.gitignore`.
* PostgreSQL dumps may contain user records, audit logs, and encrypted biometric face templates.
* Store production backups in a protected backup location with appropriate access controls.
* Keep encryption keys and database credentials separate from backup files.
* Never use `docker compose down -v` as part of routine backup or restore testing because `-v` can remove PostgreSQL volumes.
* Prefer testing a backup by restoring it to a separate temporary database before relying on it for disaster recovery.
* After backup operations, verify the Git working tree:

```powershell
git status --short
```

A clean result means no tracked or untracked project changes are present.
