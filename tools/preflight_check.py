#!/usr/bin/env python3
"""
deploy.bat preflight check script
Exit codes:
  0  all OK
  1  PostgreSQL server unreachable
  2  pgvector extension install failed
  3  Feishu API credentials not configured
"""
import sys
import os

def check_db(configs_dir: str) -> int:
    """Check DB connection, auto-create database & pgvector if missing. Returns exit code."""
    import yaml
    import psycopg

    db_info_path = os.path.join(configs_dir, "db_info.yml")
    abs_path = os.path.abspath(db_info_path)
    print(f"  [DB] 读取配置: {abs_path}")
    with open(db_info_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f).get("database", {})

    host = cfg.get("host", "localhost")
    port = cfg.get("port", 5432)
    user = cfg.get("user", "postgres")
    pw   = cfg.get("password", "")
    db   = cfg.get("database", "")
    addr = f"{host}:{port}/{db}"

    # 1) Test if PG server is reachable
    try:
        mconn = psycopg.connect(
            host=host, port=port, dbname="postgres",
            user=user, password=pw, connect_timeout=5, autocommit=True
        )
    except Exception as e:
        print(f"  [FAIL] PostgreSQL server unreachable: {e}")
        return 1

    # 2) Check if target database exists, create if not
    cur = mconn.cursor()
    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db,))
    if cur.fetchone() is None:
        print(f"  [AUTO] Database '{db}' not found, creating...")
        cur.execute(f'CREATE DATABASE "{db}"')
        print(f"  [ OK ] Database '{db}' created")
    mconn.close()

    # 3) Connect to target database
    conn = psycopg.connect(
        host=host, port=port, dbname=db,
        user=user, password=pw, connect_timeout=5, autocommit=True
    )

    # 4) Ensure pgvector extension
    cur2 = conn.cursor()
    cur2.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
    if cur2.fetchone() is None:
        print("  [AUTO] pgvector extension not found, installing...")
        try:
            cur2.execute("CREATE EXTENSION IF NOT EXISTS vector")
            print("  [ OK ] pgvector extension installed")
        except Exception as e:
            print(f"  [FAIL] pgvector install failed: {e}")
            print("         Please install pgvector manually")
            conn.close()
            return 2

    conn.close()
    print(f"  [ OK ] PostgreSQL connected: {addr}")
    return 0


def check_feishu(configs_dir: str) -> int:
    """Check Feishu API credentials are configured. Returns exit code."""
    import yaml

    feishu_path = os.path.join(configs_dir, "feishu.yaml")
    with open(feishu_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f).get("feishu", {})

    # support both flat (feishu.app_id) and nested (feishu.app.app_id) structures
    app = cfg.get("app", {}) if isinstance(cfg.get("app"), dict) else {}
    aid  = cfg.get("app_id", "") or app.get("app_id", "")
    asec = cfg.get("app_secret", "") or app.get("app_secret", "")

    if not aid or aid == "cli_xxx" or not asec or asec == "xxx":
        print("  [FAIL] feishu.yaml: app_id / app_secret not configured")
        return 3

    print("  [ OK ] Feishu API credentials configured")
    return 0


def main():
    if len(sys.argv) < 2:
        print("Usage: preflight_check.py <configs_dir>")
        sys.exit(99)

    configs_dir = sys.argv[1]

    # DB check
    rc = check_db(configs_dir)
    if rc != 0:
        sys.exit(rc)

    # Feishu check
    rc = check_feishu(configs_dir)
    if rc != 0:
        sys.exit(rc)

    sys.exit(0)


if __name__ == "__main__":
    main()

