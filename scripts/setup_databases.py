"""
Helper script to create the three PostgreSQL databases for each bank.
Run this before loading data.

Usage:
    python scripts/setup_databases.py
"""

import os
import sys

import psycopg2
from dotenv import load_dotenv


DATABASES = ["cbe_reviews", "boa_reviews", "dashen_reviews"]


def create_databases(admin_password: str = "123") -> None:
    """Create the three bank review databases if they don't exist."""
    
    # Connect to default 'postgres' database to create new databases
    try:
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            user="postgres",
            password=admin_password,
            database="postgres",
        )
        conn.autocommit = True
        cursor = conn.cursor()

        for db_name in DATABASES:
            # Check if database exists
            cursor.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s", (db_name,)
            )
            exists = cursor.fetchone()

            if exists:
                print(f"Database '{db_name}' already exists.")
            else:
                cursor.execute(f'CREATE DATABASE "{db_name}"')
                print(f"Created database '{db_name}'.")

        cursor.close()
        conn.close()
        print("\n✓ All databases ready.")

    except psycopg2.Error as e:
        print(f"Error connecting to PostgreSQL: {e}")
        sys.exit(1)


def apply_schema(admin_password: str = "123") -> None:
    """Apply schema.sql to each database."""
    import subprocess
    from pathlib import Path

    schema_path = Path(__file__).resolve().parents[1] / "sql" / "schema.sql"
    
    if not schema_path.exists():
        print(f"Schema file not found at {schema_path}")
        return

    print("\nApplying schema to databases...")
    
    for db_name in DATABASES:
        try:
            # Use psql to run schema file
            env = os.environ.copy()
            env["PGPASSWORD"] = admin_password
            
            result = subprocess.run(
                [
                    "psql",
                    "-h", "localhost",
                    "-U", "postgres",
                    "-d", db_name,
                    "-f", str(schema_path),
                ],
                env=env,
                capture_output=True,
                text=True,
            )
            
            if result.returncode == 0:
                print(f"  ✓ Applied schema to '{db_name}'")
            else:
                print(f"  ✗ Error applying schema to '{db_name}': {result.stderr}")
                
        except FileNotFoundError:
            print("  ✗ 'psql' command not found. Please install PostgreSQL client tools.")
            print("    Alternatively, run manually:")
            print(f"      psql -U postgres -d {db_name} -f sql/schema.sql")
            break


if __name__ == "__main__":
    load_dotenv()
    
    # Get password from environment or use default
    password = os.getenv("POSTGRES_PASSWORD", "123")
    
    print("Setting up PostgreSQL databases for bank reviews...")
    print(f"Using password: {'*' * len(password)}")
    
    create_databases(password)
    apply_schema(password)
    
    print("\nNext steps:")
    print("1. Copy .env.example to .env")
    print("2. Run: python scripts/scrape_reviews.py")
    print("3. Run: python scripts/preprocess_reviews.py")
    print("4. Run: python scripts/load_to_postgres.py")
