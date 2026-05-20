"""API key issuance and verification.

Plaintext keys are shown exactly once, at creation, and never persisted --
the same "hashed at rest, shown once" property CLAUDE.md states as a
security invariant. A plain SHA-256 hash is enough here: an API key is a
256-bit random token, not a low-entropy user password, so there is no
brute-force-by-guessing risk for a slow/salted hash to defend against --
the only thing that matters is that the plaintext isn't recoverable from
what's stored.
"""

import hashlib
import secrets
from dataclasses import dataclass

import psycopg

_TOKEN_BYTES = 32
_PREFIX_CHARS = 12


class InvalidApiKey(Exception):
    pass


@dataclass(frozen=True)
class CreatedApiKey:
    id: int
    name: str
    plaintext: str
    prefix: str


@dataclass(frozen=True)
class ApiKeyRecord:
    id: int
    name: str
    rate_limit_rpm: int


def _hash_key(plaintext: str) -> str:
    return hashlib.sha256(plaintext.encode()).hexdigest()


def generate_api_key() -> str:
    return f"cq_{secrets.token_urlsafe(_TOKEN_BYTES)}"


def create_api_key(conn: psycopg.Connection, name: str, rate_limit_rpm: int) -> CreatedApiKey:
    plaintext = generate_api_key()
    prefix = plaintext[:_PREFIX_CHARS]
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO api_keys (name, key_prefix, key_hash, rate_limit_rpm)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (name, prefix, _hash_key(plaintext), rate_limit_rpm),
        )
        key_id = cur.fetchone()[0]
    conn.commit()
    return CreatedApiKey(id=key_id, name=name, plaintext=plaintext, prefix=prefix)


def verify_api_key(conn: psycopg.Connection, plaintext: str) -> ApiKeyRecord:
    """Raises InvalidApiKey for an unknown, malformed, or revoked key --
    the caller (api/app.py's auth dependency) turns that into a 401 without
    needing to distinguish the three, which is deliberate: telling a caller
    WHICH way their key is wrong is a minor information leak for no real
    benefit.

    Updates last_used_at on every successful call, which means every
    authenticated request costs one extra write. Accepted at this project's
    scale; a busier deployment would batch or sample this rather than writing
    on every single request.
    """
    key_hash = _hash_key(plaintext)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, name, rate_limit_rpm FROM api_keys
             WHERE key_hash = %s AND revoked_at IS NULL
            """,
            (key_hash,),
        )
        row = cur.fetchone()
        if row is None:
            raise InvalidApiKey("unknown or revoked API key")
        cur.execute("UPDATE api_keys SET last_used_at = now() WHERE id = %s", (row[0],))
    conn.commit()
    return ApiKeyRecord(id=row[0], name=row[1], rate_limit_rpm=row[2])


def revoke_api_key(conn: psycopg.Connection, key_id: int) -> None:
    with conn.cursor() as cur:
        cur.execute("UPDATE api_keys SET revoked_at = now() WHERE id = %s", (key_id,))
    conn.commit()
