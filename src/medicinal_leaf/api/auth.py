"""Bearer-token and API-key authentication.

Two ways in, for two kinds of caller:

* **A person** posts a username and password to ``/auth/token`` and gets a
  short-lived JWT. The browser sends it as ``Authorization: Bearer <token>``.
* **A script** sends a long-lived key as ``X-API-Key``, because curl in a
  cron job should not be doing a password exchange.

Nothing secret is stored in configuration. Passwords are held as bcrypt
hashes and API keys as SHA-256 digests, both supplied through the
environment — so a leaked config file yields nothing usable, and no
credential can reach git (NFR-4).

Deliberately not a user database: at this scale a dict from the environment
is honest, and swapping in Cognito or a real store means replacing
:func:`authenticate` and nothing else.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Literal

import bcrypt
import jwt

if TYPE_CHECKING:
    from medicinal_leaf.config.settings import Settings

logger = logging.getLogger(__name__)

#: Compared against when a username does not exist, so a missing user and a
#: wrong password take the same time and cannot be told apart by a stopwatch.
_DUMMY_HASH = b"$2b$12$C6UzMDM.H6dfI/f/IKcEe.6cwHMTHl.0EjqvQvdHXbzPqiXzp1Mdq"

#: Generated once per process when no key is configured (development only).
_EPHEMERAL_SECRET: str | None = None

PrincipalKind = Literal["user", "api_key", "anonymous"]


class AuthenticationError(Exception):
    """Credentials were absent, malformed, expired or wrong."""

    def __init__(self, message: str = "Not authenticated") -> None:
        super().__init__(message)
        self.message = message


class AuthConfigurationError(RuntimeError):
    """The service is configured in a way that cannot be served safely."""


@dataclass(frozen=True, slots=True)
class Principal:
    """Who is making a request."""

    name: str
    kind: PrincipalKind = "user"

    @property
    def is_anonymous(self) -> bool:
        return self.kind == "anonymous"


ANONYMOUS = Principal(name="anonymous", kind="anonymous")


# ── Hashing ──────────────────────────────────────────────────────────────


def hash_password(plain: str) -> str:
    """bcrypt hash of a password, for pasting into configuration."""
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("ascii")


def verify_password(plain: str, hashed: str) -> bool:
    """Check a password against its hash, tolerating a malformed hash."""
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        logger.warning("Stored password hash is malformed; rejecting.")
        return False


def hash_api_key(plain: str) -> str:
    """SHA-256 digest of an API key.

    A slow hash buys nothing here: API keys are long random strings, so there
    is no dictionary to attack the way there is with a chosen password.
    """
    return hashlib.sha256(plain.encode("utf-8")).hexdigest()


def generate_api_key() -> str:
    """A fresh key to hand to a machine client."""
    return f"leaf_{secrets.token_urlsafe(32)}"


# ── Secret resolution ────────────────────────────────────────────────────


def resolve_secret(settings: Settings) -> str:
    """The JWT signing key.

    Refuses to invent one in production: an ephemeral key would silently
    invalidate every token on each restart and differ between replicas, so a
    missing key there is a configuration error, not something to paper over.
    """
    if settings.auth.secret_key:
        return settings.auth.secret_key

    if settings.env == "production":
        raise AuthConfigurationError(
            "MLC_AUTH__SECRET_KEY must be set when auth is enabled in production."
        )

    global _EPHEMERAL_SECRET
    if _EPHEMERAL_SECRET is None:
        _EPHEMERAL_SECRET = secrets.token_urlsafe(48)
        logger.warning(
            "No MLC_AUTH__SECRET_KEY configured; generated an ephemeral one. "
            "Tokens will not survive a restart. Set a real key before deploying."
        )
    return _EPHEMERAL_SECRET


# ── Tokens ───────────────────────────────────────────────────────────────


def create_access_token(subject: str, settings: Settings) -> tuple[str, int]:
    """Sign a token for ``subject``. Returns ``(token, seconds_until_expiry)``."""
    expires_in = settings.auth.access_token_expire_minutes * 60
    issued = datetime.now(UTC)
    payload = {
        "sub": subject,
        "iat": issued,
        "exp": issued + timedelta(seconds=expires_in),
    }
    token = jwt.encode(payload, resolve_secret(settings), algorithm=settings.auth.algorithm)
    return token, expires_in


def decode_access_token(token: str, settings: Settings) -> str:
    """Return the subject of a valid token, or raise."""
    try:
        payload = jwt.decode(
            token,
            resolve_secret(settings),
            algorithms=[settings.auth.algorithm],
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthenticationError("Token has expired.") from exc
    except jwt.InvalidTokenError as exc:
        # Never echo the decoder's reason back to the caller — it tells an
        # attacker which part of a forged token was wrong.
        logger.info("Rejected token: %s", exc)
        raise AuthenticationError("Invalid token.") from exc

    subject = payload.get("sub")
    if not subject:
        raise AuthenticationError("Invalid token.")
    return str(subject)


# ── Credential checks ────────────────────────────────────────────────────


def authenticate(username: str, password: str, settings: Settings) -> Principal:
    """Verify a username and password, or raise.

    The same error and the same amount of work either way, so an unknown
    username is indistinguishable from a wrong password.
    """
    stored = settings.auth.users.get(username)
    if stored is None:
        # Burn an equivalent bcrypt check so timing reveals nothing.
        bcrypt.checkpw(password.encode("utf-8"), _DUMMY_HASH)
        raise AuthenticationError("Incorrect username or password.")

    if not verify_password(password, stored):
        raise AuthenticationError("Incorrect username or password.")

    return Principal(name=username, kind="user")


def authenticate_api_key(key: str, settings: Settings) -> Principal:
    """Verify an API key against the configured digests."""
    candidate = hash_api_key(key)
    for known in settings.auth.api_key_sha256:
        # compare_digest so a partial match is not detectable by timing.
        if hmac.compare_digest(candidate, known.strip().lower()):
            return Principal(name=f"key:{candidate[:8]}", kind="api_key")
    raise AuthenticationError("Invalid API key.")


def principal_from_credentials(
    settings: Settings,
    *,
    bearer_token: str | None = None,
    api_key: str | None = None,
) -> Principal:
    """Resolve whoever is calling, from whichever credential they sent."""
    if not settings.auth.enabled:
        return ANONYMOUS

    if api_key:
        return authenticate_api_key(api_key, settings)

    if bearer_token:
        return Principal(name=decode_access_token(bearer_token, settings), kind="user")

    raise AuthenticationError("Not authenticated.")


def warn_if_unprotected(settings: Settings) -> None:
    """Log loudly about configurations that leave the service open."""
    if not settings.auth.enabled:
        logger.warning(
            "Authentication is DISABLED. Anyone who can reach this port can "
            "upload archives and consume compute."
        )
        return

    if not settings.auth.users and not settings.auth.api_key_sha256:
        logger.error(
            "Authentication is enabled but no users or API keys are configured — "
            "every request will be rejected. Set MLC_AUTH__USERS or "
            "MLC_AUTH__API_KEY_SHA256. Generate values with `leaf-hash`."
        )


# ── Credential helper CLI ────────────────────────────────────────────────


def main() -> None:
    """``leaf-hash`` — produce config values without writing secrets to disk.

    Reads the password interactively so it never lands in shell history.
    """
    import argparse
    import getpass

    parser = argparse.ArgumentParser(
        prog="leaf-hash",
        description="Generate credentials for MLC_AUTH__USERS and MLC_AUTH__API_KEY_SHA256.",
    )
    parser.add_argument("--api-key", action="store_true", help="Generate an API key instead.")
    parser.add_argument("--username", default="admin", help="Username for the hashed password.")
    args = parser.parse_args()

    if args.api_key:
        key = generate_api_key()
        print("\nAPI key (shown once — store it now):")
        print(f"  {key}")
        print("\nAdd the digest to your environment:")
        print(f'  MLC_AUTH__API_KEY_SHA256=["{hash_api_key(key)}"]')
        return

    password = getpass.getpass("Password: ")
    if password != getpass.getpass("Confirm: "):
        raise SystemExit("Passwords did not match.")
    if len(password) < 12:
        print("Warning: shorter than 12 characters.")

    print("\nAdd to your environment:")
    print(f'  MLC_AUTH__USERS={{"{args.username}":"{hash_password(password)}"}}')
    print("\nAnd a signing key:")
    print(f"  MLC_AUTH__SECRET_KEY={secrets.token_urlsafe(48)}")


if __name__ == "__main__":
    main()
