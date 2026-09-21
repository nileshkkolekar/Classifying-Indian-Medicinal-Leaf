"""Password hashing, token signing and credential checks.

bcrypt is deliberately slow, so this module hashes once at import and reuses
the result rather than paying for it per test.
"""

from __future__ import annotations

import time

import bcrypt
import jwt
import pytest

from medicinal_leaf.api.auth import (
    ANONYMOUS,
    AuthConfigurationError,
    AuthenticationError,
    authenticate,
    authenticate_api_key,
    create_access_token,
    decode_access_token,
    generate_api_key,
    hash_api_key,
    hash_password,
    principal_from_credentials,
    resolve_secret,
    verify_password,
)
from medicinal_leaf.config.settings import Settings

PASSWORD = "correct-horse-battery-staple"
PASSWORD_HASH = hash_password(PASSWORD)
SECRET = "test-signing-key-not-used-anywhere-real"


@pytest.fixture
def settings() -> Settings:
    config = Settings()
    config.auth.enabled = True
    config.auth.secret_key = SECRET
    config.auth.users = {"botanist": PASSWORD_HASH}
    config.auth.api_key_sha256 = []
    return config


# ── Password hashing ─────────────────────────────────────────────────────


def test_hash_is_not_the_password():
    assert PASSWORD not in PASSWORD_HASH
    assert PASSWORD_HASH.startswith("$2b$")


def test_correct_password_verifies():
    assert verify_password(PASSWORD, PASSWORD_HASH) is True


def test_wrong_password_does_not():
    assert verify_password("nearly-right", PASSWORD_HASH) is False


def test_hashes_are_salted():
    """Two hashes of the same password must differ, or a rainbow table wins."""
    assert hash_password(PASSWORD) != hash_password(PASSWORD)


def test_malformed_hash_is_rejected_not_raised():
    assert verify_password(PASSWORD, "not-a-bcrypt-hash") is False


# ── API keys ─────────────────────────────────────────────────────────────


def test_generated_keys_are_unique_and_prefixed():
    first, second = generate_api_key(), generate_api_key()
    assert first != second
    assert first.startswith("leaf_")
    assert len(first) > 32


def test_api_key_digest_is_stable():
    key = generate_api_key()
    assert hash_api_key(key) == hash_api_key(key)
    assert key not in hash_api_key(key)


def test_valid_api_key_is_accepted(settings):
    key = generate_api_key()
    settings.auth.api_key_sha256 = [hash_api_key(key)]

    principal = authenticate_api_key(key, settings)
    assert principal.kind == "api_key"


def test_unknown_api_key_is_rejected(settings):
    settings.auth.api_key_sha256 = [hash_api_key(generate_api_key())]
    with pytest.raises(AuthenticationError):
        authenticate_api_key(generate_api_key(), settings)


# ── Username and password ────────────────────────────────────────────────


def test_valid_credentials_identify_the_user(settings):
    principal = authenticate("botanist", PASSWORD, settings)
    assert principal.name == "botanist"
    assert principal.kind == "user"


def test_wrong_password_is_rejected(settings):
    with pytest.raises(AuthenticationError):
        authenticate("botanist", "wrong", settings)


def test_unknown_user_gives_the_same_error(settings):
    """A different message would confirm which usernames exist."""
    with pytest.raises(AuthenticationError) as unknown:
        authenticate("nobody", PASSWORD, settings)
    with pytest.raises(AuthenticationError) as wrong:
        authenticate("botanist", "wrong", settings)
    assert unknown.value.message == wrong.value.message


def test_unknown_user_still_runs_a_hash_comparison(settings, monkeypatch):
    """Returning early for a missing user would leak account existence.

    Asserted by observing that the comparison happens, rather than with a
    stopwatch: wall-clock assertions are flaky on a shared or virtualised
    CPU, and they test the clock as much as the code.
    """
    comparisons: list[bytes] = []
    real_checkpw = bcrypt.checkpw

    def counting_checkpw(password: bytes, hashed: bytes) -> bool:
        comparisons.append(hashed)
        return real_checkpw(password, hashed)

    monkeypatch.setattr("medicinal_leaf.api.auth.bcrypt.checkpw", counting_checkpw)

    with pytest.raises(AuthenticationError):
        authenticate("nobody", PASSWORD, settings)

    assert comparisons, "no bcrypt comparison ran for an unknown user"


# ── Tokens ───────────────────────────────────────────────────────────────


def test_token_round_trips(settings):
    token, expires_in = create_access_token("botanist", settings)
    assert decode_access_token(token, settings) == "botanist"
    assert expires_in == settings.auth.access_token_expire_minutes * 60


def test_token_signed_with_another_key_is_rejected(settings):
    token, _ = create_access_token("botanist", settings)
    settings.auth.secret_key = "a-completely-different-key"
    with pytest.raises(AuthenticationError, match="Invalid token"):
        decode_access_token(token, settings)


def test_expired_token_is_rejected(settings):
    settings.auth.access_token_expire_minutes = 1
    payload = {"sub": "botanist", "exp": int(time.time()) - 10}
    expired = jwt.encode(payload, SECRET, algorithm=settings.auth.algorithm)

    with pytest.raises(AuthenticationError, match="expired"):
        decode_access_token(expired, settings)


def test_garbage_token_is_rejected(settings):
    with pytest.raises(AuthenticationError):
        decode_access_token("not.a.token", settings)


def test_token_without_a_subject_is_rejected(settings):
    empty = jwt.encode({"exp": int(time.time()) + 600}, SECRET, algorithm="HS256")
    with pytest.raises(AuthenticationError):
        decode_access_token(empty, settings)


# ── Secret resolution ────────────────────────────────────────────────────


def test_configured_secret_is_used(settings):
    assert resolve_secret(settings) == SECRET


def test_development_invents_a_secret(settings):
    settings.auth.secret_key = ""
    settings.env = "development"
    assert len(resolve_secret(settings)) > 20


def test_production_refuses_to_invent_one(settings):
    """A per-replica random key would break tokens across instances."""
    settings.auth.secret_key = ""
    settings.env = "production"
    with pytest.raises(AuthConfigurationError, match="MLC_AUTH__SECRET_KEY"):
        resolve_secret(settings)


# ── Resolution from a request ────────────────────────────────────────────


def test_disabled_auth_lets_everyone_through(settings):
    settings.auth.enabled = False
    assert principal_from_credentials(settings) == ANONYMOUS


def test_no_credentials_is_rejected_when_enabled(settings):
    with pytest.raises(AuthenticationError, match="Not authenticated"):
        principal_from_credentials(settings)


def test_bearer_token_resolves(settings):
    token, _ = create_access_token("botanist", settings)
    assert principal_from_credentials(settings, bearer_token=token).name == "botanist"


def test_api_key_takes_precedence(settings):
    key = generate_api_key()
    settings.auth.api_key_sha256 = [hash_api_key(key)]
    principal = principal_from_credentials(settings, api_key=key, bearer_token="ignored")
    assert principal.kind == "api_key"
