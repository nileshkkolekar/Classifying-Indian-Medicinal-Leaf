import { useState, type FormEvent } from "react";

import { ApiError, login } from "../api";

interface Props {
  onAuthenticated: () => void;
}

export default function Login({ onAuthenticated }: Props) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await login(username, password);
      onAuthenticated();
    } catch (exception) {
      // The server deliberately gives one message for a wrong password and an
      // unknown user alike; pass it through rather than guessing at a nicer one.
      setError(
        exception instanceof ApiError
          ? exception.message
          : "Could not reach the server.",
      );
      setBusy(false);
    }
  }

  return (
    <div className="login">
      <form className="panel" onSubmit={handleSubmit}>
        <div>
          <h1 style={{ margin: 0, fontSize: "1.3rem" }}>Sign in</h1>
          <p className="hint" style={{ marginTop: 4 }}>
            Medicinal leaf classification
          </p>
        </div>

        <div>
          <label htmlFor="username">Username</label>
          <input
            id="username"
            type="text"
            autoComplete="username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
            autoFocus
          />
        </div>

        <div>
          <label htmlFor="password">Password</label>
          <input
            id="password"
            type="password"
            autoComplete="current-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        </div>

        {error && <div className="alert">{error}</div>}

        <button className="btn" type="submit" disabled={busy}>
          {busy ? "Signing in…" : "Sign in"}
        </button>
      </form>
    </div>
  );
}
