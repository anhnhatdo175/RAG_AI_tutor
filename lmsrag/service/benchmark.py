"""Measure API and per-stage p50/p95 latency for repeated recommendations."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import time
from datetime import datetime, timezone
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from sqlalchemy import select

from lmsrag.service.db import create_database, decision_log


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile from an empty sample")
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return round(ordered[index], 3)


def run_benchmark(
    *,
    base_url: str,
    token: str,
    database_url: str,
    learner_id_salt: str,
    user_id: str,
    skill_tag: str | None,
    request_count: int,
) -> dict[str, object]:
    if request_count < 1:
        raise ValueError("request_count must be positive")
    if len(token) < 32 or len(learner_id_salt) < 32:
        raise ValueError("SERVICE_TOKEN and LEARNER_ID_SALT must be at least 32 characters")

    learner_hash = hmac.new(
        learner_id_salt.encode("utf-8"),
        f"learner:{user_id.strip()}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    request_body: dict[str, object] = {"user_id": user_id, "k": 5}
    if skill_tag:
        request_body["skill_tag"] = skill_tag
    payload = json.dumps(request_body).encode("utf-8")

    database = create_database(database_url)
    started_at = datetime.now(timezone.utc)
    elapsed_ms = []
    try:
        for _ in range(request_count):
            started = time.perf_counter()
            request = Request(
                f"{base_url.rstrip('/')}/recommend",
                data=payload,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urlopen(request, timeout=120) as response:
                    response.read()
                    if not 200 <= response.status < 300:
                        raise RuntimeError(f"Recommendation request returned HTTP {response.status}")
            except HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"Recommendation request returned HTTP {exc.code}: {body}"
                ) from exc
            elapsed_ms.append((time.perf_counter() - started) * 1000)

        with database.engine.connect() as connection:
            rows = connection.execute(
                select(
                    decision_log.c.stage_latency_ms,
                    decision_log.c.latency_ms,
                )
                .where(
                    decision_log.c.endpoint == "recommend",
                    decision_log.c.learner_hash == learner_hash,
                    decision_log.c.created_at >= started_at,
                )
                .order_by(decision_log.c.created_at)
            ).fetchall()
        if len(rows) < request_count:
            raise RuntimeError(
                f"Expected {request_count} decision_log rows, found {len(rows)}"
            )
        selected_rows = rows[-request_count:]
        layers: dict[str, list[float]] = {}
        service_latency = []
        for row in selected_rows:
            service_latency.append(float(row.latency_ms))
            for name, value in (row.stage_latency_ms or {}).items():
                layers.setdefault(str(name), []).append(float(value))

        summary: dict[str, object] = {
            "endpoint": "POST /recommend",
            "requests": request_count,
            "api_latency_ms": {
                "p50": percentile(elapsed_ms, 0.50),
                "p95": percentile(elapsed_ms, 0.95),
            },
            "service_latency_ms": {
                "p50": percentile(service_latency, 0.50),
                "p95": percentile(service_latency, 0.95),
            },
            "stage_latency_ms": {
                name: {
                    "p50": percentile(values, 0.50),
                    "p95": percentile(values, 0.95),
                }
                for name, values in sorted(layers.items())
            },
        }
        return summary
    finally:
        database.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--skill-tag")
    parser.add_argument("--requests", type=int, default=200)
    args = parser.parse_args()
    required_environment = {
        "SERVICE_TOKEN": os.getenv("SERVICE_TOKEN"),
        "DATABASE_URL": os.getenv("DATABASE_URL"),
        "LEARNER_ID_SALT": os.getenv("LEARNER_ID_SALT"),
    }
    missing = [key for key, value in required_environment.items() if not value]
    if missing:
        raise RuntimeError("Missing environment variables: " + ", ".join(missing))
    result = run_benchmark(
        base_url=args.base_url,
        token=str(required_environment["SERVICE_TOKEN"]),
        database_url=str(required_environment["DATABASE_URL"]),
        learner_id_salt=str(required_environment["LEARNER_ID_SALT"]),
        user_id=args.user_id,
        skill_tag=args.skill_tag,
        request_count=args.requests,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
