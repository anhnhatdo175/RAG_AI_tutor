"""The issued-task store is shared by FastAPI's worker threads."""
from __future__ import annotations

import threading

from lmsrag.application import _IssuedTaskStore


def test_store_works_from_other_threads(tmp_path):
    store = _IssuedTaskStore(tmp_path / "issued.sqlite")
    errors: list[BaseException] = []
    task_ids: list[str] = []

    def worker(user_id: int) -> None:
        try:
            task_id = store.issue(user_id, f"item-{user_id}", {"prompt": "p"}, {"status": "ok"})
            task_ids.append(task_id)
            assert store.get_open(user_id)[0] == task_id
            store.begin_submission(task_id, f"event-{user_id}", "42", 10.0)
            store.complete(task_id, "graded")
            assert store.get(task_id, user_id)[1] == "graded"
        except BaseException as exc:  # noqa: BLE001 - surfaced in the main thread
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(user_id,)) for user_id in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    store.close()

    assert errors == []
    assert len(set(task_ids)) == 8
