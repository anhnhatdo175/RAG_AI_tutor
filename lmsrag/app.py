"""Run the Layer 1-6 terminal learning loop: ``python -m lmsrag.app``."""
from __future__ import annotations

import argparse
import sys
import time

from lmsrag.application import LearningActivity, LearningPipeline, NoLearningContentError


def _display_activity(activity: LearningActivity) -> None:
    generation = activity.generation
    if generation.get("status") == "insufficient_context":
        print("Chưa đủ tài liệu phù hợp để tạo hoạt động học an toàn.")
        return
    task = generation["task"]
    print(f"\nBài tập [{activity.item_id}]")
    print(task["prompt"])
    if task.get("personalization_reason"):
        print(f"Gợi ý cá nhân hóa: {task['personalization_reason']}")
    if activity.status == "submitting":
        print("Đang khôi phục lần nộp trước...")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--skill", required=True, help="Canonical skill tag, e.g. Addition-and-Subtraction-Fractions")
    parser.add_argument("--query", help="Learning question or topic; defaults to --skill")
    parser.add_argument("--subject", default="Mathematics")
    parser.add_argument("--grade", default="lower_secondary")
    args = parser.parse_args(argv)

    try:
        pipeline = LearningPipeline.from_artifacts()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Không thể khởi tạo pipeline: {exc}", file=sys.stderr)
        return 2

    try:
        query = args.query or args.skill.replace("-", " ")
        print("Pipeline Layer 1-6 đã sẵn sàng. Nhập :quit để thoát.")
        while True:
            try:
                activity = pipeline.get_pending_activity(args.user_id)
                if activity is None:
                    activity = pipeline.prepare_activity(
                        user_id=args.user_id,
                        query=query,
                        target_subject=args.subject,
                        target_grade=args.grade,
                        target_skill=args.skill,
                    )
            except NoLearningContentError as exc:
                print(f"Không thể tạo bài: {exc}", file=sys.stderr)
                return 2
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                print(f"Lỗi pipeline: {exc}", file=sys.stderr)
                return 2

            if activity.generation.get("status") == "insufficient_context":
                _display_activity(activity)
                return 2
            _display_activity(activity)
            if activity.status == "submitting":
                result = pipeline.submit_answer(
                    user_id=args.user_id,
                    task_id=activity.task_id or "",
                )
            else:
                started = time.perf_counter()
                try:
                    response = input("Câu trả lời (:quit để lưu bài và thoát): ").strip()
                except EOFError:
                    print("\nĐã thoát; bài chưa nộp được giữ lại để tiếp tục lần sau.")
                    return 0
                if response == ":quit":
                    print("Đã lưu bài đang chờ; lần chạy sau có thể tiếp tục.")
                    return 0
                result = pipeline.submit_answer(
                    user_id=args.user_id,
                    task_id=activity.task_id or "",
                    response=response,
                    rt_ms=(time.perf_counter() - started) * 1000,
                )

            if result["status"] == "graded":
                print(f"Điểm: {result['score']:.2f}")
                if result["score"] < 1.0:
                    hints = activity.generation["task"].get("hints", [])
                    if hints:
                        print(f"Gợi ý: {hints[0]}")
                state = result["state"]
                print(f"Độ thành thạo hiện tại: {state.canonical_M.get(args.skill, 0.5):.2f}")
            else:
                print("Bài cần được giáo viên xem lại; trạng thái học viên chưa bị cập nhật.")
            print("\n--- Lượt học tiếp theo ---")
    except KeyboardInterrupt:
        print("\nĐã dừng; bài chưa nộp được giữ lại để tiếp tục lần sau.")
        return 0
    finally:
        pipeline.close()


if __name__ == "__main__":
    raise SystemExit(main())
