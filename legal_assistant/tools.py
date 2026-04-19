from __future__ import annotations

from datetime import date, datetime, timedelta
import re


def legal_utility_tool(question: str) -> str:
    try:
        lower = question.lower()
        if "date" in lower or "today" in lower or "time" in lower:
            return f"Current date: {date.today().isoformat()}. Current time: {datetime.now().strftime('%H:%M')}."

        deadline_match = re.search(r"(\d+)\s*(day|days)", lower)
        if "deadline" in lower and deadline_match:
            days = int(deadline_match.group(1))
            target = date.today() + timedelta(days=days)
            return f"A {days}-day deadline from today falls on {target.isoformat()}."

        numbers = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", question)]
        if any(word in lower for word in ["calculate", "total", "fee", "pages", "cost"]) and numbers:
            return f"Calculation result: sum={sum(numbers):.2f}, count={len(numbers)}."

        return "Tool result: no matching utility action was found for this question."
    except Exception as exc:
        return f"Tool error: {exc}"

